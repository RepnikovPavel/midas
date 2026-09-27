"""Orchestrator: balanced simultaneous download + unpack (+convert) of PandaSet.

Two target disks. Sequences are assigned to disks by greedy LPT on compressed
size (byte balance). A pool of stream workers pulls sequence jobs; each worker
downloads its sequence's byte span in ordered chunks over HTTP Range and feeds
a SpanExtractor, so unpacking happens while downloading. Completed sequences
are handed to converter threads producing the npz sweep layout.

State is kept in a json file so interrupted runs resume at sequence/member
granularity (already extracted member files are skipped by SpanExtractor).
"""

import json
import os
import queue
import threading
import time
import traceback

from . import zipindex
from .streaming_unzip import SpanExtractor
from . import convert as convert_mod

LOCK = threading.Lock()


def _log(msg):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class State:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        try:
            with open(path, "r") as f:
                self.data = json.load(f)
        except OSError:
            self.data = {"downloaded": {}, "converted": {}, "failed": {}}

    def mark(self, kind, seq, info=True):
        with self.lock:
            self.data[kind][seq] = info
            self._save()

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)

    def is_done(self, kind, seq):
        with self.lock:
            return seq in self.data[kind]


def download_sequence(reader, seq, group, out_root, chunk_bytes, progress, tag=""):
    """Download one sequence span in order, streaming-extract to out_root."""
    start, end = group["span"]
    extractor = SpanExtractor(group["members"], start, out_root)
    pos = start
    last_log = time.time()
    while pos <= end:
        cend = min(pos + chunk_bytes - 1, end)
        data = reader.fetch(pos, cend)
        extractor.feed(data)
        progress.add(len(data))
        pos = cend + 1
        if time.time() - last_log > 60:
            _log(f"{tag}{seq}: {(pos - start) / 2**20:.0f}/{(end - start) / 2**20:.0f} MiB "
                 f"extracted_members={extractor.idx}")
            last_log = time.time()
    return extractor.finish()


class Progress:
    def __init__(self, total_bytes):
        self.total = total_bytes
        self.done = 0
        self.lock = threading.Lock()
        self.t0 = time.time()

    def add(self, n):
        with self.lock:
            self.done += n

    def summary(self):
        with self.lock:
            d = self.done
        dt = max(time.time() - self.t0, 1e-9)
        return (f"{d / 2**30:.1f}/{self.total / 2**30:.1f} GiB "
                f"({100 * d / max(self.total, 1):.1f}%) avg {d / dt / 2**20:.1f} MiB/s")


def run(url, disks, workdir, streams=16, chunk_mb=16, auth_header=None, proxy=None,
        do_convert=True, converters=2, only_seqs=None, index_cache=None):
    """
    disks: list of {"id": str, "raw": raw_root, "npz": npz_root}
    """
    os.makedirs(workdir, exist_ok=True)
    reader = zipindex.HttpRangedReader(url, auth_header=auth_header, proxy=proxy)
    _log("resolving download url (redirect chain + CDN IP pool)...")
    reader.size()  # warm resolve + IP pool before anything else
    _log("fetching zip central directory...")
    members = zipindex.fetch_member_index(reader, cache_path=index_cache
                                          or os.path.join(workdir, "zip_index.json"))
    groups = zipindex.group_by_sequence(members)
    # clamp spans to actual file size (last member's data end is an overestimate)
    fsize = reader.size()
    for g in groups.values():
        s, e = g["span"]
        g["span"] = (s, min(e, fsize - 1))
    _log(f"{len(members)} members, {len(groups)} sequences, file {fsize / 2**30:.1f} GiB")
    if only_seqs:
        groups = {s: g for s, g in groups.items() if s in set(only_seqs)}
        _log(f"restricted to {len(groups)} sequences")

    disk_ids = [d["id"] for d in disks]
    assign = zipindex.assign_sequences(groups, disk_ids)
    by_id = {d["id"]: d for d in disks}
    for d in disk_ids:
        gib = sum(groups[s]["csize"] for s in assign[d]) / 2**30
        _log(f"disk {d} ({by_id[d]['raw']}): {len(assign[d])} seqs, {gib:.1f} GiB compressed")

    state = State(os.path.join(workdir, "state.json"))
    total = sum(g["csize"] for g in groups.values())
    progress = Progress(total)

    job_q = queue.Queue()
    n_queued = 0
    # interleave disks so both HDDs download/convert in parallel
    per_disk = {}
    for d in disk_ids:
        per_disk[d] = [s for s in assign[d] if not state.is_done("downloaded", s)]
    while any(per_disk.values()):
        for d in disk_ids:
            if per_disk[d]:
                job_q.put((d, per_disk[d].pop(0)))
                n_queued += 1
    _log(f"to download: {n_queued} sequences ({state.data['downloaded'].__len__()} already done)")

    conv_q = queue.Queue()
    stop_flag = {"stop": False}

    def converter_worker():
        while not stop_flag["stop"]:
            try:
                d, seq = conv_q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                disk = by_id[d]
                seq_path = os.path.join(disk["raw"], seq)
                out_dir = os.path.join(disk["npz"], f"sweep_{seq}")
                if not state.is_done("converted", seq):
                    t0 = time.time()
                    st = convert_mod.convert_sequence(seq_path, out_dir)
                    state.mark("converted", seq, True)
                    _log(f"converted {seq} ({st['frames']} frames, "
                         f"{len(st['warnings'])} warn) in {time.time() - t0:.0f}s")
                    for w in st["warnings"][:5]:
                        _log(f"  warn {seq}: {w}")
            except Exception as e:  # noqa: BLE001
                _log(f"CONVERT FAIL {seq}: {e}")
                traceback.print_exc()
            finally:
                conv_q.task_done()

    def dl_worker(wid):
        while True:
            try:
                d, seq = job_q.get_nowait()
            except queue.Empty:
                return
            try:
                disk = by_id[d]
                g = groups[seq]
                t0 = time.time()
                last_err = None
                _log(f"[w{wid}] start {seq} -> {d} ({g['csize'] / 2**20:.0f} MiB span)")
                for attempt in range(2):
                    try:
                        written, skipped, nbytes = download_sequence(
                            reader, seq, g, disk["raw"], chunk_mb << 20, progress,
                            tag=f"[w{wid}] ")
                        last_err = None
                        break
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                        _log(f"[w{wid}] {seq} attempt {attempt + 1} failed: {e}")
                if last_err is not None:
                    raise last_err
                state.mark("downloaded", seq, True)
                dt = time.time() - t0
                _log(f"[w{wid}] {seq}: {nbytes / 2**20:.0f} MiB in {dt:.0f}s "
                     f"({nbytes / dt / 2**20:.1f} MiB/s) written={written} skipped={skipped} | "
                     f"total {progress.summary()}")
                if do_convert:
                    conv_q.put((d, seq))
            except Exception as e:  # noqa: BLE001
                state.mark("failed", seq, str(e))
                _log(f"[w{wid}] FAIL {seq}: {e}")
            finally:
                job_q.task_done()

    conv_threads = []
    if do_convert:
        # already-downloaded but not converted sequences -> convert queue
        for d in disk_ids:
            for seq in assign[d]:
                if state.is_done("downloaded", seq) and not state.is_done("converted", seq):
                    conv_q.put((d, seq))
        for _ in range(converters):
            t = threading.Thread(target=converter_worker, daemon=True)
            t.start()
            conv_threads.append(t)

    threads = []
    for wid in range(streams):
        t = threading.Thread(target=dl_worker, args=(wid,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if do_convert:
        conv_q.join()
        stop_flag["stop"] = True
        for t in conv_threads:
            t.join(timeout=5)

    n_fail = len(state.data["failed"])
    _log(f"DONE. downloaded={len(state.data['downloaded'])} "
         f"converted={len(state.data['converted'])} failed={n_fail}")
    return 0 if n_fail == 0 else 2
