"""Remote zip64 central-directory parsing over HTTP Range requests.

Lets us list all members of a huge remote zip without downloading it,
then fetch arbitrary byte spans later (per-sequence contiguous spans).

Uses `requests` so traffic can go through a SOCKS proxy (byeDPI) when the
direct route to the CDN is throttled.
"""

import json
import socket
import struct
import threading
import time
import urllib.parse

import requests
import urllib3.util.connection as _urllib3_conn

from .netpool import CDNIPPool

# The server network has broken IPv6 (connects hang until timeout) and Python
# tries AAAA addresses serially before IPv4 -> every HTTPS call stalls for
# minutes. Force IPv4 for all requests/urllib3 connections.
_urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CDH_SIG = b"PK\x01\x02"


class HttpRangedReader:
    """Range reader that resolves redirects once and reuses the signed URL.

    Redirects are walked manually (no body reads): some CDNs answer 200 to a
    ranged request arriving through a redirect chain and stream the whole
    40+ GiB file. The final URL is probed once with a 1-byte range (206 check),
    then all range GETs go to it directly.
    """

    def __init__(self, url, auth_header=None, proxy=None, retries=8, timeout=60,
                 use_ip_pool=True):
        self.url = url
        self.auth_header = auth_header
        self.retries = retries
        self.timeout = timeout
        self.use_ip_pool = use_ip_pool
        self._size = None
        self._resolved = None
        self._proxies = {"http": proxy, "https": proxy} if proxy else None
        self._pool = None
        self._resolve_lock = threading.Lock()

    def _headers(self, start=None, end=None):
        h = {}
        if self.auth_header:
            h["Authorization"] = self.auth_header
        if start is not None:
            h["Range"] = f"bytes={start}-{end}"
        return h

    def resolve(self):
        """Resolve redirects manually; cache final CDN url + total size."""
        with self._resolve_lock:
            if self._resolved is not None and self._size is not None:
                return self._resolved
            return self._resolve_inner()

    def _resolve_inner(self):
        last = None
        for attempt in range(self.retries):
            try:
                url = self.url
                for _ in range(10):  # walk redirect chain manually
                    r = requests.get(url, headers=self._headers(),
                                     allow_redirects=False, timeout=min(self.timeout, 30),
                                     proxies=self._proxies, stream=True)
                    r.close()
                    if r.is_redirect or r.is_permanent_redirect:
                        url = requests.compat.urljoin(url, r.headers["Location"])
                        continue
                    if r.status_code == 200:
                        break
                    raise IOError(f"unexpected status {r.status_code} for {url[:80]}")
                self._resolved = url
                # total size: prefer the CDN IP pool (dodges throttled routes);
                # fall back to a plain 1-byte probe.
                if self.use_ip_pool and not self._proxies:
                    try:
                        self._maybe_start_pool(url)
                        if self._pool is not None:
                            u = urllib.parse.urlparse(url)
                            pq = u.path + ("?" + u.query if u.query else "")
                            self._size = self._pool.probe_size(pq)
                            return url
                    except Exception as e:  # noqa: BLE001
                        last = e
                r = requests.get(url, headers=self._headers(0, 0),
                                 allow_redirects=False, timeout=min(self.timeout, 30),
                                 proxies=self._proxies, stream=True)
                cr = r.headers.get("Content-Range")
                status = r.status_code
                r.close()
                if status != 206 or not cr:
                    raise IOError(f"range not honored on final url (status {status})")
                self._size = int(cr.split("/")[-1])
                self._maybe_start_pool(url)
                return url
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(2 ** attempt, 20))
        raise IOError(f"resolve failed: {last}")

    def size(self):
        if self._size is None:
            self.resolve()
        return self._size

    def _maybe_start_pool(self, final_url):
        """Start/refresh the CDN IP pool for the resolved host (direct mode only)."""
        if not self.use_ip_pool or self._proxies:
            return
        u = urllib.parse.urlparse(final_url)
        host = u.hostname
        origin = urllib.parse.urlparse(self.url).hostname
        if not host or host == origin:
            return
        pq = u.path + ("?" + u.query if u.query else "")
        if self._pool is None or self._pool.host != host:
            if self._pool is not None:
                self._pool.stop()
            self._pool = CDNIPPool(host, extra_headers=self._headers())
            self._pool.start(pq)
        else:
            self._pool._path_query_for_probe = pq

    def _fetch_via_pool(self, start, end, want):
        """One attempt through the IP pool. Raises on failure."""
        from .netpool import HttpStatusError
        if self._pool is None:
            raise IOError("no pool")
        ip = self._pool.pick()
        if ip is None:
            raise IOError("ip pool empty")
        u = urllib.parse.urlparse(self._resolved)
        pq = u.path + ("?" + u.query if u.query else "")
        try:
            return self._pool.fetch_range(ip, pq, start, end,
                                          timeout=min(self.timeout, 30))
        except HttpStatusError as e:
            if e.status in (401, 403):
                # signed url expired -> re-resolve; the ip is not at fault
                self._resolved = None
                self.resolve()
                raise IOError("re-resolved expired signed url")
            self._pool.report_bad(ip)
            raise
        except Exception:
            self._pool.report_bad(ip)
            raise

    def fetch(self, start, end):
        """Fetch inclusive byte range [start, end] with retries."""
        assert end >= start
        want = end - start + 1
        if self._resolved is None:
            self.resolve()
        last = None
        for attempt in range(self.retries):
            # preferred path: rotating probed CDN IPs
            if self._pool is not None and self._pool.ips:
                try:
                    return self._fetch_via_pool(start, end, want)
                except Exception as e:  # noqa: BLE001
                    last = e
                    continue
            try:
                r = requests.get(self._resolved, headers=self._headers(start, end),
                                 allow_redirects=False, timeout=self.timeout,
                                 proxies=self._proxies, stream=True)
                if r.status_code in (401, 403):
                    r.close()
                    self._resolved = None
                    self.resolve()
                    continue
                if r.status_code != 206:
                    r.close()
                    raise IOError(f"expected 206, got {r.status_code}")
                chunks = []
                got = 0
                for c in r.iter_content(1 << 20):
                    chunks.append(c)
                    got += len(c)
                    if got >= want:
                        break
                r.close()
                if got != want:
                    raise IOError(f"short read: {got}/{want}")
                return b"".join(chunks)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(2 ** attempt, 30))
                # re-resolve in case the signed url died mid-run
                try:
                    self._resolved = None
                    self.resolve()
                except Exception:  # noqa: BLE001
                    pass
        raise IOError(f"range {start}-{end} failed after {self.retries} tries: {last}")


def _parse_zip64_locator(tail, eocd_pos, file_size):
    loc_pos = tail.rfind(EOCD64_LOC_SIG, 0, eocd_pos)
    if loc_pos < 0:
        raise IOError("zip64 EOCD locator not found")
    _, _, eocd64_offset, _ = struct.unpack_from("<IIQI", tail, loc_pos)
    return eocd64_offset


def central_directory_info(reader):
    """Return (cd_offset, cd_size, n_entries)."""
    fsize = reader.size()
    tail_len = min(1 << 17, fsize)
    tail = reader.fetch(fsize - tail_len, fsize - 1)
    eocd_pos = tail.rfind(EOCD_SIG)
    if eocd_pos < 0:
        raise IOError("EOCD not found")
    (_, _, _, n_disk, n_total, cd_size, cd_offset, _) = struct.unpack_from("<IHHHHIIH", tail, eocd_pos)
    if n_total == 0xFFFF or cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        eocd64_offset = _parse_zip64_locator(tail, eocd_pos, fsize)
        head = reader.fetch(eocd64_offset, eocd64_offset + 55)
        if head[:4] != EOCD64_SIG:
            raise IOError("zip64 EOCD not found")
        (_, _, _, _, _, _, _, n_total, cd_size, cd_offset) = struct.unpack_from("<IQHHIIQQQQ", head, 0)
    return cd_offset, cd_size, n_total


def fetch_member_index(reader, cache_path=None):
    """Fetch and parse the central directory.

    Returns list of dicts: name, lho (local header offset), csize, usize, method, crc.
    Cached to cache_path (json) if given.
    """
    if cache_path:
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except OSError:
            pass
    cd_offset, cd_size, n_total = central_directory_info(reader)
    cd = reader.fetch(cd_offset, cd_offset + cd_size - 1)
    members = []
    offs = 0
    while offs < len(cd):
        if cd[offs:offs + 4] != CDH_SIG:
            raise IOError(f"bad central header at {offs}")
        (_, _, flag, method, _, _, crc, csize, usize,
         nlen, elen, clen, _, _, _, lho) = struct.unpack_from("<HHHHHHIIIHHHHHII", cd, offs + 4)
        name = cd[offs + 46:offs + 46 + nlen].decode("utf-8", "replace")
        extra = cd[offs + 46 + nlen:offs + 46 + nlen + elen]
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or lho == 0xFFFFFFFF:
            p = 0
            while p < len(extra):
                tag, sz = struct.unpack_from("<HH", extra, p)
                if tag == 0x0001:
                    raw = extra[p + 4:p + 4 + sz]
                    q = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack_from("<Q", raw, q)[0]
                        q += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack_from("<Q", raw, q)[0]
                        q += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack_from("<Q", raw, q)[0]
                    break
                p += 4 + sz
        members.append({
            "name": name, "lho": lho, "csize": csize, "usize": usize,
            "method": method, "crc": crc, "dd": bool(flag & 0x08),
        })
        offs += 46 + nlen + elen + clen
    if len(members) != n_total:
        raise IOError(f"central directory truncated: {len(members)} != {n_total}")
    if cache_path:
        import os
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(members, f)
        os.replace(tmp, cache_path)
    return members


def group_by_sequence(members, top_prefix="pandaset/"):
    """Group members by sequence id (second path component under top_prefix).

    Returns {seq_id: {"members": [...sorted by lho], "span": (start, end_inclusive), "csize": int}}
    Sequences are contiguous spans in this archive; span covers the whole byte range
    from the first member's local header to the end of the last member's data,
    so downloading the span yields every member of the sequence.
    """
    groups = {}
    for m in members:
        name = m["name"]
        if top_prefix and name.startswith(top_prefix):
            rel = name[len(top_prefix):]
        else:
            rel = name
        parts = rel.split("/")
        if len(parts) < 1 or not parts[0]:
            continue
        seq = parts[0]
        g = groups.setdefault(seq, {"members": [], "csize": 0})
        g["members"].append(m)
        g["csize"] += m["csize"]
    out = {}
    for seq, g in groups.items():
        ms = sorted(g["members"], key=lambda m: m["lho"])
        start = ms[0]["lho"]
        end = 0
        for m in ms:
            # local header upper bound: 30 + name + extra; extra unknown, overestimate
            data_end = m["lho"] + 30 + len(m["name"].encode("utf-8")) + 512 + m["csize"]
            end = max(end, data_end)
        out[seq] = {"members": ms, "span": (start, end - 1), "csize": g["csize"]}
    return out


def assign_sequences(seq_groups, disks):
    """Greedy LPT assignment of sequences to disks for byte balance.

    disks: list of disk ids (e.g. [0, 1]). Returns {disk_id: [seq, ...]} sorted big first.
    """
    order = sorted(seq_groups.items(), key=lambda kv: -kv[1]["csize"])
    loads = {d: 0 for d in disks}
    assign = {d: [] for d in disks}
    for seq, g in order:
        d = min(disks, key=lambda x: loads[x])
        assign[d].append(seq)
        loads[d] += g["csize"]
    return assign
