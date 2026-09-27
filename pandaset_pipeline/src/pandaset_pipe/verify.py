"""CLI integrity check of downloaded + converted PandaSet.

Runs inside the pipeline container on the server:

    docker run --rm \
      --mount type=bind,src=/mnt/hdd1/datasets,target=/mnt/hdd1/datasets \
      --mount type=bind,src=/mnt/hdd2/datasets,target=/mnt/hdd2/datasets \
      pandaset-pipeline:latest python -m pandaset_pipe.verify \
        --zip-index /mnt/hdd1/datasets/pandaset_work/zip_index.json

Checks:
  1. every zip member file exists on its disk with the exact uncompressed size
  2. every sequence has a sweep_<seq> npz dir with .converted marker
  3. per sequence: lidar / cuboids / semseg file counts match between raw and npz
Exit code 0 = OK, 1 = problems.
"""

import argparse
import glob
import json
import os

DEFAULT_RAW = ["/mnt/hdd1/datasets/pandaset_raw", "/mnt/hdd2/datasets/pandaset_raw"]
DEFAULT_NPZ = ["/mnt/hdd1/datasets/pandaset_npz", "/mnt/hdd2/datasets/pandaset_npz"]


def _find(roots, rel):
    for r in roots:
        p = os.path.join(r, rel)
        if os.path.isdir(p):
            return p
    return None


def check_raw(index_path, raw_roots):
    members = json.load(open(index_path))
    files = [m for m in members if not m["name"].endswith("/") and m["usize"] > 0]
    missing, badsize = [], []
    for m in files:
        parts = m["name"].split("/")  # pandaset/SEQ/rest...
        if len(parts) < 3:
            continue
        seq, rest = parts[1], "/".join(parts[2:])
        base = _find(raw_roots, seq)
        if base is None:
            missing.append((m["name"], "no seq dir"))
            continue
        p = os.path.join(base, rest)
        if not os.path.exists(p):
            missing.append((m["name"], "missing"))
        elif os.path.getsize(p) != m["usize"]:
            badsize.append((m["name"], os.path.getsize(p), m["usize"]))
    print(f"raw: {len(files)} files from index, missing={len(missing)}, badsize={len(badsize)}")
    for x in missing[:10]:
        print("  MISS", x)
    for x in badsize[:10]:
        print("  SIZE", x)
    return not missing and not badsize


def check_npz(raw_roots, npz_roots):
    seqs = sorted(os.path.basename(p)
                  for r in raw_roots for p in glob.glob(os.path.join(r, "[0-9]*")))
    ok = True
    n_lidar = n_sem = 0
    for s in seqs:
        rp = _find(raw_roots, s)
        sp = _find(npz_roots, f"sweep_{s}")
        if sp is None:
            print(f"  NPZ MISSING sweep_{s}")
            ok = False
            continue
        if not os.path.exists(os.path.join(sp, ".converted")):
            print(f"  {s}: no .converted marker")
            ok = False
        pairs = [
            (glob.glob(os.path.join(rp, "lidar/*.pkl*")),
             glob.glob(os.path.join(sp, "lidar_*.npz"))),
            (glob.glob(os.path.join(rp, "annotations/cuboids/*.pkl*")),
             glob.glob(os.path.join(sp, "boxes_*.npz"))),
            (glob.glob(os.path.join(rp, "annotations/semseg/*.pkl*")),
             glob.glob(os.path.join(sp, "semseg_*.npz"))),
        ]
        n_lidar += len(pairs[0][1])
        n_sem += len(pairs[2][1])
        for raw_list, npz_list in pairs:
            if len(raw_list) != len(npz_list):
                print(f"  {s}: raw={len(raw_list)} npz={len(npz_list)} mismatch")
                ok = False
    print(f"npz: {len(seqs)} sequences, {n_lidar} lidar frames, "
          f"{n_sem} semseg frames (semseg exists for a subset of sequences)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-index", required=True, help="path to zip_index.json")
    ap.add_argument("--raw", nargs="+", default=DEFAULT_RAW)
    ap.add_argument("--npz", nargs="+", default=DEFAULT_NPZ)
    args = ap.parse_args()
    ok = check_raw(args.zip_index, args.raw)
    ok = check_npz(args.raw, args.npz) and ok
    print("VERDICT:", "OK" if ok else "PROBLEMS")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
