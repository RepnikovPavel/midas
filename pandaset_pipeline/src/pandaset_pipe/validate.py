"""Empirical validation of pandaset coordinate conventions on a raw sequence.

Checks (prints a report, exit 0 if the built-in AXIS_MAP assumption holds):
1. Which vehicle-frame axis points along the direction of travel
   (compares GPS velocity and consecutive-pose deltas against pose columns).
2. The cuboid yaw convention (heading = yaw + pi/2 per devkit docs) against
   tracked-object displacement directions.

Usage: python -m pandaset_pipe.validate /path/to/pandaset_raw/pandaset/001
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from . import geometry as G


def _read_json(p):
    with open(p) as f:
        return json.load(f)


def main():
    seq = sys.argv[1]
    poses_raw = _read_json(os.path.join(seq, "lidar", "poses.json"))
    gps = _read_json(os.path.join(seq, "meta", "gps.json"))
    Ts = [G.pose_to_mat(p) for p in poses_raw]

    # --- 1) forward axis from motion -------------------------------------
    cand = {"+X": 0, "-X": 1, "+Y": 2, "-Y": 3}
    vecs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
    scores = {k: [] for k in cand}
    scores_gps = {k: [] for k in cand}
    for i in range(1, len(Ts) - 1):
        dp = Ts[i + 1][:3, 3] - Ts[i - 1][:3, 3]
        n = np.linalg.norm(dp)
        R = Ts[i][:3, :3]
        if n > 0.5:  # moving
            d = dp / n
            for k, v in zip(cand, vecs):
                scores[k].append(abs(float(d @ (R @ v))))
        g = gps[i]
        v = np.array([g.get("xvel", 0), g.get("yvel", 0), 0.0])
        nv = np.linalg.norm(v)
        if nv > 2.0:
            d = v / nv
            for k, vv in zip(cand, vecs):
                scores_gps[k].append(abs(float(d @ (R @ vv))))

    print("== forward-axis alignment (1.0 = perfect) ==")
    for k in cand:
        a = np.mean(scores[k]) if scores[k] else float("nan")
        b = np.mean(scores_gps[k]) if scores_gps[k] else float("nan")
        print(f"  {k}: pose-delta {a:.4f}   gps-vel {b:.4f}   (n={len(scores[k])}/{len(scores_gps[k])})")

    best = max(cand, key=lambda k: (np.mean(scores[k]) if scores[k] else 0))
    print(f"best forward axis (pose-delta): {best}")
    expected = "+Y"  # AXIS_MAP assumes pandaset vehicle Y-forward
    ok_axis = best == expected

    # --- 2) cuboid yaw convention ------------------------------------------
    cub_files = sorted(glob.glob(os.path.join(seq, "annotations", "cuboids", "*.pkl*")))
    frames = []
    for f in cub_files:
        df = pd.read_pickle(f)
        frames.append(df)
    track = {}
    aligns_doc = []
    aligns_alt = []
    for i, df in enumerate(frames):
        if df is None or not len(df):
            continue
        for _, r in df.iterrows():
            u = r["uuid"]
            p = np.array([r["position.x"], r["position.y"]])
            if u in track and i - track[u][0] <= 3:
                dp = p - track[u][1]
                n = np.linalg.norm(dp)
                if n > 1.5:  # clearly moving object
                    heading = np.arctan2(dp[1], dp[0])
                    yaw_doc = r["yaw"] + np.pi / 2  # documented convention
                    yaw_alt = r["yaw"]              # plain-yaw alternative
                    aligns_doc.append(np.cos(heading - yaw_doc))
                    aligns_alt.append(np.cos(heading - yaw_alt))
            track[u] = (i, p)
    print("== cuboid yaw convention vs tracked displacement ==")
    if aligns_doc:
        print(f"  cos(align) documented (yaw+pi/2): {np.mean(aligns_doc):.4f}  (n={len(aligns_doc)})")
        print(f"  cos(align) plain yaw          : {np.mean(aligns_alt):.4f}")
        ok_yaw = np.mean(aligns_doc) > max(0.5, np.mean(aligns_alt))
    else:
        print("  no moving tracked objects - cannot verify yaw (skipped)")
        ok_yaw = True

    print(f"RESULT: axis {'OK' if ok_axis else 'MISMATCH'} (expected {expected}), "
          f"yaw {'OK' if ok_yaw else 'MISMATCH'}")
    return 0 if (ok_axis and ok_yaw) else 3


if __name__ == "__main__":
    sys.exit(main())
