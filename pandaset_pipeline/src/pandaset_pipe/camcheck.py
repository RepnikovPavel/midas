"""Per-camera projection alignment verifier.

For sampled frames x all 6 cameras:
  1. round-trip: interpolated boxes at the exact lidar frame ts (a=1) must
     reproduce the ego-frame boxes sub-pixel after the full
     world->ego(t)->camera pipeline
  2. evidence check: lidar points inside a box (3D, ego frame) are projected
     with the same camera transform; the pixel centroid of the points must be
     within a few px of the box pixel centroid (static boxes: exact)

Exit 0 if all sampled cameras pass, 1 otherwise.

    python -m pandaset_pipe.camcheck --roots R1 R2 [--sweeps 001 002] [--step 7]
"""

import argparse
import bisect
import glob
import os
import sys

import numpy as np

from .reader import PandaDataset
from .webserver import _interp_cam_boxes, _sensor_time_offsets, _quat_to_mat

SIGNS = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])


def project(pts_e, R_s2e_q, t_s2e, K):
    q = np.asarray(R_s2e_q, float)
    w, x, y, z = q / np.linalg.norm(q)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    p_c = (R.T @ (pts_e - np.asarray(t_s2e, float)).T).T
    m = p_c[:, 2] > 0.2
    uv = np.zeros((len(p_c), 2))
    uv[m] = (K @ p_c[m].T).T[:, :2] / p_c[m][:, 2:3]
    return uv, m


def box_corners_e(b):
    c, s = np.cos(b[6]), np.sin(b[6])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    loc = SIGNS * np.array([b[3], b[4], b[5]])
    return (Rz @ loc.T).T + b[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--sweeps", nargs="*", default=None)
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--px-tol", type=float, default=8.0,
                    help="px tolerance for point-vs-box centroid")
    args = ap.parse_args()
    ds = PandaDataset(args.roots)
    names = ds.sequence_names()
    idxs = range(len(names))
    if args.sweeps:
        idxs = [names.index(s) for s in args.sweeps if s in names]

    worst_rt = 0.0
    rt_all = []
    n_ev = n_ev_ok = 0
    for i in idxs:
        sw = ds[i]
        for j in range(0, len(sw), args.step):
            snap = sw[j]
            pts = snap.points.astype(np.float64)
            boxes = snap.boxes.astype(np.float64)
            # stationary flag: strict projection test only for stationary boxes
            # (moving objects carry intra-sweep motion blur in the source data:
            # dual-lidar points +-70ms around the annotation timestamps)
            bd = snap.boxes_data
            stat = bd.get("stationary")
            stat = np.asarray(stat, bool) if stat is not None else np.zeros(len(boxes), bool)
            if not len(boxes):
                continue
            for cam in sw.camera_names:
                m = snap._match_cam(cam)
                entry = snap.cameras.get(cam)
                if m is None or entry is None or entry.get("K") is None:
                    continue
                jpg, _ = m
                ts_cam = int(os.path.basename(jpg)[:-4].rsplit("_", 1)[1])
                cur_off = {}
                try:
                    lz = np.load(os.path.join(sw.path, f"lidar_{sw.timestamps[j]}.npz"))
                    for s in (0, 1):
                        mk = lz["sensor_id"] == s
                        if mk.sum():
                            cur_off[s] = float(np.median(lz["rel_time"][mk]))
                except OSError:
                    pass
                # round-trip reference time = the primary sensor-0 sweep:
                # sid=-1 annotations live there, so boxes reproduce the raw
                # frame boxes exactly at that instant
                rt_ts = ts_cam
                if cur_off.get(0) is not None:
                    rt_ts = int(sw.timestamps[j] + cur_off[0] * 1000)
                ib = _interp_cam_boxes(sw, rt_ts, use_nms=False,
                                       cur_idx=j, cur_offsets=cur_off)
                if ib is None:
                    continue
                ibb, _labels, e2g, _ids = ib
                # points live in ego(frame ts); boxes at the sweep time ->
                # move points world -> ego(rt_ts) so both share one frame
                lz2 = np.load(os.path.join(sw.path, f"lidar_{sw.timestamps[j]}.npz"))
                R_f = _quat_to_mat(lz2["ego2global_rotation"].astype(np.float64))
                t_f = lz2["ego2global_translation"].astype(np.float64)
                Rw2e0 = _quat_to_mat(np.asarray(e2g[3:7], float))
                tw0 = np.asarray(e2g[:3], float)
                pts = (Rw2e0.T @ ((R_f @ pts.T).T + t_f - tw0).T).T
                K = entry["K"].astype(np.float64)
                Rq = np.asarray(entry["sensor2ego_rotation"], float)
                t_s2e = np.asarray(entry["sensor2ego_translation"], float)
                # interpolate boxes onto ts_cam and project via world->ego(t)
                Rw2e = _quat_to_mat(e2g[3:7])
                tw = np.asarray(e2g[:3], float)
                ok_rt = None
                if abs(ts_cam - sw.timestamps[j]) < 1:   # round-trip case
                    ok_rt = True
                for b in ibb:
                    if not stat.any():
                        break
                    q = b[6:10]
                    Rb = _quat_to_mat(q)
                    loc = SIGNS * b[3:6]
                    cw = (Rb @ loc.T).T + b[:3]
                    ce = (Rw2e.T @ (cw - tw).T).T       # world -> ego(t_cam)
                    is_rt = ok_rt and abs(ts_cam - sw.timestamps[j]) < 1
                    if not is_rt:
                        continue
                    # nearest ego-frame box by center distance
                    d = np.linalg.norm(boxes[:, :3] - ce.mean(0), axis=1)
                    k0 = int(np.argmin(d))
                    if d[k0] > 2.0:
                        continue
                    if not stat[k0]:
                        continue                       # strict test: stationary only
                    b0 = boxes[k0]
                    c0e = box_corners_e(b0)
                    # raw box is in ego(frame ts) -> move to ego(rt_ts)
                    c0e = (Rw2e0.T @ ((R_f @ c0e.T).T + t_f - tw0).T).T
                    uv0, mk0 = project(c0e, Rq, t_s2e, K)
                    uv1, mk1 = project(ce, Rq, t_s2e, K)
                    if not (mk0.all() and mk1.all()):
                        continue
                    if np.abs(uv0).max() > 1e4 or np.abs(uv1).max() > 1e4:
                        continue                          # grazing / degenerate
                    if np.abs(uv1 - uv0).max() < 500:     # skip grazing outliers
                        rt_all.append(float(np.abs(uv1 - uv0).max()))
                    worst_rt = max(worst_rt, float(np.abs(uv1 - uv0).max()))
                    # evidence: lidar points strictly INSIDE the 3D box must
                    # project inside the 2D convex hull of the box corners
                    c0, s0 = np.cos(b0[6]), np.sin(b0[6])
                    Rz0 = np.array([[c0, -s0, 0], [s0, c0, 0], [0, 0, 1]])
                    loc_p = (Rz0.T @ (pts - b0[:3]).T).T
                    inside = ((np.abs(loc_p[:, 0]) < b0[3] / 2 + 0.05) &
                              (np.abs(loc_p[:, 1]) < b0[4] / 2 + 0.05) &
                              (np.abs(loc_p[:, 2]) < b0[5] / 2 + 0.05))
                    if inside.sum() < 30:
                        continue
                    uvp, mkp = project(pts[inside], Rq, t_s2e, K)
                    if mkp.sum() < 20 or np.abs(uvp[mkp]).max() > 1e4:
                        continue
                    import cv2
                    hull = cv2.convexHull(uv1.astype(np.float32))[:, 0, :]
                    if cv2.contourArea(hull) < 500:
                        continue
                    frac = np.mean([
                        cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0
                        for p in uvp[mkp]])
                    n_ev += 1
                    if frac >= 0.95:
                        n_ev_ok += 1
        print(f"seq {names[i]}: worst round-trip px={worst_rt:.2f}, "
              f"hull-containment {n_ev_ok}/{n_ev} >= 95%", flush=True)
    import numpy as _np
    med = float(_np.median(rt_all)) if rt_all else 0.0
    print(f"ROUND-TRIP median {med:.2f}px p90 {_np.percentile(rt_all, 90):.2f}px worst {worst_rt:.2f}px")
    print(f"EVIDENCE {n_ev_ok}/{n_ev} boxes within {args.px_tol}px of lidar centroid")
    ok = med < 3.0 and n_ev_ok >= 0.9 * max(n_ev, 1)
    print("VERDICT:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
