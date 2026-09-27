"""Render standalone camera images with projected boxes (+ optional lidar).

Produces full-resolution PNGs: the camera photo itself with the per-camera
timestamp-interpolated box wireframes drawn on top — not a UI screenshot.

    python -m pandaset_pipe.renderproj --roots R1 R2 --seq 002 \
        --frames 6 9 12 --cam left_camera --lidar --out /out/dir
"""

import argparse
import colorsys
import glob
import os

import numpy as np

from .reader import PandaDataset
from .webserver import _interp_cam_boxes, _quat_to_mat

SIGNS = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def class_color(name):
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    hue = (h * 137.508) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, 0.55, 0.85)
    return int(r * 255), int(g * 255), int(b * 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frames", nargs="+", type=int, required=True)
    ap.add_argument("--cam", default="left_camera")
    ap.add_argument("--lidar", action="store_true", help="also draw lidar points")
    ap.add_argument("--nms", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    ds = PandaDataset(args.roots)
    i = ds.sequence_names().index(args.seq)
    sw = ds[i]
    os.makedirs(args.out, exist_ok=True)

    for j in args.frames:
        snap = sw[j]
        m = snap._match_cam(args.cam)
        entry = snap.cameras.get(args.cam)
        if m is None or entry is None:
            print(f"frame {j}: no {args.cam}")
            continue
        ts_cam = int(os.path.basename(m[0])[:-4].rsplit("_", 1)[1])
        cur_off = {}
        try:
            lz = np.load(os.path.join(sw.path, f"lidar_{sw.timestamps[j]}.npz"))
            for s in (0, 1):
                mk = lz["sensor_id"] == s
                if mk.sum():
                    cur_off[s] = float(np.median(lz["rel_time"][mk]))
        except OSError:
            pass
        ib = _interp_cam_boxes(sw, ts_cam, use_nms=args.nms,
                               cur_idx=j, cur_offsets=cur_off)
        img = cv2.imread(m[0])
        if img is None:
            print(f"frame {j}: cannot read {m[0]}")
            continue
        H, W = img.shape[:2]

        if args.lidar:
            # points: ego(frame) -> world -> ego(t_cam) -> camera
            pts = snap.points.astype(np.float64)
            e2g_t = snap.lidar["ego2global_translation"].astype(np.float64)
            e2g_q = snap.lidar["ego2global_rotation"].astype(np.float64)
            R_f = _quat_to_mat(e2g_q)
            pw = (R_f @ pts.T).T + e2g_t
            Rw2e = _quat_to_mat(ib[2][3:7])
            tw = np.asarray(ib[2][:3])
            pe = (Rw2e.T @ (pw - tw).T).T
            K = entry["K"].astype(np.float64)
            R_s2e = _quat_to_mat(np.asarray(entry["sensor2ego_rotation"], float))
            t_s2e = np.asarray(entry["sensor2ego_translation"], float)
            pc = (R_s2e.T @ (pe - t_s2e).T).T
            msk = pc[:, 2] > 0.5
            uv = (K @ pc[msk].T).T
            uv = uv[:, :2] / uv[:, 2:3]
            zsel = pc[msk][:, 2]
            zin = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            for (u, v), z in zip(uv[zin], zsel[zin]):
                t = min(max(z / 60.0, 0.0), 0.999)   # range colormap
                c = (int(255 * (1 - t)), int(255 * (1 - abs(2 * t - 1))), int(255 * t))
                cv2.circle(img, (int(u), int(v)), 1, c, -1)

        K = entry["K"].astype(np.float64)
        q = np.asarray(entry["sensor2ego_rotation"], float)
        R_s2e = _quat_to_mat(q)
        t_s2e = np.asarray(entry["sensor2ego_translation"], float)
        Rw2e = _quat_to_mat(ib[2][3:7])
        tw = np.asarray(ib[2][:3])
        for b, lab in zip(ib[0], ib[1]):
            Rb = _quat_to_mat(b[6:10])
            loc = SIGNS * b[3:6]
            cw = (Rb @ loc.T).T + b[:3]
            ce = (Rw2e.T @ (cw - tw).T).T
            pc = (R_s2e.T @ (ce - t_s2e).T).T
            if pc[:, 2].min() < 0.3:
                continue
            uv = (K @ pc.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            if uv[:, 0].max() < -30 or uv[:, 0].min() > W + 30 or \
               uv[:, 1].max() < -30 or uv[:, 1].min() > H + 30:
                continue
            col = class_color(lab)
            for a, b2 in EDGES:
                p1 = tuple(uv[a].round().astype(int))
                p2 = tuple(uv[b2].round().astype(int))
                cv2.line(img, p1, p2, col, 3, cv2.LINE_AA)
            topi = int(np.argmin(uv[4:8][:, 1])) + 4
            tx, ty = int(uv[topi, 0]), int(uv[topi, 1])
            (tw2, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (tx - tw2 // 2 - 4, ty - th - 10),
                          (tx + tw2 // 2 + 4, ty - 4), (0, 0, 0), -1)
            cv2.putText(img, lab, (tx - tw2 // 2, ty - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        outp = os.path.join(args.out, f"proj_{args.seq}_f{j}_{args.cam.replace('_camera', '')}.png")
        cv2.imwrite(outp, img)
        print("wrote", outp)


if __name__ == "__main__":
    main()
