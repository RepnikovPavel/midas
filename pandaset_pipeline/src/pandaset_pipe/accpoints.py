"""Object point accumulation by track id (one-directional, t -> t+1 -> ...).

For every lidar frame t the points that fall INSIDE a vehicle box are recorded
in the box LOCAL frame (position relative to the box's 6 faces) with their
birth frame. Points are retained for the whole trip; at frame t the viewer
shows every accumulated point with birth <= t placed by the box's CURRENT pose.

Per-point OBJECT-motion compensation: a spinning sweep spans ~100 ms, so the
box itself moves while being scanned (track 60 in seq 158: 2.5 -> 10 m/s).
Referencing each point through the frame-instant box pose leaves an
intra-sweep smear of v * 0.1 s (up to ~1 m). Instead a pre-pass builds the
track's WORLD pose series (center + yaw, per annotation frame); each point,
measured at t_i = ts + rel_time, is expressed through the pose interpolated
at t_i between the bracketing frames. Annotation z jitters frame to frame
and would bake vertical layering into the cloud, so the interpolated center
uses the track-median world z instead.

Dual-lidar handling: the two sensors measure the same object with a
systematic 20-30 cm bias, so mixing them doubles surfaces in the overlap
zone. Accumulation therefore runs per ZONE OF RESPONSIBILITY: each frame the
sensor with dominant in-box point count owns the object; only that sensor's
points are accumulated. When dominance flips (with a persistency/hysteresis
guard against noise) the current segment is closed and a NEW accumulation
segment starts in the new sensor's zone — the object stays internally
consistent, and the viewer shows the segment that owns the current frame.

Binary response layout (served as /api/acc):
    u32 header_len | JSON header | fp16 pts blob (all segments) | u16 birth
Per segment: absolute byte offsets into the blobs.
"""

import os

import numpy as np


def _quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _pose_series(sw, labels):
    """Pre-pass: per-uuid world pose series for label-filtered boxes.

    Returns {uuid: {"t": (F,) ms, "c": (F,3) world centers, "yaw": (F,) world
    yaws (unwrapped), "z_ref": track-median world z}} sorted by time.
    """
    series = {}
    for j in range(len(sw)):
        d = np.load(os.path.join(sw.path, f"boxes_{sw.timestamps[j]}.npz"),
                    allow_pickle=True)
        if "boxes" not in d or len(d["boxes"]) == 0:
            continue
        labs = [str(v) for v in d["class_names"]]
        ey = _quat_yaw(d["ego2global_rotation"].astype(np.float64))
        c, s = np.cos(ey), np.sin(ey)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        tr = d["ego2global_translation"].astype(np.float64)
        for k in range(len(d["boxes"])):
            if labs[k] not in labels:
                continue
            b = d["boxes"][k].astype(np.float64)
            cw = R @ b[:3] + tr
            series.setdefault(str(d["uuids"][k]), []).append(
                (float(sw.timestamps[j]), cw, float(ey + b[6])))
    out = {}
    for uuid, vals in series.items():
        vals.sort(key=lambda v: v[0])
        t = np.array([v[0] for v in vals])
        c = np.array([v[1] for v in vals])
        yaw = np.array([v[2] for v in vals])
        dy = (np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi
        yaw = np.concatenate(([yaw[0]], yaw[0] + np.cumsum(dy)))
        out[uuid] = {"t": t, "c": c, "yaw": yaw,
                     "z_ref": float(np.median(c[:, 2]))}
    return out


VOXEL = 0.05          # local-frame downsample grid, m (bounded payload)
DIM_TOL = 0.06        # max dimension variation for a "constant shape" track
MIN_N = 60            # min in-box points for a sensor to be a candidate
SWITCH_RATIO = 1.3    # challenger needs this multiple of the owner's count
SWITCH_RUN = 2        # ... for this many consecutive frames


VEHICLE_LABELS = ("Car", "Pickup Truck", "Medium-sized Truck")


def accumulate_sweep(sw, labels=VEHICLE_LABELS):
    """Full forward pass. Returns (header, pts_blob, birth_blob)."""
    series = _pose_series(sw, labels)
    tracks = {}   # tid -> record
    for j in range(len(sw)):
        snap = sw[j]
        pts = snap.points
        if not len(pts) or not len(snap.boxes):
            continue
        lz = np.load(os.path.join(sw.path, f"lidar_{sw.timestamps[j]}.npz"))
        sid = lz["sensor_id"]
        rt = lz["rel_time"].astype(np.float64)
        R = None
        pw = None
        t_i = None
        boxes = snap.boxes
        labels_arr = snap.box_labels
        uuids = snap.box_uuids
        for k in range(len(boxes)):
            lab = str(labels_arr[k])
            if lab not in labels:
                continue
            b = boxes[k]
            tid = str(uuids[k])
            dx, dy, dz = b[3], b[4], b[5]
            ser = series.get(tid)
            if ser is not None:
                # lazy per-frame world points + per-point ray times
                if pw is None:
                    q = lz["ego2global_rotation"].astype(np.float64)
                    w, x, y, z = q
                    R = np.array([
                        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
                    pw = pts.astype(np.float64) @ R.T + \
                        lz["ego2global_translation"].astype(np.float64)
                    t_i = float(sw.timestamps[j]) + rt * 1000.0
                # bracketing pose pair on the track's own timeline
                i1 = np.searchsorted(ser["t"], t_i)
                i0 = np.clip(i1 - 1, 0, len(ser["t"]) - 1)
                i1 = np.clip(i1, 0, len(ser["t"]) - 1)
                t0, t1 = ser["t"][i0], ser["t"][i1]
                a = np.clip((t_i - t0) / np.maximum(t1 - t0, 1e-6), 0.0, 1.0)
                c0, c1 = ser["c"][i0], ser["c"][i1]
                y0w, y1w = ser["yaw"][i0], ser["yaw"][i1]
                cw = c0 + a[:, None] * (c1 - c0)
                cw[:, 2] = ser["z_ref"]
                yv = y0w + a * (y1w - y0w)
                cy, sy = np.cos(yv), np.sin(yv)
                rel = pw - cw
                lx = rel[:, 0] * cy + rel[:, 1] * sy
                ly = -rel[:, 0] * sy + rel[:, 1] * cy
                lzz = rel[:, 2]
            else:
                c, s = np.cos(b[6]), np.sin(b[6])
                d = pts - b[:3]
                lx = c * d[:, 0] + s * d[:, 1]
                ly = -s * d[:, 0] + c * d[:, 1]
                lzz = d[:, 2]
            inside = ((np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2) &
                      (np.abs(lzz) <= dz / 2))
            if not inside.any():
                continue

            rec = tracks.get(tid)
            if rec is None:
                rec = tracks[tid] = {
                    "dims": np.array([dx, dy, dz], np.float64),
                    "label": lab, "segs": [], "owner": None,
                    "challenger": None, "run": 0,
                }
            elif np.abs(rec["dims"] - [dx, dy, dz]).max() > DIM_TOL:
                rec["dead"] = True
            if rec.get("dead"):
                continue

            # ---- zone of responsibility: dominant sensor owns the frame ----
            n = [int((inside & (sid == v)).sum()) for v in (0, 1)]
            owner = rec["owner"]
            if owner is None:
                owner = 0 if n[0] >= n[1] else 1
                if n[owner] < MIN_N and n[0] + n[1] >= MIN_N:
                    owner = 0 if n[0] >= n[1] else 1
                rec["owner"] = owner
                rec["segs"].append({"sensor": owner, "start": j, "vox": {}})
            else:
                ch = 1 - owner
                flip = (n[ch] >= MIN_N and
                        (n[owner] < MIN_N or n[ch] >= SWITCH_RATIO * n[owner]))
                if flip and rec["challenger"] == ch:
                    rec["run"] += 1
                elif flip:
                    rec["challenger"] = ch
                    rec["run"] = 1
                else:
                    rec["challenger"] = None
                    rec["run"] = 0
                if rec["run"] >= SWITCH_RUN:
                    owner = ch
                    rec["owner"] = owner
                    rec["challenger"] = None
                    rec["run"] = 0
                    rec["segs"].append({"sensor": owner, "start": j, "vox": {}})

            seg = rec["segs"][-1]
            if seg["sensor"] != owner:      # owner changed without new seg
                seg = {"sensor": owner, "start": j, "vox": {}}
                rec["segs"].append(seg)
            sel = np.nonzero(inside & (sid == owner))[0]
            if not len(sel):
                continue
            vox = seg["vox"]
            for i in sel:
                key = (int(lx[i] / VOXEL), int(ly[i] / VOXEL), int(lzz[i] / VOXEL))
                if key not in vox:
                    vox[key] = (lx[i], ly[i], lzz[i], j)

    # ---- pack ----
    header = {"tracks": {}, "voxel": VOXEL}
    pts_chunks, birth_chunks = [], []
    off_pts = off_birth = 0
    for tid, rec in tracks.items():
        if rec.get("dead") or not rec["segs"]:
            continue
        segs_out = []
        keep = False
        for seg in rec["segs"]:
            if not seg["vox"]:
                continue
            n = len(seg["vox"])
            vals = list(seg["vox"].values())
            arr = np.array([(v[0], v[1], v[2]) for v in vals], dtype=np.float16)
            birth = np.array([v[3] for v in vals], dtype=np.uint16)
            pts_chunks.append(arr)
            birth_chunks.append(birth)
            segs_out.append({"s": seg["sensor"], "f0": seg["start"], "n": n,
                             "po": off_pts, "bo": off_birth})
            off_pts += arr.nbytes
            off_birth += birth.nbytes
            keep = True
        if keep:
            header["tracks"][tid] = {"label": rec["label"],
                                     "dims": [float(v) for v in rec["dims"]],
                                     "segs": segs_out}
    pts_blob = b"".join(a.tobytes() for a in pts_chunks)
    birth_blob = b"".join(a.tobytes() for a in birth_chunks)
    return header, pts_blob, birth_blob
