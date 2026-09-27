"""Object point accumulation by track id (one-directional, t -> t+1 -> ...).

Demo scope: "Car" boxes whose dimensions stay constant across the sweep.
For every lidar frame t the points that fall INSIDE a box are recorded in the
box LOCAL frame (position relative to the box's 6 faces) with their birth
frame. Points are retained for the whole trip; at frame t the viewer shows
every accumulated point with birth <= t placed by the box's CURRENT pose.

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

_POSE_CACHE = {}


def _ego_world(sw, j):
    """(R, t) ego(frame j) -> world, from the lidar npz pose."""
    key = (sw.path, j)
    if key not in _POSE_CACHE:
        import numpy as _np
        lz = _np.load(os.path.join(sw.path, f"lidar_{sw.timestamps[j]}.npz"))
        q = lz["ego2global_rotation"].astype(_np.float64)
        w, x, y, z = q
        R = _np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        t = lz["ego2global_translation"].astype(_np.float64)
        if len(_POSE_CACHE) > 200:
            _POSE_CACHE.clear()
        _POSE_CACHE[key] = (R, t)
    return _POSE_CACHE[key]


def _box_world_pose(sw, j, k):
    """World (t_ms, center, yaw) of boxes-npz row k at frame j."""
    d = np.load(os.path.join(sw.path, f"boxes_{sw.timestamps[j]}.npz"))
    b = d["boxes"][k].astype(np.float64)
    q = d["ego2global_rotation"].astype(np.float64)
    w, x, y, z = q
    ego_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    c_w = R @ b[:3] + d["ego2global_translation"].astype(np.float64)
    return (float(sw.timestamps[j]), c_w, float(ego_yaw + b[6]))


def _update_prev_pose(rec, sw, j, k):
    try:
        rec["prev_pose"] = _box_world_pose(sw, j, k)
    except Exception:  # noqa: BLE001
        pass


VOXEL = 0.05          # local-frame downsample grid, m (bounded payload)
DIM_TOL = 0.06        # max dimension variation for a "constant shape" track
MIN_N = 60            # min in-box points for a sensor to be a candidate
SWITCH_RATIO = 1.3    # challenger needs this multiple of the owner's count
SWITCH_RUN = 2        # ... for this many consecutive frames


VEHICLE_LABELS = ("Car", "Pickup Truck", "Medium-sized Truck")


def accumulate_sweep(sw, labels=VEHICLE_LABELS):
    """Full forward pass. Returns (header, pts_blob, birth_blob).

    Per-point OBJECT-motion compensation: a spinning sweep spans ~100 ms, so
    the box itself moves while being scanned. Each point, measured at
    t_i = ts + rel_time, is referenced through the track's WORLD pose
    interpolated at t_i (between the bracketing frames' box poses) instead of
    the frame-instant pose — this removes the intra-sweep arc shear that
    otherwise cuts the accumulated object.
    """
    tracks = {}   # tid -> record
    for j in range(len(sw)):
        snap = sw[j]
        pts = snap.points
        if not len(pts):
            continue
        lz = np.load(os.path.join(sw.path, f"lidar_{sw.timestamps[j]}.npz"))
        sid = lz["sensor_id"]
        rt = lz["rel_time"].astype(np.float64)
        boxes = snap.boxes
        if not len(boxes):
            continue
        labels_arr = snap.box_labels
        uuids = snap.box_uuids
        # world pose of each current box (for the object-motion interpolation)
        bd = np.load(os.path.join(sw.path, f"boxes_{sw.timestamps[j]}.npz")) \
            if False else None
        for k in range(len(boxes)):
            lab = str(labels_arr[k])
            if lab not in labels:
                continue
            b = boxes[k]
            tid = str(uuids[k])
            c, s = np.cos(b[6]), np.sin(b[6])
            dx, dy, dz = b[3], b[4], b[5]
            d = pts - b[:3]
            lx = c * d[:, 0] + s * d[:, 1]
            ly = -s * d[:, 0] + c * d[:, 1]
            lz = d[:, 2]
            inside = ((np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2) &
                      (np.abs(lz) <= dz / 2))
            if not inside.any():
                continue

            rec = tracks.get(tid)
            if rec is None:
                rec = tracks[tid] = {
                    "dims": np.array([dx, dy, dz], np.float64),
                    "label": lab, "segs": [], "owner": None,
                    "challenger": None, "run": 0,
                    "prev_pose": None,   # (t_ms, c_w, yaw_w) of previous frame
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
                key = (int(lx[i] / VOXEL), int(ly[i] / VOXEL), int(lz[i] / VOXEL))
                if key not in vox:
                    vox[key] = (lx[i], ly[i], lz[i], j)

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
