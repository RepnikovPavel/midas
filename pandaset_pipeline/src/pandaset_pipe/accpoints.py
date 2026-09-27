"""Object point accumulation by track id (one-directional, t -> t+1 -> ...).

Demo scope: "Car" boxes whose dimensions stay constant across the sweep
(objects that do not change shape). For every lidar frame t the points that
fall INSIDE a box are recorded in the box LOCAL frame (i.e. their position
relative to the box's 6 faces) together with their birth frame t. Points are
retained for the whole trip: at frame t a viewer shows every accumulated
point with birth <= t, placed by the box's CURRENT pose.

Binary response layout (served by the webserver as /api/acc):
    u32 header_len | JSON header | fp16 pts (N*3) | u16 birth (N)
per track, concatenated; the header maps track_id -> {dims, label, n, byte
offsets of its pts/birth arrays}.
"""

import numpy as np

VOXEL = 0.05          # local-frame downsample grid, m (bounded payload)
DIM_TOL = 0.06        # max dimension variation for a "constant shape" track


def accumulate_sweep(sw, labels=("Car",)):
    """Full forward pass over the sweep. Returns (header, pts_f16, birth_u16).

    sw: pandaset_pipe.reader.Sweep
    """
    acc = {}   # tid -> {dims, label, vox: {(ix,iy,iz,birth): None}, order}
    for j in range(len(sw)):
        snap = sw[j]
        pts = snap.points
        if not len(pts):
            continue
        boxes = snap.boxes
        if not len(boxes):
            continue
        labels_arr = snap.box_labels
        uuids = snap.box_uuids
        for k in range(len(boxes)):
            lab = str(labels_arr[k])
            if lab not in labels:
                continue
            b = boxes[k]
            tid = str(uuids[k])
            c, s = np.cos(b[6]), np.sin(b[6])
            dx, dy, dz = b[3], b[4], b[5]
            # points in the box local frame (relative to the 6 faces)
            d = pts - b[:3]
            lx = c * d[:, 0] + s * d[:, 1]
            ly = -s * d[:, 0] + c * d[:, 1]
            lz = d[:, 2]
            inside = ((np.abs(lx) <= dx / 2) & (np.abs(ly) <= dy / 2) &
                      (np.abs(lz) <= dz / 2))
            if not inside.any():
                continue
            rec = acc.get(tid)
            if rec is None:
                rec = acc[tid] = {"dims": np.array([dx, dy, dz], np.float64),
                                  "label": lab, "vox": {}, "frames": 0}
            else:
                # constant-shape guard: dims must stay within tolerance
                if np.abs(rec["dims"] - [dx, dy, dz]).max() > DIM_TOL:
                    rec["dead"] = True
            if rec.get("dead"):
                continue
            rec["frames"] += 1
            sel = np.nonzero(inside)[0]
            vox = rec["vox"]
            for i in sel:
                key = (int(lx[i] / VOXEL), int(ly[i] / VOXEL), int(lz[i] / VOXEL))
                if key not in vox:
                    # one point per local voxel, earliest birth frame wins
                    vox[key] = (lx[i], ly[i], lz[i], j)

    # pack into one binary blob
    header = {"tracks": {}, "voxel": VOXEL}
    pts_chunks, birth_chunks = [], []
    off_pts = off_birth = 0
    for tid, rec in acc.items():
        if rec.get("dead") or not rec["vox"]:
            continue
        n = len(rec["vox"])
        vals = list(rec["vox"].values())
        arr = np.array([(v[0], v[1], v[2]) for v in vals], dtype=np.float16)
        birth = np.array([v[3] for v in vals], dtype=np.uint16)
        pts_chunks.append(arr)
        birth_chunks.append(birth)
        header["tracks"][tid] = {
            "label": rec["label"],
            "dims": [float(v) for v in rec["dims"]],
            "n": n, "pts_off": off_pts, "birth_off": off_birth,
        }
        off_pts += arr.nbytes
        off_birth += birth.nbytes
    pts_blob = b"".join(a.tobytes() for a in pts_chunks)
    birth_blob = b"".join(a.tobytes() for a in birth_chunks)
    return header, pts_blob, birth_blob
