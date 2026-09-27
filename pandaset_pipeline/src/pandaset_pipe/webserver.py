"""Web backend for the PandaSet viewer (serves frontend + frame data).

Run:  python -m pandaset_pipe.webserver --roots R1 R2 --port 8777 [--osm-dir DIR]
Open: http://localhost:8777

OSM tiles: served from the predownloaded tree (pandaset_pipe.osmtiles) via
/api/tile/z/x/y; missing tiles fall back to tile.openstreetmap.org and are
cached to disk when writable.
"""
import argparse
import asyncio
import concurrent.futures
import json
import os
import struct

import numpy as np
from aiohttp import web

from .reader import PandaDataset
from . import geofit
from .accpoints import accumulate_sweep

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web_static")
OSM_UPSTREAM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
UA = "pandaset-pipeline-viewer/1.0"


def _mat_to_quat(R):
    """3x3 rotation -> quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_slerp(q0, q1, a):
    """Quaternion slerp (wxyz), numpy."""
    q0 = np.asarray(q0, float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, float) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


_SENSOR_TS_CACHE = {}   # (sweep_path, frame_idx) -> {sensor_id: median rel_time}


_TRACK_ID_CACHE = {}   # sweep path -> {uuid: int track id (0..N-1, first-appearance order)}


def _track_id_map(sw):
    """Stable per-sweep mapping uuid -> small integer track id (0-based)."""
    key = sw.path
    if key in _TRACK_ID_CACHE:
        return _TRACK_ID_CACHE[key]
    m = {}
    for j in range(len(sw)):
        p = os.path.join(sw.path, f"boxes_{sw.timestamps[j]}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        for u in d["uuids"]:
            us = str(u)
            if us not in m:
                m[us] = len(m)
    if len(_TRACK_ID_CACHE) > 8:
        _TRACK_ID_CACHE.clear()
    _TRACK_ID_CACHE[key] = m
    return m


def _sensor_time_offsets(sw, j, cur_idx, cur_offsets):
    """Median rel_time per lidar sensor for frame j (dual-sensor sweeps:
    Pandar64 and PandarGT have different capture times inside one merged
    frame). Cached; the current frame's offsets are reused when passed."""
    if j == cur_idx:
        return cur_offsets
    key = (sw.path, j)
    if key in _SENSOR_TS_CACHE:
        return _SENSOR_TS_CACHE[key]
    off = {}
    try:
        d = np.load(os.path.join(sw.path, f"lidar_{sw.timestamps[j]}.npz"))
        rt, sid = d["rel_time"], d["sensor_id"]
        for s in (0, 1):
            m = sid == s
            if m.sum():
                off[s] = float(np.median(rt[m]))
    except OSError:
        pass
    if len(_SENSOR_TS_CACHE) > 40:
        _SENSOR_TS_CACHE.clear()
    _SENSOR_TS_CACHE[key] = off
    return off


def _interp_cam_boxes(sw, ts_cam, use_nms=True, cur_idx=None, cur_offsets=None):
    """Boxes + ego pose interpolated/extrapolated onto a camera's own timestamp.

    Lidar and cameras are not synchronized (offsets 0-100 ms per camera) and a
    merged lidar frame contains two sensor sweeps with different capture times.
    For each track (uuid) the detections around ts_cam — each stamped with its
    own sensor-sweep time — form the trajectory; the adjacent pair nearest to
    ts_cam is linearly interpolated, or extrapolated (bounded) when the camera
    timestamp lies past a track's end. The ego pose is lerped/slerped to ts_cam
    the same way. Calibration is assumed correct (per the dataset).

    Returns (boxes, labels, e2g): boxes rows are [x y z_world dx dy dz qw qx
    qy qz], e2g is [tx ty tz qw qx qy qz] at ts_cam.
    """
    import bisect
    ts = sw.timestamps
    n = len(ts)
    if n == 0:
        return None

    def load(j, off):
        p = os.path.join(sw.path, f"boxes_{ts[j]}.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        t = d["ego2global_translation"].astype(np.float64)
        q_ego = d["ego2global_rotation"].astype(np.float64)
        R_e2w = _quat_to_mat(q_ego)                     # full ego->world rotation
        b = d["boxes"].astype(np.float64)
        labels = d["class_names"]
        uuids = d["uuids"]
        sids = d["sensor_id"]
        if use_nms and "nms_keep" in d.files:
            keep = d["nms_keep"].astype(bool)
            b, labels, uuids, sids = b[keep], labels[keep], uuids[keep], sids[keep]
        out = {}
        for k in range(len(b)):
            sid = int(sids[k]) if sids[k] is not None else -1
            # this detection's own capture time: its sensor's sweep offset.
            # sid=-1 (unattributed, the majority) is stamped on the PRIMARY
            # Pandar64 sweep: fitted photometrically from track 49 (seq 002,
            # frames 8-11) at ts-46ms +- 6ms ~= sensor-0 sweep (-48ms).
            off_sid = off.get(sid, off.get(0, 0.0))
            tb = ts[j] + off_sid * 1000.0
            cen_w = R_e2w @ b[k, :3] + t
            yaw = b[k, 6]
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            q_box = _mat_to_quat(R_e2w @ Rz)            # exact box pose in world
            out[str(uuids[k])] = (cen_w, b[k, 3], b[k, 4], b[k, 5],
                                  q_box, str(labels[k]), tb)
        return out, t, q_ego

    i = bisect.bisect_left(ts, ts_cam)
    # load a window of frames around ts_cam so tracks can be bracketed or
    # extrapolated from the nearest pair of detections on their own times
    frames = sorted({min(max(k, 0), n - 1) for k in (i - 2, i - 1, i, i + 1)})
    samples = {}   # uuid -> [(t_ms, cen_w, q_box, dx, dy, dz, label), ...]
    ego = {}       # frame idx -> (t, q)
    for jf in frames:
        off = _sensor_time_offsets(sw, jf, cur_idx, cur_offsets or {})
        L = load(jf, off)
        if L is None:
            continue
        b, t, q = L
        ego[jf] = (t, q)
        for uuid, (c0, dx, dy, dz, q_box, lab, tb) in b.items():
            samples.setdefault(uuid, []).append((tb, c0, q_box, dx, dy, dz, lab))

    # ---- ego pose at ts_cam: bracket or extrapolate from nearest pair ----
    js = sorted(ego)
    if not js:
        return None
    j_lo = max((k for k in js if ts[k] <= ts_cam), default=min(js))
    j_hi = min((k for k in js if ts[k] > ts_cam), default=max(js))
    if j_lo == j_hi:
        j_hi = j_lo + 1 if j_lo + 1 in ego else j_lo - 1
    t0, q0 = ego[j_lo]
    t1, q1 = ego[j_hi]
    dt_e = float(ts[j_hi] - ts[j_lo]) or 1.0
    a_e = (ts_cam - ts[j_lo]) / dt_e
    a_e = min(max(a_e, -0.6), 1.6)                  # bounded ego extrapolation
    e2g = np.concatenate([t0 + a_e * (t1 - t0), _quat_slerp(q0, q1, a_e)])

    # ---- per-track interpolation / extrapolation onto ts_cam ----
    boxes, labels, uuids = [], [], []
    for uuid, ss in samples.items():
        ss.sort(key=lambda s: s[0])
        if len(ss) == 1:
            if abs(ss[0][0] - ts_cam) > 150:        # single stale detection
                continue
            tb, c0, q_box, dx, dy, dz, lab = ss[0]
            cen, qb = c0, q_box
        else:
            # adjacent pair nearest in time to ts_cam
            best, bd = None, None
            for a_, b_ in zip(ss[:-1], ss[1:]):
                d = max(abs(a_[0] - ts_cam), abs(b_[0] - ts_cam))
                if bd is None or d < bd:
                    bd, best = d, (a_, b_)
            (ta, ca, qa, dx, dy, dz, lab), (tb, cb, qb2, *_rest) = best[0], best[1]
            if abs(ta - ts_cam) > 250 and abs(tb - ts_cam) > 250:
                continue                            # track too far from camera ts
            dt = float(tb - ta) or 1.0
            a = (ts_cam - ta) / dt
            if not (-0.35 <= a <= 1.35):            # bounded extrapolation
                a = min(max(a, -0.35), 1.35)
            cen = ca + a * (cb - ca)
            qb = _quat_slerp(qa, qb2, a)
        boxes.append([cen[0], cen[1], cen[2], dx, dy, dz] + list(qb))
        labels.append(lab)
        uuids.append(uuid)
    if not boxes:
        return None
    tid = _track_id_map(sw)
    return (np.array(boxes, dtype=np.float64), labels, e2g.tolist(),
            [str(tid.get(u, -1)) for u in uuids])


def make_app(roots, osm_dir=None):
    ds = PandaDataset(roots)
    if len(ds) == 0:
        raise SystemExit("no sweeps found in " + ", ".join(roots))
    app = web.Application()
    app["ds"] = ds
    app["osm_dir"] = osm_dir
    app["gps_cache"] = {}      # sweep idx -> track dict
    app["classes_cache"] = {}  # sweep idx -> {class: count}
    app["acc_cache"] = {}      # sweep idx -> binary accumulated-points payload
    app["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def index(request):
        return web.FileResponse(os.path.join(STATIC_DIR, "index.html"),
                                headers={"Cache-Control": "no-cache, must-revalidate"})

    async def sweeps(request):
        return web.json_response({
            "sweeps": ds.sequence_names(),
            "counts": [len(ds[i]) for i in range(len(ds))],
        })

    def _gps_track_sync(i):
        sw = ds[i]
        n = len(sw)
        lat = np.zeros(n); lon = np.zeros(n); spd = np.zeros(n)
        wxy = np.full((n, 2), np.nan)
        for j in range(n):
            snap = sw[j]
            g = snap.gps
            lat[j] = g.get("lat", 0.0)
            lon[j] = g.get("long", 0.0)
            spd[j] = g.get("speed", 0.0)
            bpath = os.path.join(sw.path, f"boxes_{sw.timestamps[j]}.npz")
            if os.path.exists(bpath):
                wxy[j] = np.load(bpath)["ego2global_translation"][:2]
        aln = geofit.fit_alignment(wxy, lat, lon)
        return {"lat": lat.tolist(), "lon": lon.tolist(),
                "speed": (spd * 3.6).tolist(), "ts": sw.timestamps, "aln": aln}

    def gps_track(i):
        """Full per-frame GPS track + world<->ENU alignment of a sweep; cached."""
        if i not in app["gps_cache"]:
            app["gps_cache"][i] = _gps_track_sync(i)
        return app["gps_cache"][i]

    def box_classes(i):
        """Unique box class names + frame counts of a sweep; cached."""
        if i in app["classes_cache"]:
            return app["classes_cache"][i]
        sw = ds[i]
        counts = {}
        for j in range(len(sw)):
            for lab in sw[j].box_labels:
                lab = str(lab)
                counts[lab] = counts.get(lab, 0) + 1
        app["classes_cache"][i] = counts
        return counts

    async def meta(request):
        i = int(request.query.get("sweep", 0))
        sw = ds[i]
        return web.json_response({
            "name": sw.name,
            "frames": len(sw),
            "cameras": sw.camera_names,
            "semseg_classes": sw.semseg_classes,
            "has_semseg": len(sw.semseg_classes) > 0,
            "box_classes": await asyncio.get_event_loop().run_in_executor(
                app["executor"], box_classes, i),
        })

    async def gps_track_api(request):
        i = int(request.query.get("sweep", 0))
        loop = asyncio.get_event_loop()
        return web.json_response(await loop.run_in_executor(
            app["executor"], gps_track, i))

    def _frame_sync(i, j, use_nms):
        sw = ds[i]
        j = max(0, min(j, len(sw) - 1))
        snap = sw[j]
        # raw lidar npz: keep points as stored (fp16) -> half the payload
        lz = np.load(os.path.join(sw.path, f"lidar_{sw.timestamps[j]}.npz"))
        pts = lz["points"]
        pts_dtype = pts.dtype
        inten = lz["intensity"]
        ego_t = lz["ego2global_translation"]
        ego_q = lz["ego2global_rotation"]
        sem = snap.semseg
        if sem is None:
            sem = np.zeros(len(pts), np.uint8)
        # per-sensor sweep time offsets of THIS merged frame (dual lidar)
        cur_offsets = {}
        for s in (0, 1):
            msk = lz["sensor_id"] == s
            if msk.sum():
                cur_offsets[s] = float(np.median(lz["rel_time"][msk]))
        if use_nms:
            boxes, labels = snap.boxes_nms()
        else:
            boxes, labels = snap.boxes, snap.box_labels
        boxes = boxes.astype(np.float32, copy=False)
        cams = {}
        for cam in sw.camera_names:
            m = snap._match_cam(cam)
            if m is None:
                continue
            jpg, npz_path = m
            entry = snap.cameras.get(cam)
            if entry is None or entry.get("K") is None:
                continue
            ts_cam = int(os.path.basename(jpg)[:-4].rsplit("_", 1)[1])
            cams[cam] = {
                "ts": ts_cam,
                "K": entry["K"].astype(float).tolist(),
                "R": np.asarray(entry["sensor2ego_rotation"]).astype(float).tolist(),
                "t": np.asarray(entry["sensor2ego_translation"]).astype(float).tolist(),
                "w": int(entry["image"].shape[1]),
                "h": int(entry["image"].shape[0]),
            }
            # boxes interpolated onto THIS camera's own timestamp (async
            # sensors + dual-lidar per-box sensor timelines)
            ib = _interp_cam_boxes(sw, ts_cam, use_nms=use_nms,
                                   cur_idx=j, cur_offsets=cur_offsets)
            if ib is not None:
                boxes_c, labels_c, e2g_c, ids_c = ib
                cams[cam]["ib"] = boxes_c.astype(float).tolist()
                cams[cam]["ib_labels"] = labels_c
                cams[cam]["ib_ids"] = ids_c
                cams[cam]["ib_e2g"] = e2g_c
        header = {
            "ts": snap.ts, "n": len(pts), "nb": len(boxes),
            "labels": [str(x) for x in labels],
            "ids": [str(_track_id_map(sw).get(str(u), -1)) for u in snap.box_uuids],
            "gps": snap.gps, "cams": cams,
            "frame": j, "frames": len(sw),
            "pts_dtype": str(pts_dtype),
            # ego pose in world (t + quat wxyz): places OSM tiles in the ego frame
            "e2g": [float(v) for v in ego_t] + [float(v) for v in ego_q],
        }
        hj = json.dumps(header).encode()
        if len(hj) % 2:          # keep the fp16 blob 2-byte aligned
            hj += b" "
        bufs = [struct.pack("<I", len(hj)), hj,
                pts.tobytes(), inten.tobytes(), sem.tobytes(), boxes.tobytes()]
        return b"".join(bufs)

    async def frame(request):
        i = int(request.query.get("sweep", 0))
        j = int(request.query.get("frame", 0))
        use_nms = request.query.get("nms", "1") != "0"
        loop = asyncio.get_event_loop()
        body = await loop.run_in_executor(app["executor"], _frame_sync, i, j, use_nms)
        return web.Response(body=body, content_type="application/octet-stream",
                            headers={"Cache-Control": "no-store"})

    async def camimg(request):
        i = int(request.query.get("sweep", 0))
        cam = request.query["cam"]
        ts = request.query["ts"]
        sw = ds[i]
        path = os.path.join(sw.path, f"{cam}_{ts}.jpg")
        if not os.path.exists(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "max-age=3600"})

    # ---- OSM tiles (predownloaded tree + upstream fallback) -----------------
    async def osm_plan(request):
        i = int(request.query.get("sweep", 0))
        name = ds.sequence_names()[i]
        p = os.path.join(app["osm_dir"] or "", "plans", f"{name}.json") \
            if app["osm_dir"] else ""
        if p and os.path.exists(p):
            with open(p) as f:
                return web.json_response(json.load(f))
        return web.json_response(None)

    async def osm_roads(request):
        i = int(request.query.get("sweep", 0))
        name = ds.sequence_names()[i]
        p = os.path.join(app["osm_dir"] or "", "plans", f"{name}_roads.json") \
            if app["osm_dir"] else ""
        if p and os.path.exists(p):
            return web.FileResponse(p, headers={"Cache-Control": "max-age=86400"})
        return web.json_response({"ways": []})

    async def tile(request):
        z = int(request.match_info["z"])
        x = int(request.match_info["x"])
        y = int(request.match_info["y"])
        if app["osm_dir"]:
            path = os.path.join(app["osm_dir"], str(z), str(x), f"{y}.png")
            if os.path.exists(path):
                return web.FileResponse(
                    path, headers={"Cache-Control": "public, max-age=86400"})
        # fallback: upstream, best-effort cache to disk
        import aiohttp
        try:
            async with aiohttp.ClientSession(headers={"User-Agent": UA}) as sess:
                async with sess.get(OSM_UPSTREAM.format(z=z, x=x, y=y),
                                    timeout=aiohttp.ClientTimeout(total=8)) as r:
                    if r.status != 200:
                        raise web.HTTPNotFound()
                    data = await r.read()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise web.HTTPNotFound()
        if app["osm_dir"] and data[:8] == b"\x89PNG\r\n\x1a\n":
            try:
                path = os.path.join(app["osm_dir"], str(z), str(x), f"{y}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(data)
            except OSError:
                pass
        return web.Response(body=data, content_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})

    def _acc_sync(i):
        """One-directional per-track point accumulation (binary payload)."""
        if i in app["acc_cache"]:
            return app["acc_cache"][i]
        sw = ds[i]
        header, pts_blob, birth_blob = accumulate_sweep(sw)
        tid = _track_id_map(sw)
        header["tracks"] = {str(tid.get(u, -1)): t
                            for u, t in header["tracks"].items()}
        header["__pts_bytes"] = len(pts_blob)
        hj = json.dumps(header).encode()
        if len(hj) % 4:          # keep fp16 blobs 2-byte aligned
            hj += b" " * (4 - len(hj) % 4)
        body = (struct.pack("<I", len(hj)) + hj + pts_blob + birth_blob)
        if len(app["acc_cache"]) > 4:
            app["acc_cache"].clear()
        app["acc_cache"][i] = body
        return body

    async def acc_api(request):
        i = int(request.query.get("sweep", 0))
        loop = asyncio.get_event_loop()
        body = await loop.run_in_executor(app["executor"], _acc_sync, i)
        return web.Response(body=body, content_type="application/octet-stream",
                            headers={"Cache-Control": "no-store"})

    app.router.add_get("/", index)
    app.router.add_get("/api/sweeps", sweeps)
    app.router.add_get("/api/meta", meta)
    app.router.add_get("/api/gps_track", gps_track_api)
    app.router.add_get("/api/frame", frame)
    app.router.add_get("/api/camimg", camimg)
    app.router.add_get("/api/osm_plan", osm_plan)
    app.router.add_get("/api/osm_roads", osm_roads)
    app.router.add_get("/api/acc", acc_api)
    app.router.add_get("/api/tile/{z}/{x}/{y}.png", tile)
    app.router.add_static("/static/", STATIC_DIR, show_index=False)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--osm-dir", default=os.environ.get("OSM_TILE_DIR"),
                    help="predownloaded OSM tile tree (pandaset_pipe.osmtiles)")
    args = ap.parse_args()
    app = make_app(args.roots, osm_dir=args.osm_dir)
    print(f"serving on http://0.0.0.0:{args.port} (osm tiles: {args.osm_dir})")
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
