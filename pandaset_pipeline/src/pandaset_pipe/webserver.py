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

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web_static")
OSM_UPSTREAM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
UA = "pandaset-pipeline-viewer/1.0"


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
    """Boxes + ego pose interpolated onto a camera's own timestamp.

    Cameras are not synchronized with the lidar: projecting lidar-frame boxes
    onto a camera image shows a motion offset. Boxes (matched by uuid) are
    interpolated linearly in the world frame between the surrounding lidar
    frames; the ego pose is lerped/slerped the same way.

    Returns (boxes, labels, e2g) where boxes is (M,7) [x y z_world dx dy dz
    yaw_world] and e2g is [tx ty tz qw qx qy qz] at ts_cam.
    """
    import bisect
    ts = sw.timestamps
    n = len(ts)
    if n == 0:
        return None
    i = bisect.bisect_left(ts, ts_cam)
    j1 = min(max(i, 0), n - 1)
    j0 = min(max(i - 1, 0), n - 1)
    # per-sensor sweep time offsets for both bracketing frames
    off0 = _sensor_time_offsets(sw, j0, cur_idx, cur_offsets or {})
    off1 = _sensor_time_offsets(sw, j1, cur_idx, cur_offsets or {})

    def load(j, off):
        p = os.path.join(sw.path, f"boxes_{ts[j]}.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        t = d["ego2global_translation"].astype(np.float64)
        q = d["ego2global_rotation"].astype(np.float64)
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                         1 - 2 * (q[2] ** 2 + q[3] ** 2))
        b = d["boxes"].astype(np.float64)
        labels = d["class_names"]
        uuids = d["uuids"]
        sids = d["sensor_id"]
        if use_nms and "nms_keep" in d.files:
            keep = d["nms_keep"].astype(bool)
            b, labels, uuids, sids = b[keep], labels[keep], uuids[keep], sids[keep]
        # yaw-only ego rotation: pandaset world poses are essentially yaw + tiny
        # pitch/roll; box centers sit near the ground so the error is cm-scale
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        cen_w = (Rz @ b[:, :3].T).T + t
        out = {}
        for k in range(len(b)):
            sid = int(sids[k]) if sids[k] is not None else -1
            # this box's own capture time: its sensor's sweep offset
            tb = ts[j] + off.get(sid, 0.0) * 1000.0
            out[str(uuids[k])] = (cen_w[k], b[k, 3], b[k, 4], b[k, 5],
                                  b[k, 6] + yaw, str(labels[k]), tb, sid)
        return out, t, q

    L0 = load(j0, off0)
    L1 = load(j1, off1)
    if L0 is None and L1 is None:
        return None
    if L0 is None:
        L0 = L1
    if L1 is None:
        L1 = L0
    b0, t0, q0 = L0
    b1, t1, q1 = L1
    dt = float(ts[j1] - ts[j0])
    # ego pose interpolated onto the camera timestamp
    a_pose = min(max((ts_cam - ts[j0]) / dt, 0.0), 1.0) if dt else 0.0
    e2g = np.concatenate([t0 + a_pose * (t1 - t0), _quat_slerp(q0, q1, a_pose)])
    boxes, labels = [], []
    for uuid, (c0, dx, dy, dz, yaw0, lab, tb0, sid) in b0.items():
        # per-box interpolation fraction on the box's own sensor timeline
        a = min(max((ts_cam - tb0) / dt, 0.0), 1.0) if dt else 0.0
        if uuid in b1:
            c1, _, _, _, yaw1, _, _, _ = b1[uuid]
            cen = c0 + a * (c1 - c0)
            dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
            yaw = yaw0 + a * dyaw
        else:
            cen, yaw = c0, yaw0
        boxes.append([cen[0], cen[1], cen[2], dx, dy, dz, yaw])
        labels.append(lab)
    return np.array(boxes, dtype=np.float64), labels, e2g.tolist()


def make_app(roots, osm_dir=None):
    ds = PandaDataset(roots)
    if len(ds) == 0:
        raise SystemExit("no sweeps found in " + ", ".join(roots))
    app = web.Application()
    app["ds"] = ds
    app["osm_dir"] = osm_dir
    app["gps_cache"] = {}      # sweep idx -> track dict
    app["classes_cache"] = {}  # sweep idx -> {class: count}
    app["executor"] = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    async def index(request):
        return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))

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
                boxes_c, labels_c, e2g_c = ib
                cams[cam]["ib"] = boxes_c.astype(float).tolist()
                cams[cam]["ib_labels"] = labels_c
                cams[cam]["ib_e2g"] = e2g_c
        header = {
            "ts": snap.ts, "n": len(pts), "nb": len(boxes),
            "labels": [str(x) for x in labels],
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

    app.router.add_get("/", index)
    app.router.add_get("/api/sweeps", sweeps)
    app.router.add_get("/api/meta", meta)
    app.router.add_get("/api/gps_track", gps_track_api)
    app.router.add_get("/api/frame", frame)
    app.router.add_get("/api/camimg", camimg)
    app.router.add_get("/api/osm_plan", osm_plan)
    app.router.add_get("/api/osm_roads", osm_roads)
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
