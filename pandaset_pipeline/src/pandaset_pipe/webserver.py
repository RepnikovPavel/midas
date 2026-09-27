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
