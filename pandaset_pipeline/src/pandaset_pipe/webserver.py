"""Web backend for the PandaSet viewer (serves frontend + frame data).

Run:  python -m pandaset_pipe.webserver --roots R1 R2 --port 8777
Open: http://localhost:8777
"""
import argparse
import json
import os
import struct

import numpy as np
from aiohttp import web

from .reader import PandaDataset

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web_static")


def make_app(roots):
    ds = PandaDataset(roots)
    if len(ds) == 0:
        raise SystemExit("no sweeps found in " + ", ".join(roots))
    app = web.Application()
    app["ds"] = ds
    app["gps_cache"] = {}      # sweep idx -> track arrays
    app["classes_cache"] = {}  # sweep idx -> {class: count}

    async def index(request):
        return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))

    async def sweeps(request):
        return web.json_response({
            "sweeps": ds.sequence_names(),
            "counts": [len(ds[i]) for i in range(len(ds))],
        })

    def gps_track(i):
        """Full per-frame GPS track of a sweep (lat, lon, speed kmh); cached."""
        if i in app["gps_cache"]:
            return app["gps_cache"][i]
        sw = ds[i]
        lat = np.zeros(len(sw)); lon = np.zeros(len(sw)); spd = np.zeros(len(sw))
        for j in range(len(sw)):
            g = sw[j].gps
            lat[j] = g.get("lat", 0.0)
            lon[j] = g.get("long", 0.0)
            spd[j] = g.get("speed", 0.0)
        out = {"lat": lat.tolist(), "lon": lon.tolist(), "speed": (spd * 3.6).tolist(),
               "ts": sw.timestamps}
        app["gps_cache"][i] = out
        return out

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
            "box_classes": box_classes(i),
            "timestamps": sw.timestamps,
        })

    async def gps_track_api(request):
        i = int(request.query.get("sweep", 0))
        return web.json_response(gps_track(i))

    async def frame(request):
        i = int(request.query.get("sweep", 0))
        j = int(request.query.get("frame", 0))
        use_nms = request.query.get("nms", "1") != "0"
        sw = ds[i]
        j = max(0, min(j, len(sw) - 1))
        snap = sw[j]
        pts = snap.points.astype(np.float32, copy=False)
        inten = snap.intensity
        sem = snap.semseg
        if sem is None:
            sem = np.zeros(len(pts), np.uint8)
        if use_nms:
            boxes, labels = snap.boxes_nms()
        else:
            boxes, labels = snap.boxes, snap.box_labels
        boxes = boxes.astype(np.float32, copy=False)
        # per-camera: matched ts (for image url) + projection matrices
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
        }
        hj = json.dumps(header).encode()
        bufs = [struct.pack("<I", len(hj)), hj,
                pts.tobytes(), inten.tobytes(), sem.tobytes(), boxes.tobytes()]
        return web.Response(body=b"".join(bufs),
                            content_type="application/octet-stream")

    async def camimg(request):
        i = int(request.query.get("sweep", 0))
        cam = request.query["cam"]
        ts = request.query["ts"]
        sw = ds[i]
        path = os.path.join(sw.path, f"{cam}_{ts}.jpg")
        if not os.path.exists(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "max-age=3600"})

    app.router.add_get("/", index)
    app.router.add_get("/api/sweeps", sweeps)
    app.router.add_get("/api/meta", meta)
    app.router.add_get("/api/gps_track", gps_track_api)
    app.router.add_get("/api/frame", frame)
    app.router.add_get("/api/camimg", camimg)
    app.router.add_static("/static/", STATIC_DIR, show_index=False)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()
    app = make_app(args.roots)
    print(f"serving on http://0.0.0.0:{args.port}")
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
