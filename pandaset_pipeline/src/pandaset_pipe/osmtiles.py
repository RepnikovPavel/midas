"""Pre-download OSM raster tiles covering every sequence's GPS track.

Run ONCE on the server (pipeline container, both dataset disks mounted):

    python -m pandaset_pipe.osmtiles \
        --roots /mnt/hdd1/datasets/pandaset_npz /mnt/hdd2/datasets/pandaset_npz \
        --out /mnt/hdd1/datasets/pandaset_osm

Produces:
    <out>/<z>/<x>/<y>.png      tiles (shared across sequences)
    <out>/plans/<SEQ>.json     per-sweep tile plan: z, x0, y0, ntx, nty, lat0, lon0

The plan constants MUST match the viewer (z in [15,18], margin 120 m,
bbox capped at 3840 px) — the viewer reads the plan, it never recomputes it.
Idempotent: existing tiles are skipped. Concurrency and retries included.
"""

import argparse
import concurrent.futures
import io
import json
import math
import os
import queue
import threading
import time
import urllib.request

import numpy as np

from . import geofit

R_EARTH = 6378137.0
CIRC = 2 * math.pi * R_EARTH
MARGIN_M = 250.0          # lidar range ~200 m -> keep map well past the route
CANVAS_MAX_PX = 3840.0
ZMIN, ZMAX = 15, 18
UA = "pandaset-pipeline-osm-preload/1.0 (contact: dataset viewer; bulk: one-time)"
UPSTREAM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OVERPASS = [  # rotate: public instances, all share the same OSM data
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",  # most reliable here
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
ROADS_QUERY = ("[out:json][timeout:180];"
               "way['highway']({s},{w},{n},{e});out geom;")


def lon2px(lon, z):
    return (lon / 360.0 + 0.5) * 256 * 2 ** z


def lat2py(lat, z):
    s = math.sin(math.radians(lat))
    return (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * 256 * 2 ** z


def sweep_plan(lat, lon):
    """GPS arrays -> tile plan dict or None (too few points)."""
    pts = [(la, lo) for la, lo in zip(lat, lon) if la or lo]
    if len(pts) < 2:
        return None
    lat0, lon0 = pts[0]
    ca = math.cos(math.radians(lat0))
    e, n = geofit.enu_from_latlon([p[0] for p in pts], [p[1] for p in pts], lat0, lon0)
    e0, e1 = float(e.min()) - MARGIN_M, float(e.max()) + MARGIN_M
    n0, n1 = float(n.min()) - MARGIN_M, float(n.max()) + MARGIN_M

    z = ZMAX
    while z > ZMIN and max(e1 - e0, n1 - n0) / (CIRC * ca / (256 * 2 ** z)) > CANVAS_MAX_PX:
        z -= 1

    def lon_at(ee):
        return lon0 + math.degrees(ee / (R_EARTH * ca))

    def lat_at(nn):
        return lat0 + math.degrees(nn / R_EARTH)

    px0, px1 = lon2px(lon_at(e0), z), lon2px(lon_at(e1), z)
    py0, py1 = lat2py(lat_at(n1), z), lat2py(lat_at(n0), z)  # py grows southwards
    x0, x1 = math.floor(px0 / 256), math.floor((px1 - 1e-6) / 256)
    y0, y1 = math.floor(py0 / 256), math.floor((py1 - 1e-6) / 256)
    return {"z": z, "x0": x0, "y0": y0, "ntx": x1 - x0 + 1, "nty": y1 - y0 + 1,
            "lat0": lat0, "lon0": lon0}


def fetch_tile(z, x, y, out_dir, retries=3):
    """Download one tile to out_dir (returns 'hit' | 'ok' | 'fail')."""
    path = os.path.join(out_dir, str(z), str(x), f"{y}.png")
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return "hit"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = UPSTREAM.format(z=z, x=x, y=y)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
            if data[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("not a png")
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
            return "ok"
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  FAIL tile {z}/{x}/{y}: {e}", flush=True)
                return "fail"
            time.sleep(1.0 + attempt)


def plan_bbox_latlon(plan):
    """(south, west, north, east) covered by the plan's tile grid."""
    z = plan["z"]
    n2 = 256 * 2 ** z
    west = ((plan["x0"] * 256) / n2 - 0.5) * 360.0
    east = (((plan["x0"] + plan["ntx"]) * 256) / n2 - 0.5) * 360.0
    def py2lat(py):
        r = 2 * math.atan(math.exp((0.5 - py / n2) * 2 * math.pi)) - math.pi / 2
        return math.degrees(r)
    north = py2lat(plan["y0"] * 256)
    south = py2lat((plan["y0"] + plan["nty"]) * 256)
    return south, west, north, east


_road_ep = {"i": 0, "ok": None}

def fetch_roads(plan, out_path, retries=5):
    """Download the OSM road graph (highway ways) via rotating Overpass mirrors.
    Tries the last successful mirror first (sticky)."""
    import urllib.parse
    s, w, n, e = plan_bbox_latlon(plan)
    q = ROADS_QUERY.format(s=f"{s:.6f}", w=f"{w:.6f}", n=f"{n:.6f}", e=f"{e:.6f}")
    order = []
    start = _road_ep["ok"] if _road_ep["ok"] is not None else _road_ep["i"]
    for k in range(len(OVERPASS)):
        order.append(OVERPASS[(start + k) % len(OVERPASS)])
    for attempt in range(retries):
        ep = order[min(attempt, len(order) - 1)]
        try:
            req = urllib.request.Request(
                ep + "?data=" + urllib.parse.quote(q), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=200) as r:
                data = json.loads(r.read().decode("utf-8"))
            ways = []
            for el in data.get("elements", []):
                if el.get("type") == "way" and "geometry" in el:
                    ways.append([[p["lat"], p["lon"]] for p in el["geometry"]])
            with open(out_path + ".tmp", "w") as f:
                json.dump({"ways": ways}, f)
            os.replace(out_path + ".tmp", out_path)
            _road_ep["ok"] = OVERPASS.index(ep)
            _road_ep["i"] += 1
            return len(ways)
        except Exception as ex:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  roads FAIL {ep}: {ex}", flush=True)
                return -1
            time.sleep(2.0)


def main():
    import glob as _glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel tile downloads (be polite to OSM)")
    ap.add_argument("--skip-roads", action="store_true",
                    help="do not fetch the OSM road graphs (Overpass)")
    args = ap.parse_args()

    seqs = []
    for r in args.roots:
        for d in sorted(_glob.glob(os.path.join(r, "sweep_*"))):
            seqs.append(os.path.basename(d)[len("sweep_"):])
    if args.only:
        seqs = [s for s in seqs if s in set(args.only)]
    print(f"{len(seqs)} sequences; tiles -> {args.out}")

    os.makedirs(os.path.join(args.out, "plans"), exist_ok=True)
    plans = {}
    jobs = queue.Queue()
    n_tiles = 0
    for seq in seqs:
        sweep_dir = None
        for r in args.roots:
            d = os.path.join(r, f"sweep_{seq}")
            if os.path.isdir(d):
                sweep_dir = d
                break
        lats, lons = [], []
        for f in sorted(_glob.glob(os.path.join(sweep_dir, "gps_*.npz"))):
            d = np.load(f)
            lats.append(float(d["lat"]))
            lons.append(float(d["long"]))
        plan = sweep_plan(lats, lons)
        if plan is None:
            print(f"  {seq}: no gps, skipped")
            continue
        with open(os.path.join(args.out, "plans", f"{seq}.json"), "w") as f:
            json.dump(plan, f)
        plans[seq] = plan
        for i in range(plan["ntx"]):
            for j in range(plan["nty"]):
                jobs.put((plan["z"], plan["x0"] + i, plan["y0"] + j))
                n_tiles += 1
    print(f"planned {n_tiles} tile downloads (dedup on disk)")

    stats = {"hit": 0, "ok": 0, "fail": 0}
    lock = threading.Lock()

    def worker():
        while True:
            try:
                z, x, y = jobs.get_nowait()
            except queue.Empty:
                return
            res = fetch_tile(z, x, y, args.out)
            with lock:
                stats[res] += 1

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda _: worker(), range(args.workers)))
    print(f"tiles: cached={stats['hit']} downloaded={stats['ok']} "
          f"failed={stats['fail']} in {time.time() - t0:.0f}s")

    # ---- OSM road graphs (Overpass), one file per sequence ------------------
    if not args.skip_roads:
        todo = [(seq, plan) for seq, plan in sorted(plans.items())
                if not os.path.exists(
                    os.path.join(args.out, "plans", f"{seq}_roads.json"))]
        lock2 = threading.Lock()
        done_ct = {"ok": 0, "fail": 0}

        def road_job(sp):
            seq, plan = sp
            out_path = os.path.join(args.out, "plans", f"{seq}_roads.json")
            ok = fetch_roads(plan, out_path) >= 0
            with lock2:
                done_ct["ok" if ok else "fail"] += 1
                n = done_ct["ok"] + done_ct["fail"]
                if n % 10 == 0:
                    print(f"  roads: {n}/{len(todo)}", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            list(ex.map(road_job, todo))
        print(f"roads: ok={done_ct['ok']} failed={done_ct['fail']}")
    print("DONE" if stats["fail"] == 0 else "DONE (with failures)")


if __name__ == "__main__":
    main()
