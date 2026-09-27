"""Verify map localization for every sequence: fit quality + road-pixel check.

For each sweep:
  1. fit the robust world<->ENU alignment (geofit), report RMS and #used
  2. project the ego position (and a point 20 m ahead along the ego heading)
     onto the pre-downloaded OSM tile raster and check the pixel is
     road-colored (OSM roads render white / pale yellow: min(RGB) > 225)

Exit code 0 if all sequences pass, 1 otherwise.

    python -m pandaset_pipe.geocheck \
        --roots R1 R2 --osm-dir /path/to/pandaset_osm [--only 001 002]
"""

import argparse
import glob
import json
import math
import os

import numpy as np

from . import geofit
from .osmtiles import sweep_plan, lon2px, lat2py


def _load_quat_mat(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return np.array([
        [1-(yy+zz), xy-wz, xz+wy],
        [xy+wz, 1-(xx+zz), yz-wx],
        [xz-wy, yz+wx, 1-(xx+yy)],
    ])


def check_sweep(sweep_dir, osm_dir):
    import gzip
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow required for geocheck")

    lats, lons, e2g_t, e2g_q = [], [], [], []
    for f in sorted(glob.glob(os.path.join(sweep_dir, "gps_*.npz"))):
        ts = os.path.basename(f)[4:-4]
        d = np.load(f)
        lats.append(float(d["lat"])); lons.append(float(d["long"]))
        b = np.load(os.path.join(sweep_dir, f"boxes_{ts}.npz"))
        e2g_t.append(b["ego2global_translation"].astype(np.float64))
        e2g_q.append(b["ego2global_rotation"].astype(np.float64))
    n = len(lats)
    if n < 3:
        return {"seq": os.path.basename(sweep_dir), "skip": "no gps"}
    wxy = np.array([t[:2] for t in e2g_t])
    aln = geofit.fit_alignment(wxy, lats, lons)
    if aln is None:
        return {"seq": os.path.basename(sweep_dir), "skip": "no fit"}

    plan = sweep_plan(lats, lons)
    seq = os.path.basename(sweep_dir)[len("sweep_"):]
    if plan is None or not os.path.exists(os.path.join(osm_dir, "plans", f"{seq}.json")):
        return {"seq": seq, "rms": aln["rms"], "skip": "no tile plan"}

    cA, sA = math.cos(aln["rot"]), math.sin(aln["rot"])
    R = np.array([[cA, -sA], [sA, cA]])
    ca = math.cos(math.radians(aln["lat0"]))
    z, x0, y0 = plan["z"], plan["x0"], plan["y0"]
    ntx, nty = plan["ntx"], plan["nty"]
    # stitch the planned tiles once
    img = Image.new("RGB", (ntx * 256, nty * 256))
    for i in range(ntx):
        for j in range(nty):
            p = os.path.join(osm_dir, str(z), str(x0 + i), f"{y0 + j}.png")
            if os.path.exists(p):
                img.paste(Image.open(p).convert("RGB"), (i * 256, j * 256))
    px = np.asarray(img)

    # canvas origin in ENU
    m_per_px = (2 * math.pi * 6378137.0) * ca / (256 * 2 ** z)
    lon0, lat0 = plan["lon0"], plan["lat0"]

    def lonlat_px(lon, lat):
        u = lon2px(lon, z) - x0 * 256
        v = lat2py(lat, z) - y0 * 256
        return int(round(u)), int(round(v))

    # OSM carto road fill styles (exact constants of the standard render)
    ROAD_COLORS = np.array([
        [232, 146, 162],  # motorway
        [249, 178, 156],  # trunk
        [252, 214, 164],  # primary
        [247, 250, 191],  # secondary / tertiary
        [255, 255, 255],  # residential white
        [253, 253, 252],  # service / unpaved
        [235, 235, 235],  # living street / pedestrian-ish
        [255, 214, 209],  # motorway link
        [254, 229, 226],  # link-ish pink
    ])

    def road_ok(u, v):
        """True if a road-fill pixel occurs within ~2 m of the sample point."""
        if not (0 <= u < px.shape[1] and 0 <= v < px.shape[0]):
            return None
        r0, r1 = max(0, v - 4), min(px.shape[0], v + 5)
        c0, c1 = max(0, u - 4), min(px.shape[1], u + 5)
        win = px[r0:r1, c0:c1].reshape(-1, 3).astype(np.int32)
        for rc in ROAD_COLORS:
            d = np.abs(win - rc).max(axis=1)
            if (d <= 14).any():
                return True
        return False

    on_road = ahead_ok = checked = 0
    for k in range(0, n, 2):
        if not (lats[k] or lons[k]):
            continue
        # ego ENU from world pose via the fit (same path the viewer uses)
        e = aln["s"] * (R[0, 0] * e2g_t[k][0] + R[0, 1] * e2g_t[k][1]) + aln["t"][0]
        nn = aln["s"] * (R[1, 0] * e2g_t[k][0] + R[1, 1] * e2g_t[k][1]) + aln["t"][1]
        u, v = lonlat_px(lon0 + math.degrees(e / (6378137.0 * ca)),
                         lat0 + math.degrees(nn / 6378137.0))
        r1 = road_ok(u, v)
        # 20 m ahead along ego +X
        Rw = _load_quat_mat(e2g_q[k])
        pw = Rw @ np.array([20.0, 0.0, -1.7]) + e2g_t[k]
        e2 = aln["s"] * (R[0, 0] * pw[0] + R[0, 1] * pw[1]) + aln["t"][0]
        n2 = aln["s"] * (R[1, 0] * pw[0] + R[1, 1] * pw[1]) + aln["t"][1]
        u2, v2 = lonlat_px(lon0 + math.degrees(e2 / (6378137.0 * ca)),
                           lat0 + math.degrees(n2 / 6378137.0))
        r2 = road_ok(u2, v2)
        if r1 is None and r2 is None:
            continue
        checked += 1
        on_road += 1 if r1 else 0
        ahead_ok += 1 if r2 else 0
    pct = 100.0 * on_road / checked if checked else 0.0
    pct2 = 100.0 * ahead_ok / checked if checked else 0.0
    return {"seq": seq, "rms": round(aln["rms"], 2), "n_used": aln["n_used"],
            "z": z, "tiles": ntx * nty, "frames_checked": checked,
            "ego_on_road_pct": round(pct, 1), "ahead20_on_road_pct": round(pct2, 1),
            "ok": pct >= 60 and pct2 >= 60}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--osm-dir", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    sweeps = []
    for r in args.roots:
        sweeps += sorted(glob.glob(os.path.join(r, "sweep_*")))
    if args.only:
        sweeps = [s for s in sweeps if os.path.basename(s)[6:] in set(args.only)]

    n_ok = 0
    for s in sweeps:
        try:
            r = check_sweep(s, args.osm_dir)
        except Exception as e:  # noqa: BLE001
            r = {"seq": os.path.basename(s), "error": str(e)}
        if r.get("ok"):
            n_ok += 1
        print(json.dumps(r), flush=True)
    print(f"PASS {n_ok}/{len(sweeps)}")
    raise SystemExit(0 if n_ok == len(sweeps) else 1)


if __name__ == "__main__":
    main()
