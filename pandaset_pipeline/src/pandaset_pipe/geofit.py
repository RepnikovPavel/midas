"""Robust world(xyz of ego2global) <-> ENU(from GPS lat/lon) alignment.

Shared by the web server (map placement), the OSM tile predownloader and the
geocheck verifier. Fits the similarity  [e;n] = s R(rot) [x;y] + t  with
iterative outlier rejection (GPS glitches), and reports the RMS residual.
"""

import numpy as np


def enu_from_latlon(lat, lon, lat0, lon0):
    """Degree arrays -> ENU meters relative to (lat0, lon0)."""
    r = 6378137.0
    ca = np.cos(np.radians(lat0))
    e = np.radians(np.asarray(lon) - lon0) * r * ca
    n = np.radians(np.asarray(lat) - lat0) * r
    return e, n


def fit_alignment(wxy, lat, lon):
    """wxy: (N,2) ego2global translations; lat/lon: (N,) degrees.

    Returns {"s","rot","t","lat0","lon0","rms","n_used"} or None if it cannot
    be fit (too few valid points / degenerate scale).
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    ok = (lat != 0) | (lon != 0)
    ok &= ~np.isnan(wxy[:, 0])
    if ok.sum() < 3:
        return None
    i0 = int(np.flatnonzero(ok)[0])
    lat0, lon0 = float(lat[i0]), float(lon[i0])
    e, n = enu_from_latlon(lat, lon, lat0, lon0)
    G_all = np.stack([e, n], 1)
    sel = ok.copy()
    s, R, t = 0.0, np.eye(2), np.zeros(2)
    for _ in range(4):
        Wc = wxy[sel].mean(0)
        Gc = G_all[sel].mean(0)
        W0 = wxy[sel] - Wc
        G0 = G_all[sel] - Gc
        U, S, Vt = np.linalg.svd(W0.T @ G0)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, d]) @ U.T
        s = float((S[0] + d * S[1]) / max((W0 ** 2).sum(), 1e-9))
        t = Gc - s * R @ Wc
        if sel.sum() <= 4:
            break
        res = np.linalg.norm(G_all - (s * wxy @ R.T + t), axis=1)
        r_sel = res[sel]
        mad = np.median(np.abs(r_sel - np.median(r_sel)))
        new = ok & (res <= max(3.0 * 1.4826 * mad, 2.0))
        if new.sum() < 4:
            break
        if (new == sel).all():
            sel = new
            break
        sel = new
    res = np.linalg.norm(G_all - (s * wxy @ R.T + t), axis=1)
    rms = float(np.sqrt((res[sel] ** 2).mean())) if sel.any() else 1e9
    if not (s > 0.5):
        return None
    return {"s": s, "rot": float(np.arctan2(R[1, 0], R[0, 0])),
            "t": [float(t[0]), float(t[1])],
            "lat0": lat0, "lon0": lon0, "rms": rms, "n_used": int(sel.sum())}


def apply_alignment(aln, wxy):
    """world (N,2) -> ENU (N,2) via the fitted similarity."""
    c, s = np.cos(aln["rot"]), np.sin(aln["rot"])
    R = np.array([[c, -s], [s, c]])
    return aln["s"] * wxy @ R.T + np.asarray(aln["t"])
