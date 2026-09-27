"""Ego-motion (deskew) compensation for the spinning lidar + object-motion
compensation for per-track point accumulation.

A spinning-lidar sweep spans ~100 ms while the ego keeps moving. Points stored
in the ego(frame ts) frame therefore mix measurement instants, which shears
the cloud ("rays cut the object"). Deskew references every point, measured at
its own time t_i = ts + rel_time, to the snapshot instant ts:

    p_snap = T_ego(ts)^-1 @ T_ego(t_i) @ p_stored

T_ego(t) is interpolated (lerp translation / slerp quaternion) between the
bracketing frame poses. If the dataset world points are already exact, this
transform is an identity for statics; if they were pushed with a single pose,
it removes exactly the ego-motion skew. Either way it never harms statics.

For points inside a moving object's box the object ALSO moves during the
sweep; `box_local_at` compensates it by referencing each point through the
BOX's world pose at the point's own time (track trajectory interpolated
between consecutive frames), which is what the acc_pts accumulation uses.
"""

import numpy as np


def _q_to_mat(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return np.array([
        [1-(yy+zz), xy-wz, xz+wy],
        [xy+wz, 1-(xx+zz), yz-wx],
        [xz-wy, yz+wx, 1-(xx+yy)]])


def _slerp_vec(q0, q1, a):
    """Batched quaternion slerp: q0/q1 (n,4) wxyz rows, a (n,) in [0,1]."""
    q0 = np.atleast_2d(np.asarray(q0, float))
    q1 = np.atleast_2d(np.asarray(q1, float))
    q0 = q0 / np.linalg.norm(q0, axis=1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    d = (q1 * q0).sum(axis=1)
    flip = d < 0
    q1 = q1.copy()
    q1[flip] *= -1
    d = np.abs(d)
    out = np.empty((len(a), 4))
    lin = d > 0.9995
    if lin.any():
        ql = q0[lin] + a[lin, None] * (q1[lin] - q0[lin])
        out[lin] = ql / np.linalg.norm(ql, axis=1, keepdims=True)
    k = ~lin
    if k.any():
        th = np.arccos(np.clip(d[k], -1, 1))
        s = np.sin(th)
        s = np.where(np.abs(s) < 1e-9, 1e-9, s)
        out[k] = (np.sin((1 - a[k]) * th) / s)[:, None] * q0[k] + \
                 (np.sin(a[k] * th) / s)[:, None] * q1[k]
    return out  # (n, 4)


def _pose_at(ts_frames, poses_t, poses_q, t_ms):
    """Pose (Q (n,4), tvec (n,3)) at times t_ms, bracketed by frame poses."""
    ts = np.asarray(ts_frames, float)
    n = len(ts)
    i1 = np.clip(np.searchsorted(ts, t_ms, side="left"), 1, n - 1)
    i0 = i1 - 1
    dt = ts[i1] - ts[i0]
    a = np.clip((t_ms - ts[i0]) / np.where(dt == 0, 1, dt), 0, 1)
    t0 = poses_t[i0]
    t1 = poses_t[i1]
    tvec = t0 + a[:, None] * (t1 - t0)
    Q = _slerp_vec(poses_q[i0], poses_q[i1], a)
    return Q, tvec


import os





def deskew(sw, j, pts, rel_time):
    """Vectorized deskew. Returns corrected (n,3) float32 in ego(ts) frame."""
    n = len(pts)
    if n == 0:
        return pts.astype(np.float32)
    ts_list = sw.timestamps
    m = len(ts_list)
    idx = []
    for k in (max(0, j - 1), j, min(m - 1, j + 1)):
        if k not in idx:
            idx.append(k)
    poses_t, poses_q, ts_arr = [], [], []
    for k in idx:
        lz = np.load(os.path.join(sw.path, f"lidar_{ts_list[k]}.npz"))
        poses_t.append(lz["ego2global_translation"].astype(np.float64))
        poses_q.append(lz["ego2global_rotation"].astype(np.float64))
        ts_arr.append(ts_list[k])
    poses_t = np.array(poses_t)
    poses_q = np.array(poses_q)
    ts_arr = np.array(ts_arr, float)
    ts_ref = float(ts_list[j])

    t_ms = ts_ref + rel_time.astype(np.float64) * 1000.0
    Q, tvec = _pose_at(ts_arr, poses_t, poses_q, t_ms)   # T(t_i): Q (n,4), tvec (n,3)

    # p_world_i = R(t_i) p_stored + t(t_i)  (standard single-pose model)
    w, x, y, z = Q.T
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    pw = np.einsum("nij,nj->ni", R, pts) + tvec

    # back to the snapshot frame: p_snap = T(ts_ref)^-1 p_world_i
    d0 = np.load(os.path.join(sw.path, f"lidar_{ts_list[j]}.npz"))
    q0 = d0["ego2global_rotation"].astype(np.float64)
    t0 = d0["ego2global_translation"].astype(np.float64)
    R0 = _q_to_mat(q0)
    p_rel = pw - t0
    out = (R0.T @ p_rel.T).T
    return out.astype(np.float32)
