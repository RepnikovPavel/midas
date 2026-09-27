"""Numba-jitted hot paths for conversion and rendering.

Everything loop-heavy lives here and is compiled with @njit(cache=True).
Numpy BLAS-bound bulk math (big matmuls) stays in the callers.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------- boxes
@njit(cache=True)
def box_corners_batch(boxes):
    """boxes (M,7) x y z dx dy dz yaw -> corners (M,8,3)."""
    m = boxes.shape[0]
    out = np.empty((m, 8, 3), np.float32)
    cu = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                   [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])
    for i in range(m):
        x, y, z = boxes[i, 0], boxes[i, 1], boxes[i, 2]
        dx, dy, dz = boxes[i, 3], boxes[i, 4], boxes[i, 5]
        yaw = boxes[i, 6]
        c = np.cos(yaw)
        s = np.sin(yaw)
        for j in range(8):
            lx = cu[j, 0] * dx
            ly = cu[j, 1] * dy
            lz = cu[j, 2] * dz
            out[i, j, 0] = c * lx - s * ly + x
            out[i, j, 1] = s * lx + c * ly + y
            out[i, j, 2] = lz + z
    return out


@njit(cache=True)
def box_lines(boxes):
    """boxes (M,7) -> line vertices (M*24, 3) for segment rendering."""
    corners = box_corners_batch(boxes)
    m = boxes.shape[0]
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]])
    lines = np.empty((m * 24, 3), np.float32)
    for i in range(m):
        base = i * 24
        for k in range(12):
            p1 = edges[k, 0]
            p2 = edges[k, 1]
            lines[base + k * 2] = corners[i, p1]
            lines[base + k * 2 + 1] = corners[i, p2]
    return lines


# ---------------------------------------------------------------- NMS
@njit(cache=True)
def _bev_iou(a, b):
    """a,b: (x, y, dx, dy, yaw) axis-aligned approx (yaw ignored if small set)."""
    # conservative: treat as axis-aligned rectangles (matches nms_bev_jit usage)
    ax1 = a[0] - a[2] / 2
    ax2 = a[0] + a[2] / 2
    ay1 = a[1] - a[3] / 2
    ay2 = a[1] + a[3] / 2
    bx1 = b[0] - b[2] / 2
    bx2 = b[0] + b[2] / 2
    by1 = b[1] - b[3] / 2
    by2 = b[1] + b[3] / 2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    if ua <= 0:
        return 0.0
    return inter / ua


@njit(cache=True)
def bev_nms(boxes_bev, scores, iou_thresh):
    """boxes_bev (M,5) x y dx dy yaw, scores (M,) -> keep mask (M,) bool."""
    m = boxes_bev.shape[0]
    order = np.argsort(-scores)
    keep = np.ones(m, np.bool_)
    for oi in range(m):
        i = order[oi]
        if not keep[i]:
            continue
        for oj in range(oi + 1, m):
            j = order[oj]
            if not keep[j]:
                continue
            if _bev_iou(boxes_bev[i], boxes_bev[j]) > iou_thresh:
                keep[j] = False
    return keep


# ------------------------------------------------------- camera splatting
@njit(cache=True)
def project_depth_colors(pts_cam, K, img_h, img_w):
    """Points in camera frame -> per-pixel depth-colored overlay.

    Returns (us, vs, colors_bgr) for points inside the image, z-buffered
    (nearest point wins per pixel).
    """
    n = pts_cam.shape[0]
    depth_img = np.full((img_h, img_w), 1e30, np.float32)
    for i in range(n):
        z = pts_cam[i, 2]
        if z < 0.5:
            continue
        u = K[0, 0] * pts_cam[i, 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[i, 1] / z + K[1, 2]
        iu = int(u)
        iv = int(v)
        if 0 <= iu < img_w and 0 <= iv < img_h:
            if z < depth_img[iv, iu]:
                depth_img[iv, iu] = z
    # find max depth for normalization
    dmax = 1.0
    for iv in range(img_h):
        for iu in range(img_w):
            d = depth_img[iv, iu]
            if d < 1e29 and d > dmax:
                dmax = d
    us = []
    vs = []
    cols = []
    for iv in range(img_h):
        for iu in range(img_w):
            d = depth_img[iv, iu]
            if d < 1e29:
                nd = d / dmax
                if nd > 1.0:
                    nd = 1.0
                us.append(iu)
                vs.append(iv)
                cols.append((int(255 * (1 - nd)), 0, int(255 * nd)))
    return us, vs, cols


@njit(cache=True)
def splat_depth_overlay(img, pts_cam, fx, fy, cx, cy):
    """Z-buffered depth overlay written directly into img (BGR uint8), in-place."""
    n = pts_cam.shape[0]
    h = img.shape[0]
    w = img.shape[1]
    depth_img = np.full((h, w), 1e30, np.float32)
    for i in range(n):
        z = pts_cam[i, 2]
        if z < 0.5:
            continue
        iu = int(fx * pts_cam[i, 0] / z + cx)
        iv = int(fy * pts_cam[i, 1] / z + cy)
        if 0 <= iu < w and 0 <= iv < h and z < depth_img[iv, iu]:
            depth_img[iv, iu] = z
    dmax = 1.0
    for iv in range(h):
        for iu in range(w):
            d = depth_img[iv, iu]
            if d < 1e29 and d > dmax:
                dmax = d
    for iv in range(h):
        for iu in range(w):
            d = depth_img[iv, iu]
            if d < 1e29:
                nd = d / dmax
                img[iv, iu, 0] = np.uint8(255 * (1 - nd))
                img[iv, iu, 1] = 0
                img[iv, iu, 2] = np.uint8(255 * nd)
    return img


@njit(cache=True)
def semseg_colors(classes_u8, palette):
    """classes (N,) u8 + palette (256,3) f32 -> (N,3) uint8 RGB."""
    n = classes_u8.shape[0]
    out = np.empty((n, 3), np.uint8)
    for i in range(n):
        c = classes_u8[i]
        out[i, 0] = np.uint8(min(max(palette[c, 0] * 255, 0), 255))
        out[i, 1] = np.uint8(min(max(palette[c, 1] * 255, 0), 255))
        out[i, 2] = np.uint8(min(max(palette[c, 2] * 255, 0), 255))
    return out


@njit(cache=True)
def height_colors(z, z_min, z_max):
    """z (N,) -> (N,3) uint8 RGB height colormap (blue->red)."""
    n = z.shape[0]
    out = np.empty((n, 3), np.uint8)
    span = z_max - z_min
    for i in range(n):
        t = (z[i] - z_min) / span
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        out[i, 0] = np.uint8(255 * t)
        out[i, 1] = np.uint8(102)
        out[i, 2] = np.uint8(255 * (1.0 - t))
    return out


# ----------------------------------------------------- box transformation
@njit(cache=True)
def pandaset_boxes_to_ego(pos_world, dims, yaws, R_veh_world, t_veh_world,
                          R_axis, ego_x_world_angle):
    """Vectorized pandaset cuboids -> ego boxes (M,7).

    pos_world (M,3); dims (M,3) pandaset (width,length,height);
    yaws (M,) pandaset yaw (CCW from +Y, length axis);
    R_veh_world/t_veh_world: vehicle-from-world; R_axis: veh->ego rotation.
    """
    m = pos_world.shape[0]
    out = np.empty((m, 7), np.float64)
    for i in range(m):
        px = R_veh_world[0, 0] * pos_world[i, 0] + R_veh_world[0, 1] * pos_world[i, 1] \
            + R_veh_world[0, 2] * pos_world[i, 2] + t_veh_world[0]
        py = R_veh_world[1, 0] * pos_world[i, 0] + R_veh_world[1, 1] * pos_world[i, 1] \
            + R_veh_world[1, 2] * pos_world[i, 2] + t_veh_world[1]
        pz = R_veh_world[2, 0] * pos_world[i, 0] + R_veh_world[2, 1] * pos_world[i, 1] \
            + R_veh_world[2, 2] * pos_world[i, 2] + t_veh_world[2]
        ex = R_axis[0, 0] * px + R_axis[0, 1] * py + R_axis[0, 2] * pz
        ey = R_axis[1, 0] * px + R_axis[1, 1] * py + R_axis[1, 2] * pz
        ez = R_axis[2, 0] * px + R_axis[2, 1] * py + R_axis[2, 2] * pz
        # heading world std angle = yaw + pi/2 (pandaset yaw: 0 -> +Y length axis)
        yaw_ego = yaws[i] + np.pi / 2.0 - ego_x_world_angle
        # wrap to [-pi, pi)
        yaw_ego = (yaw_ego + np.pi) % (2 * np.pi) - np.pi
        out[i, 0] = ex
        out[i, 1] = ey
        out[i, 2] = ez
        out[i, 3] = dims[i, 1]  # length along heading
        out[i, 4] = dims[i, 0]  # width
        out[i, 5] = dims[i, 2]  # height
        out[i, 6] = yaw_ego
    return out


@njit(cache=True)
def world_points_to_ego(pts_world, R_veh_world, t_veh_world, R_axis):
    """(N,3) world -> ego via vehicle frame. Returns (N,3) float64."""
    n = pts_world.shape[0]
    out = np.empty((n, 3), np.float64)
    for i in range(n):
        px = R_veh_world[0, 0] * pts_world[i, 0] + R_veh_world[0, 1] * pts_world[i, 1] \
            + R_veh_world[0, 2] * pts_world[i, 2] + t_veh_world[0]
        py = R_veh_world[1, 0] * pts_world[i, 0] + R_veh_world[1, 1] * pts_world[i, 1] \
            + R_veh_world[1, 2] * pts_world[i, 2] + t_veh_world[1]
        pz = R_veh_world[2, 0] * pts_world[i, 0] + R_veh_world[2, 1] * pts_world[i, 1] \
            + R_veh_world[2, 2] * pts_world[i, 2] + t_veh_world[2]
        out[i, 0] = R_axis[0, 0] * px + R_axis[0, 1] * py + R_axis[0, 2] * pz
        out[i, 1] = R_axis[1, 0] * px + R_axis[1, 1] * py + R_axis[1, 2] * pz
        out[i, 2] = R_axis[2, 0] * px + R_axis[2, 1] * py + R_axis[2, 2] * pz
    return out
