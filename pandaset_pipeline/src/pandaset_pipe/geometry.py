"""Geometry helpers for PandaSet frame conversions.

Frames:
  world  - static world frame (pandaset lidar points and poses are given in it)
  veh    - pandaset ego/vehicle frame = lidar sensor frame (native pandaset "ego")
  ego    - OUR normalized ego frame: X forward, Y left, Z up

`AXIS_MAP` converts veh -> ego. pandaset vehicle frame is Y-forward / X-right
(verified empirically by scripts/validate_frames.py against GPS velocity and
pose deltas), hence veh->ego is a +90 deg rotation about Z:
  ego_x =  veh_y   (forward)
  ego_y = -veh_x   (left)
  ego_z =  veh_z   (up)
"""

import numpy as np

AXIS_MAP = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])
AXIS_MAP_INV = AXIS_MAP.T

AXIS_MAP_4 = np.eye(4)
AXIS_MAP_4[:3, :3] = AXIS_MAP
AXIS_MAP_INV_4 = np.eye(4)
AXIS_MAP_INV_4[:3, :3] = AXIS_MAP_INV


def quat_to_mat(q):
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def mat_to_quat(R):
    """3x3 rotation matrix -> quaternion (w, x, y, z). Shewchuk-style stable branch."""
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def pose_to_mat(pose):
    """pandaset pose dict {'heading': quat wxyz, 'position': xyz} -> 4x4 (world_from_sensor)."""
    q = [pose["heading"]["w"], pose["heading"]["x"], pose["heading"]["y"], pose["heading"]["z"]]
    p = [pose["position"]["x"], pose["position"]["y"], pose["position"]["z"]]
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(q)
    T[:3, 3] = p
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T, pts):
    """Apply 4x4 to (N,3)."""
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def world_to_ego(T_world_veh, pts_world):
    """world points -> our ego frame (X fwd, Y left)."""
    return (AXIS_MAP @ (invert_T(T_world_veh)[:3, :3] @ pts_world.T + invert_T(T_world_veh)[:3, 3:4])).T


def ego2global(T_world_veh):
    """4x4 world_from_ego for our ego frame."""
    return T_world_veh @ AXIS_MAP_INV_4


def pandaset_box_to_ego(pos_world, dim_xyz, yaw_ps, T_world_veh):
    """Convert one pandaset cuboid to our ego box (x, y, z, dx, dy, dz, yaw).

    pandaset doc: yaw is CCW about world Z; yaw=0 -> box length axis points along
    world +Y; yaw=pi/2 -> along -X. dimensions: x=width, y=length, z=height.

    Our convention: yaw CCW about ego Z from ego +X; dx = length along heading,
    dy = width, dz = height.
    """
    T_veh_world = invert_T(T_world_veh)
    pos_veh = T_veh_world[:3, :3] @ pos_world + T_veh_world[:3, 3]
    pos_ego = AXIS_MAP @ pos_veh

    # heading direction in world: yaw_ps=0 -> +Y, pi/2 -> -X  =>  d = (-sin, cos)
    heading_world_angle = np.arctan2(np.cos(yaw_ps), -np.sin(yaw_ps))  # == yaw_ps + pi/2
    # ego +X axis expressed in world:
    T_world_ego = T_world_veh @ AXIS_MAP_INV_4
    ego_x_world = T_world_ego[:3, 0]
    ego_x_world_angle = np.arctan2(ego_x_world[1], ego_x_world[0])
    yaw_ego = heading_world_angle - ego_x_world_angle
    yaw_ego = (yaw_ego + np.pi) % (2 * np.pi) - np.pi

    length = dim_xyz[1]  # pandaset y dim = length (front to back)
    width = dim_xyz[0]   # pandaset x dim = width (left to right)
    height = dim_xyz[2]
    return np.array([pos_ego[0], pos_ego[1], pos_ego[2], length, width, height, yaw_ego])


def quat_mul(a, b):
    wa, xa, ya, za = a
    wb, xb, yb, zb = b
    return np.array([
        wa * wb - xa * xb - ya * yb - za * zb,
        wa * xb + xa * wb + ya * zb - za * yb,
        wa * yb - xa * zb + ya * wb + za * xb,
        wa * zb + xa * yb - ya * xb + za * wb,
    ])
