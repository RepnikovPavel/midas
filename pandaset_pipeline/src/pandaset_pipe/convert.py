"""Convert one raw extracted PandaSet sequence into the npz sweep layout.

Per frame (spine = lidar frames, 80 per sequence):
  lidar_<ts_ms>.npz    points fp16 (N,3) ego[X fwd,Y left,Z up], intensity u8,
                       sensor_id u8 (0=Pandar64, 1=PandarGT), rel_time f32,
                       ego2global (translation f32, rotation quat wxyz f32)
  boxes_<ts_ms>.npz    boxes f32 (M,7) [x y z dx(length) dy(width) dz yaw@+X CCW],
                       class_names, uuids, stationary, object_motion,
                       sensor_id (cuboids.sensor_id), sibling_id
  semseg_<ts_ms>.npz   classes u8 (N,) aligned with lidar points (index order)
  gps_<ts_ms>.npz      lat long height xvel yvel speed
  <cam>_<camts>.jpg    copied camera image
  <cam>_<camts>.npz    intrinsic K f32 (3,3), sensor2ego (t f32, quat wxyz f32),
                       ego2global
  sweep_meta.json      sequence-level info incl. semseg classes mapping
"""

import glob
import gzip
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd

from . import fastops
from . import geometry as G

# per-sensor confidence for BEV NMS dedup of overlap-region cuboids
SENSOR_SCORES = {0: 1.0, 1: 0.95, -1: 0.7}
NMS_IOU_THRESH = 0.2

CAMERAS = ["front_camera", "front_left_camera", "front_right_camera",
           "left_camera", "right_camera", "back_camera"]


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _read_pickle(path):
    # pandaset pkl / pkl.gz both handled; pd.read_pickle infers compression
    try:
        return pd.read_pickle(path)
    except Exception:  # noqa: BLE001
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


def _ts_ms(ts_seconds):
    return int(round(ts_seconds * 1000.0))


def convert_sequence(seq_path, out_sweep_dir, write_cameras=True):
    """Convert raw sequence dir (e.g. .../pandaset/001) -> sweep dir of npz.

    Returns dict with stats. Idempotent: existing complete frames are skipped.
    """
    seq_name = os.path.basename(seq_path.rstrip("/"))
    os.makedirs(out_sweep_dir, exist_ok=True)

    lidar_dir = os.path.join(seq_path, "lidar")
    meta_dir = os.path.join(seq_path, "meta")
    cub_dir = os.path.join(seq_path, "annotations", "cuboids")
    sem_dir = os.path.join(seq_path, "annotations", "semseg")
    cam_root = os.path.join(seq_path, "camera")

    poses_raw = _read_json(os.path.join(lidar_dir, "poses.json"))
    lidar_ts = _read_json(os.path.join(lidar_dir, "timestamps.json"))
    gps = _read_json(os.path.join(meta_dir, "gps.json"))

    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.pkl*")))
    cub_files = sorted(glob.glob(os.path.join(cub_dir, "*.pkl*")))
    sem_files = sorted(glob.glob(os.path.join(sem_dir, "*.pkl*")))

    n_frames = len(lidar_files)
    stats = {"seq": seq_name, "frames": n_frames, "warnings": []}
    if not (len(poses_raw) == len(lidar_ts) == n_frames):
        stats["warnings"].append(
            f"count mismatch: lidar={n_frames} poses={len(poses_raw)} ts={len(lidar_ts)}")

    # semseg classes mapping (sequence-level)
    classes_path = os.path.join(sem_dir, "classes.json")
    classes = _read_json(classes_path) if os.path.exists(classes_path) else {}

    # camera meta
    cam_meta = {}
    if write_cameras and os.path.isdir(cam_root):
        for cam in sorted(os.listdir(cam_root)):
            cdir = os.path.join(cam_root, cam)
            if not os.path.isdir(cdir):
                continue
            try:
                intr = _read_json(os.path.join(cdir, "intrinsics.json"))
                cposes = _read_json(os.path.join(cdir, "poses.json"))
                cts = _read_json(os.path.join(cdir, "timestamps.json"))
                imgs = sorted(glob.glob(os.path.join(cdir, "*.jpg")))
                cam_meta[cam] = {"intr": intr, "poses": cposes, "ts": cts, "imgs": imgs}
            except OSError as e:
                stats["warnings"].append(f"camera {cam}: {e}")

    T_ego_global_list = [G.ego2global(G.pose_to_mat(p)) for p in poses_raw]

    meta = {
        "sequence": seq_name,
        "n_lidar_frames": n_frames,
        "ego_frame": "X forward, Y left, Z up",
        "axis_map_from_pandaset_vehicle": G.AXIS_MAP.tolist(),
        "boxes_convention": "x y z dx(length) dy(width) dz(height) yaw(rad CCW from ego +X)",
        "semseg_classes": classes,
        "cameras": sorted(cam_meta.keys()),
    }
    with open(os.path.join(out_sweep_dir, "sweep_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    for i in range(n_frames):
        frame_stem = os.path.basename(lidar_files[i]).split(".")[0]
        ts = _ts_ms(lidar_ts[i]) if i < len(lidar_ts) else int(frame_stem)
        T_world_veh = G.pose_to_mat(poses_raw[i])
        T_ego_global = T_ego_global_list[i]
        e2g_t = T_ego_global[:3, 3].astype(np.float32)
        e2g_q = G.mat_to_quat(T_ego_global[:3, :3]).astype(np.float32)

        # ---- lidar -------------------------------------------------------
        lidar_out = os.path.join(out_sweep_dir, f"lidar_{ts}.npz")
        T_veh_world = G.invert_T(T_world_veh)
        if not os.path.exists(lidar_out):
            df = _read_pickle(lidar_files[i])
            pts_world = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
            pts_ego = fastops.world_points_to_ego(
                pts_world, T_veh_world[:3, :3], T_veh_world[:3, 3], G.AXIS_MAP)
            intensity = df["i"].to_numpy(dtype=np.float64).clip(0, 255).astype(np.uint8) \
                if "i" in df else np.zeros(len(df), np.uint8)
            sensor_id = df["d"].to_numpy(dtype=np.uint8) if "d" in df \
                else np.zeros(len(df), np.uint8)
            if "t" in df and i < len(lidar_ts):
                rel_time = (df["t"].to_numpy(dtype=np.float64) - lidar_ts[i]).astype(np.float32)
            else:
                rel_time = np.zeros(len(df), np.float32)
            np.savez_compressed(
                lidar_out,
                points=pts_ego.astype(np.float16),
                intensity=intensity,
                sensor_id=sensor_id,
                rel_time=rel_time,
                ego2global_translation=e2g_t,
                ego2global_rotation=e2g_q,
            )
            n_pts = len(df)
        else:
            n_pts = -1

        # ---- cuboids ------------------------------------------------------
        boxes_out = os.path.join(out_sweep_dir, f"boxes_{ts}.npz")
        if not os.path.exists(boxes_out):
            if i < len(cub_files):
                cdf = _read_pickle(cub_files[i])
            else:
                cdf = None
            if cdf is not None and len(cdf) > 0:
                pos_w = cdf[["position.x", "position.y", "position.z"]].to_numpy(np.float64)
                dims = cdf[["dimensions.x", "dimensions.y", "dimensions.z"]].to_numpy(np.float64)
                yaws = cdf["yaw"].to_numpy(np.float64)
                T_world_ego = T_ego_global
                ego_x_w = T_world_ego[:3, 0]
                ego_x_ang = float(np.arctan2(ego_x_w[1], ego_x_w[0]))
                boxes = fastops.pandaset_boxes_to_ego(
                    pos_w, dims, yaws, T_veh_world[:3, :3], T_veh_world[:3, 3],
                    G.AXIS_MAP, ego_x_ang).astype(np.float32)
                labels = cdf["label"].astype(str).to_numpy()
                uuids = cdf["uuid"].astype(str).to_numpy()
                stationary = cdf["stationary"].astype(bool).to_numpy() \
                    if "stationary" in cdf else np.zeros(len(cdf), bool)
                motion = cdf["attributes.object_motion"].fillna("").astype(str).to_numpy() \
                    if "attributes.object_motion" in cdf else np.full(len(cdf), "", object)
                sid = cdf["cuboids.sensor_id"].fillna(-1).astype(np.int8).to_numpy() \
                    if "cuboids.sensor_id" in cdf else np.full(len(cdf), -1, np.int8)
                sib = cdf["cuboids.sibling_id"].fillna("").astype(str).to_numpy() \
                    if "cuboids.sibling_id" in cdf else np.full(len(cdf), "", object)
                # BEV NMS keep-mask (duplicate cuboids in lidar overlap region)
                scores = np.array([SENSOR_SCORES.get(int(s), 0.5) for s in sid],
                                  dtype=np.float64)
                bev = boxes[:, [0, 1, 3, 4, 6]].astype(np.float64)
                nms_keep = fastops.bev_nms(bev, scores, NMS_IOU_THRESH)
            else:
                boxes = np.zeros((0, 7), np.float32)
                labels = np.array([], dtype=str)
                uuids = np.array([], dtype=str)
                stationary = np.zeros(0, bool)
                motion = np.array([], dtype=object)
                sid = np.zeros(0, np.int8)
                sib = np.array([], dtype=object)
                nms_keep = np.zeros(0, bool)
            np.savez_compressed(
                boxes_out,
                boxes=boxes,
                class_names=labels,
                uuids=uuids,
                stationary=stationary,
                object_motion=motion,
                sensor_id=sid,
                sibling_id=sib,
                nms_keep=nms_keep,
                ego2global_translation=e2g_t,
                ego2global_rotation=e2g_q,
            )

        # ---- semseg --------------------------------------------------------
        sem_out = os.path.join(out_sweep_dir, f"semseg_{ts}.npz")
        if not os.path.exists(sem_out) and i < len(sem_files):
            try:
                sdf = _read_pickle(sem_files[i])
                cls = sdf["class"].astype(np.int32).to_numpy()
                lidar_len = n_pts if n_pts >= 0 else len(np.load(lidar_out)["points"])
                if len(cls) != lidar_len:
                    stats["warnings"].append(
                        f"frame {i}: semseg len {len(cls)} != lidar len {lidar_len}")
                    n = min(len(cls), lidar_len)
                    cls_out = np.zeros(lidar_len, np.uint8)
                    cls_out[:n] = cls[:n].clip(0, 255)
                else:
                    cls_out = cls.clip(0, 255).astype(np.uint8)
                np.savez_compressed(sem_out, classes=cls_out)
            except Exception as e:  # noqa: BLE001
                stats["warnings"].append(f"frame {i}: semseg failed: {e}")

        # ---- gps ------------------------------------------------------------
        gps_out = os.path.join(out_sweep_dir, f"gps_{ts}.npz")
        if not os.path.exists(gps_out) and i < len(gps):
            g = gps[i]
            xv = float(g.get("xvel", 0.0))
            yv = float(g.get("yvel", 0.0))
            np.savez_compressed(
                gps_out,
                lat=float(g.get("lat", 0.0)),
                long=float(g.get("long", 0.0)),
                height=float(g.get("height", 0.0)),
                xvel=xv, yvel=yv,
                speed=float(np.hypot(xv, yv)),
            )

        # ---- cameras ---------------------------------------------------------
        for cam, cm in cam_meta.items():
            if i >= len(cm["imgs"]) or i >= len(cm["poses"]):
                continue
            c_ts = _ts_ms(cm["ts"][i]) if i < len(cm["ts"]) else ts
            img_out = os.path.join(out_sweep_dir, f"{cam}_{c_ts}.jpg")
            npz_out = os.path.join(out_sweep_dir, f"{cam}_{c_ts}.npz")
            if not os.path.exists(img_out):
                shutil.copyfile(cm["imgs"][i], img_out)
            if not os.path.exists(npz_out):
                K = np.array([
                    [cm["intr"]["fx"], 0.0, cm["intr"]["cx"]],
                    [0.0, cm["intr"]["fy"], cm["intr"]["cy"]],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float32)
                T_world_cam = G.pose_to_mat(cm["poses"][i])
                # ego_from_cam = ego_from_veh @ veh_from_world @ world_from_cam
                T_ego_cam = G.AXIS_MAP_4 @ G.invert_T(T_world_veh) @ T_world_cam
                s2e_t = T_ego_cam[:3, 3].astype(np.float32)
                s2e_q = G.mat_to_quat(T_ego_cam[:3, :3]).astype(np.float32)
                np.savez_compressed(
                    npz_out,
                    intrinsic=K,
                    sensor2ego_translation=s2e_t,
                    sensor2ego_rotation=s2e_q,
                    ego2global_translation=e2g_t,
                    ego2global_rotation=e2g_q,
                )

    # mark completion
    with open(os.path.join(out_sweep_dir, ".converted"), "w") as f:
        json.dump({"frames": n_frames}, f)
    return stats


def is_converted(out_sweep_dir):
    return os.path.exists(os.path.join(out_sweep_dir, ".converted"))
