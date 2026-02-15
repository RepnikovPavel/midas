import os
import json
import glob
import numpy as np
import pandas as pd
import cv2
import transforms3d as t3d
from PIL import Image
from tqdm import tqdm
from numba import njit
from numba.typed import List as NumbaList

from vispy import app, scene
from vispy.scene import visuals
from vispy.color import Colormap

from visualizer import LidarVisualizer
from nmsjit import nms_bev_jit

T_AXIS = np.array([
    [ 0,  1,  0],
    [-1,  0,  0],
    [ 0,  0,  1]
], dtype=np.float64)

T_AXIS_INV = T_AXIS.T

CALIBRATIONS_DATA = {
    'main_pandar64': { 
        'extrinsic': {
            'rotation': {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            'translation': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
    },
    'front_gt': { 
        'extrinsic': {
            'rotation': {'w': 0.021475754959146356, 'x': -0.002060907279494794, 'y': 0.01134678181520767, 'z': 0.9997028534282365},
            'translation': {'x': -0.000451117754, 'y': -0.605646431446, 'z': -0.301525235176}
        }
    },
    'back_camera': { 'intrinsic': {"fx": 933.4667, "fy": 934.6754, "cx": 896.4692, "cy": 507.3557} },
    'front_camera': { 'intrinsic': {"fx": 1970.0131, "fy": 1970.0091, "cx": 970.0002, "cy": 483.2988} },
    'front_left_camera': { 'intrinsic': {"fx": 929.8429, "fy": 930.0592, "cx": 972.1794, "cy": 508.0057} },
    'front_right_camera': { 'intrinsic': {"fx": 930.0407, "fy": 930.0324, "cx": 965.0525, "cy": 463.4161} },
    'left_camera': { 'intrinsic': {"fx": 930.4514, "fy": 930.0891, "cx": 991.6883, "cy": 541.6057} },
    'right_camera': { 'intrinsic': {"fx": 922.5465, "fy": 922.4229, "cx": 945.057, "cy": 517.575} }
}

def _heading_position_to_mat(heading, position):
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    return t3d.affines.compose(pos, t3d.quaternions.quat2mat(quat), [1.0, 1.0, 1.0])

def load_pandaset_data(root):
    lidar_dir = os.path.join(root, 'lidar')
    camera_root = os.path.join(root, 'camera')
    meta_dir = os.path.join(root, 'meta')
    cuboids_dir = os.path.join(root, 'annotations', 'cuboids')
    
    with open(os.path.join(lidar_dir, 'poses.json'), 'r') as f:
        poses_dicts = json.load(f)
    poses = [_heading_position_to_mat(p['heading'], p['position']) for p in poses_dicts]
    
    gps_path = os.path.join(meta_dir, 'gps.json')
    if os.path.exists(gps_path):
        with open(gps_path, 'r') as f:
            gps_data = json.load(f)
    else:
        gps_data = None

    cuboid_files = sorted(glob.glob(f'{cuboids_dir}/*.pkl'))
    cuboids = []
    for fp in tqdm(cuboid_files, desc="Loading Cuboids"):
        df = pd.read_pickle(fp)
        if df is None or df.empty:
            cuboids.append((np.zeros((0, 7)), np.zeros(0, dtype=object), np.zeros(0), np.zeros(0, dtype=object)))
            continue
        boxes = np.stack([
            df['position.x'].values, df['position.y'].values, df['position.z'].values,
            df['dimensions.x'].values, df['dimensions.y'].values, df['dimensions.z'].values,
            df['yaw'].values
        ], axis=1)
        labels = df['label'].values
        uuids = df['uuid'].values
        sensor_ids = df['cuboids.sensor_id'].fillna('unknown').values
        cuboids.append((boxes, labels, uuids, sensor_ids))

    pkl_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.pkl')])
    points = []
    cameras_data = {}
    cam_names = [d for d in os.listdir(camera_root) if os.path.isdir(os.path.join(camera_root, d))] if os.path.exists(camera_root) else []
    
    for cam in cam_names:
        cameras_data[cam] = {'images': [], 'intr': CALIBRATIONS_DATA.get(cam, {}).get('intrinsic')}
        intr_path = os.path.join(camera_root, cam, 'intrinsics.json')
        if os.path.exists(intr_path):
            with open(intr_path, 'r') as f:
                cameras_data[cam]['intr'] = json.load(f)
        with open(os.path.join(camera_root, cam, 'poses.json'), 'r') as f:
            cameras_data[cam]['poses'] = json.load(f)

    for pkl_file in tqdm(pkl_files, desc="Loading LiDAR & Images"):
        lidar_path = os.path.join(lidar_dir, pkl_file)
        try:
            df = pd.read_pickle(lidar_path)
            pts = df[['x', 'y', 'z']].values
        except:
            pts = np.zeros((0, 3))
        points.append(pts)
        
        frame_idx_str = pkl_file.replace('.pkl', '')
        for cam in cam_names:
            img_path = os.path.join(camera_root, cam, f"{frame_idx_str}.jpg")
            if os.path.exists(img_path):
                try:
                    img = np.array(Image.open(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cameras_data[cam]['images'].append(img)
                except:
                    cameras_data[cam]['images'].append(None)
            else:
                cameras_data[cam]['images'].append(None)

    return {
        'poses': poses,
        'cuboids': cuboids,
        'points': points,
        'cameras': cameras_data,
        'gps': gps_data
    }

def process_frame_data(
    pts_world,
    boxes_world,
    labels_world,
    uuids_world,
    sensors_world,
    lidar_pose,
    iou_thresh=0.2
):
    T_world_ego = lidar_pose
    T_ego_world = np.linalg.inv(T_world_ego)

    if pts_world.shape[0] == 0:
        pts_ego = pts_world
    else:
        pts_ego_raw = (T_ego_world[:3, :3] @ pts_world.T + T_ego_world[:3, 3:4]).T
        pts_ego = (T_AXIS @ pts_ego_raw.T).T

    if boxes_world.shape[0] == 0:
        return pts_ego, np.zeros((0, 7)), np.zeros(0, dtype=object)

    pos_world = boxes_world[:, :3]
    pos_ego_raw = (T_ego_world[:3, :3] @ pos_world.T + T_ego_world[:3, 3:4]).T
    pos_ego = (T_AXIS @ pos_ego_raw.T).T

    forward = T_world_ego[:3, 1]
    yaw_vehicle = np.arctan2(forward[1], forward[0])

    yaws_ego = boxes_world[:, 6] - yaw_vehicle + np.pi / 2.0

    boxes_ego = boxes_world.copy()
    boxes_ego[:, :3] = pos_ego
    boxes_ego[:, 6] = yaws_ego

    boxes_ego[:, 3] = boxes_world[:, 4]
    boxes_ego[:, 4] = boxes_world[:, 3]

    sensor_scores = {
        'main_pandar64': 1.0,
        'front_gt': 0.95,
        'unknown': 0.5
    }
    for cam in [
        'front_camera', 'front_left_camera', 'front_right_camera',
        'left_camera', 'right_camera', 'back_camera'
    ]:
        sensor_scores[cam] = 0.7

    scores = np.array([sensor_scores.get(s, 0.5) for s in sensors_world])
    boxes_bev = boxes_ego[:, [0, 1, 3, 4, 6]]

    keep_mask = nms_bev_jit(boxes_bev, scores.copy(), iou_thresh)

    return pts_ego, boxes_ego[keep_mask], labels_world[keep_mask]

def draw_images_grid(cameras_data, pts_world, boxes_world, labels_world, frame_idx, lidar_pose):
    display_w, display_h = 640, 360
    cam_names = ['front_left_camera', 'front_camera', 'front_right_camera',
                 'left_camera', 'back_camera', 'right_camera']
    
    row1_imgs = []
    row2_imgs = []
    
    for i, cam_name in enumerate(cam_names):
        if cam_name not in cameras_data:
            img_display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        else:
            cam_block = cameras_data[cam_name]
            pil_img = cam_block['images'][frame_idx]
            
            if pil_img is None:
                 img_display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
            else:
                img = pil_img
                
                cam_pose_data = cam_block['poses'][frame_idx]
                T_world_cam = _heading_position_to_mat(cam_pose_data['heading'], cam_pose_data['position'])
                T_cam_world = np.linalg.inv(T_world_cam)
                
                points_cam = (T_cam_world[:3, :3] @ pts_world.T + T_cam_world[:3, 3:]).T
                mask = points_cam[:, 2] > 0.5
                points_cam = points_cam[mask]
                
                intr = cam_block['intr']
                K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])
                
                uv = (K @ points_cam.T).T
                uv[:, 0] /= uv[:, 2]
                uv[:, 1] /= uv[:, 2]
                
                depths = points_cam[:, 2]
                max_depth = 80.0
                for j in range(len(uv)):
                    u, v = int(uv[j, 0]), int(uv[j, 1])
                    d = depths[j]
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        color = (int(255 * (1 - min(d/max_depth, 1.0))), 0, int(255 * min(d/max_depth, 1.0)))
                        cv2.circle(img, (u, v), 1, color, -1)
                
                if boxes_world is not None and boxes_world.shape[0] > 0:
                    corners_unit = np.array([
                        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
                    ])
                    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
                    
                    for k in range(boxes_world.shape[0]):
                        pos = boxes_world[k, :3]
                        dim = boxes_world[k, 3:6]
                        yaw = boxes_world[k, 6]
                        
                        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                        corners_world = (R_yaw @ (dim * corners_unit).T).T + pos
                        corners_cam = (T_cam_world[:3, :3] @ corners_world.T + T_cam_world[:3, 3:]).T
                        
                        if corners_cam[:, 2].max() < 0.1: continue
                        
                        uv_box = (K @ corners_cam.T).T
                        uv_box[:, 0] /= uv_box[:, 2]
                        uv_box[:, 1] /= uv_box[:, 2]
                        uv_box = uv_box[:, :2].astype(int)
                        
                        for start, end in lines:
                            cv2.line(img, tuple(uv_box[start]), tuple(uv_box[end]), (0, 255, 0), 2)
                        
                        top_center_world = pos + np.array([0, 0, dim[2]/2])
                        top_center_cam = T_cam_world[:3, :3] @ top_center_world + T_cam_world[:3, 3]
                        if top_center_cam[2] > 0.1:
                            uv_label = K @ top_center_cam
                            u_l, v_l = int(uv_label[0]/uv_label[2]), int(uv_label[1]/uv_label[2])
                            cv2.putText(img, labels_world[k], (u_l - 20, v_l - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                img_display = cv2.resize(img, (display_w, display_h))
        
        if i < 3:
            row1_imgs.append(img_display)
        else:
            row2_imgs.append(img_display)
            
    top_row = np.hstack(row1_imgs)
    bottom_row = np.hstack(row2_imgs)
    grid = np.vstack([top_row, bottom_row])
    return grid

if __name__ == "__main__":
    DATA_ROOT = '/mnt/nvme/datasets/pandaset'

    print("Loading sequence 001...")
    data = load_pandaset_data(os.path.join(DATA_ROOT, '001'))
    
    vis = LidarVisualizer(title="PandaSet 3D LiDAR")
    cv2.namedWindow("PandaSet View", cv2.WINDOW_NORMAL)
    
    current_frame_idx = 0
    
    def update_loop(event):
        global current_frame_idx
        
        if current_frame_idx >= len(data['points']):
            current_frame_idx = 0 

        pts_world = data['points'][current_frame_idx]
        boxes_all, labels_all, uuids_all, sensors_all = data['cuboids'][current_frame_idx]
        lidar_pose = data['poses'][current_frame_idx]
        
        pts_ego, boxes_ego, labels_ego = process_frame_data(
            pts_world, boxes_all, labels_all, uuids_all, sensors_all, lidar_pose
        )
        
        vis.update(pts_ego, boxes_ego, labels_ego)
        
        # ---- GPS speed ----
        gps_data = data.get('gps')
        if gps_data is not None and current_frame_idx < len(gps_data):
            gps_frame = gps_data[current_frame_idx]
            xvel = gps_frame.get('xvel', 0.0)
            yvel = gps_frame.get('yvel', 0.0)
            v_world = np.array([xvel, yvel, 0.0], dtype=np.float64)
            v_world_abs = np.linalg.norm(v_world)
            print(f"Frame {current_frame_idx:03d}: speed = {v_world_abs:6.2f} m/s")
        # --------------------
        
        T_world_ego = lidar_pose
        R_world_ego = T_world_ego[:3, :3]
        t_world_ego = T_world_ego[:3, 3]
        
        boxes_world_for_draw = np.zeros((0, 7))
        
        if boxes_ego.shape[0] > 0:
            boxes_world_for_draw = boxes_ego.copy()

            pos_ego_new = boxes_ego[:, :3]
            pos_ego_old = (T_AXIS_INV @ pos_ego_new.T).T
            pos_world_draw = (R_world_ego @ pos_ego_old.T + t_world_ego.reshape(3, 1)).T
            boxes_world_for_draw[:, :3] = pos_world_draw

            forward = T_world_ego[:3, 1]
            yaw_vehicle = np.arctan2(forward[1], forward[0])
            boxes_world_for_draw[:, 6] = boxes_ego[:, 6] + yaw_vehicle - np.pi / 2.0

            lengths = boxes_ego[:, 3].copy()
            boxes_world_for_draw[:, 3] = boxes_ego[:, 4]
            boxes_world_for_draw[:, 4] = lengths

        img_grid = draw_images_grid(
            data['cameras'], pts_world, boxes_world_for_draw, labels_ego, current_frame_idx, lidar_pose
        )
        cv2.imshow("PandaSet View", img_grid)
        
        key = cv2.waitKey(1)
        if key == 27:
            vis.close()
            app.quit()
            cv2.destroyAllWindows()
            return

        current_frame_idx += 1

    timer = app.Timer(interval=0.1, connect=update_loop, start=True)
    
    print("Starting visualization loop...")
    app.run()