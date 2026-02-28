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

# Матрица перехода из системы PandaSet Vehicle (Y-вперед) в Vispy Ego (X-вперед)
# P_vispy = T_AXIS @ P_vehicle
T_AXIS = np.array([
    [ 0,  1,  0],
    [-1,  0,  0],
    [ 0,  0,  1]
], dtype=np.float64)

# Обратная матрица: из Vispy Ego (X-вперед) в PandaSet Vehicle (Y-вперед)
# P_vehicle = T_AXIS_INV @ P_vispy
T_AXIS_INV = T_AXIS.T

# 4x4 версия для аффинных преобразований
T_AXIS_4x4 = np.eye(4, dtype=np.float64)
T_AXIS_4x4[:3, :3] = T_AXIS

T_AXIS_INV_4x4 = np.eye(4, dtype=np.float64)
T_AXIS_INV_4x4[:3, :3] = T_AXIS_INV

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
    'back_camera': { 
        'intrinsic': {"fx": 933.4667, "fy": 934.6754, "cx": 896.4692, "cy": 507.3557},
        'extrinsic': {
            'rotation': {'w': 0.713789231075861, 'x': 0.7003585531940812, 'y': -0.001595758695393934, 'z': -0.0005330311533742299},
            'translation': {'x': -0.0004217634029916384, 'y': -0.21683144949675118, 'z': -1.0553445472201475}
        }
    },
    'front_camera': { 
        'intrinsic': {"fx": 1970.0131, "fy": 1970.0091, "cx": 970.0002, "cy": 483.2988},
        'extrinsic': {
            'rotation': {'w': 0.016213200031258722, 'x': 0.0030578899383849464, 'y': 0.7114721800418571, 'z': -0.7025205466606356},
            'translation': {'x': 0.0002585796504896516, 'y': -0.03907777167811011, 'z': -0.0440125762408362}
        }
    },
    'front_left_camera': { 
        'intrinsic': {"fx": 929.8429, "fy": 930.0592, "cx": 972.1794, "cy": 508.0057},
        'extrinsic': {
            'rotation': {'w': 0.33540022607039827, 'x': 0.3277491469609924, 'y': -0.6283486651480494, 'z': 0.6206973014480826},
            'translation': {'x': -0.25842240863267835, 'y': -0.3070654284505582, 'z': -0.9244245686318884}
        }
    },
    'front_right_camera': { 
        'intrinsic': {"fx": 930.0407, "fy": 930.0324, "cx": 965.0525, "cy": 463.4161},
        'extrinsic': {
            'rotation': {'w': 0.3537633879725252, 'x': 0.34931795852655334, 'y': 0.6120314641083645, 'z': -0.6150170047424814},
            'translation': {'x': 0.2546935700219631, 'y': -0.24929449717803095, 'z': -0.8686597280810242}
        }
    },
    'left_camera': { 
        'intrinsic': {"fx": 930.4514, "fy": 930.0891, "cx": 991.6883, "cy": 541.6057},
        'extrinsic': {
            'rotation': {'w': 0.5050391917998245, 'x': 0.49253073152800625, 'y': -0.4989265501075421, 'z': 0.503409565706149},
            'translation': {'x': 0.23864835336611942, 'y': -0.2801448284013492, 'z': -0.5376795959387791}
        }
    },
    'right_camera': { 
        'intrinsic': {"fx": 922.5465, "fy": 922.4229, "cx": 945.057, "cy": 517.575},
        'extrinsic': {
            'rotation': {'w': 0.5087448402081216, 'x': 0.4947520981649951, 'y': 0.4977829953071897, 'z': -0.49860920419297333},
            'translation': {'x': -0.23097163411257893, 'y': -0.30843497058841024, 'z': -0.6850441215571058}
        }
    }
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
    
    # Load LiDAR Poses (Это позы Vehicle в World)
    with open(os.path.join(lidar_dir, 'poses.json'), 'r') as f:
        poses_dicts = json.load(f)
    poses = [_heading_position_to_mat(p['heading'], p['position']) for p in poses_dicts]
    
    # Load LiDAR Timestamps
    lidar_ts_path = os.path.join(lidar_dir, 'timestamps.json')
    if os.path.exists(lidar_ts_path):
        with open(lidar_ts_path, 'r') as f:
            lidar_timestamps_raw = json.load(f)
            lidar_timestamps = [int(ts * 1000) for ts in lidar_timestamps_raw]
    else:
        lidar_timestamps = []

    # Load Meta Timestamps
    meta_ts_path = os.path.join(meta_dir, 'timestamps.json')
    if os.path.exists(meta_ts_path):
        with open(meta_ts_path, 'r') as f:
            meta_timestamps_raw = json.load(f)
            meta_timestamps = [int(ts * 1000) for ts in meta_timestamps_raw]
    else:
        meta_timestamps = []

    # Load GPS
    gps_path = os.path.join(meta_dir, 'gps.json')
    if os.path.exists(gps_path):
        with open(gps_path, 'r') as f:
            gps_data = json.load(f)
    else:
        gps_data = None

    # Load Cuboids and Transform to Ego-Vispy immediately
    cuboid_files = sorted(glob.glob(f'{cuboids_dir}/*.pkl'))
    cuboids = []
    
    print("Loading and transforming Cuboids to Ego-Vispy frame...")
    for i, fp in tqdm(enumerate(cuboid_files), total=len(cuboid_files)):
        df = pd.read_pickle(fp)
        if df is None or df.empty:
            cuboids.append((np.zeros((0, 7)), np.zeros(0, dtype=object), np.zeros(0), np.zeros(0, dtype=object)))
            continue
        
        # Raw boxes in World Frame
        boxes_world = np.stack([
            df['position.x'].values, df['position.y'].values, df['position.z'].values,
            df['dimensions.x'].values, df['dimensions.y'].values, df['dimensions.z'].values,
            df['yaw'].values
        ], axis=1)
        labels = df['label'].values
        uuids = df['uuid'].values
        sensor_ids = df['cuboids.sensor_id'].fillna('unknown').values

        # Transform to Ego-Vispy Frame
        # 1. World -> Vehicle
        T_world_veh = poses[i]
        T_veh_world = np.linalg.inv(T_world_veh)

        pos_world = boxes_world[:, :3]
        pos_veh = (T_veh_world[:3, :3] @ pos_world.T + T_veh_world[:3, 3:4]).T
        
        # 2. Vehicle -> Vispy Ego
        pos_ego = (T_AXIS @ pos_veh.T).T

        # Transform yaw
        # Yaw in World -> Yaw in Vehicle -> Yaw in Vispy
        
        # 1. Угол машины (Vehicle Forward) в мировой системе.
        # Ось Y машины - это "вперед". Берем её из матрицы позы.
        forward_world = T_world_veh[:3, 1] 
        yaw_vehicle = np.arctan2(forward_world[1], forward_world[0])
        
        # 2. Угол бокса относительно машины.
        # boxes_world[:, 6] - это yaw бокса в мировой системе (угол оси X бокса).
        # yaw_veh_rel = (угол бокса) - (угол машины).
        # Если бокс смотрит туда же, куда и машина (вдоль Y), то его yaw в мировой системе
        # должен быть повернут на 90 град относительно X машины.
        # Но обычно берется просто разность.
        yaw_veh_rel = boxes_world[:, 6] - yaw_vehicle
        
        # 3. Приведение к Vispy.
        # В PandaSet Yaw=0 -> Ось X (Вправо). "Вперед" - это Y (Yaw=90).
        # В Vispy Yaw=0 -> Ось X (Вперед).
        # Значит, Vispy Yaw = PandaSet Yaw + 90 град.
        # Но yaw_veh_rel уже содержит разницу. 
        # Если бокс направлен вдоль Y (Forward машины), yaw_veh_rel = 0 (если в PS yaw=90 при align c Y).
        # Нет, в PS: yaw измеряется от X. Если бокс смотрит по Y, его yaw = pi/2.
        # yaw_veh_rel = (pi/2) - yaw_vehicle (если машина смотрит по X world).
        # Это запутанно. Проще так:
        # Heading бокса = yaw_world + pi/2 (т.к. Forward это Y).
        # Heading машины = yaw_vehicle.
        # Relative Heading = (yaw_world + pi/2) - yaw_vehicle.
        # В Vispy Yaw = Relative Heading.
        
        yaws_ego = (boxes_world[:, 6] + np.pi / 2.0) - yaw_vehicle

        boxes_ego = boxes_world.copy()
        boxes_ego[:, :3] = pos_ego
        boxes_ego[:, 6] = yaws_ego
        
        # Swap dimensions (length/width) because axes swap
        # Vehicle: X-right, Y-forward. Vispy: X-forward, Y-left.
        # Dim X (width) in Vehicle becomes Dim Y in Vispy.
        # Dim Y (length) in Vehicle becomes Dim X in Vispy.
        boxes_ego[:, 3] = boxes_world[:, 4] # Vispy X = Length = Vehicle Y
        boxes_ego[:, 4] = boxes_world[:, 3] # Vispy Y = Width = Vehicle X

        cuboids.append((boxes_ego, labels, uuids, sensor_ids))

    # Load Points and Camera Data
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
            
        cam_ts_path = os.path.join(camera_root, cam, 'timestamps.json')
        if os.path.exists(cam_ts_path):
            with open(cam_ts_path, 'r') as f:
                cam_ts_raw = json.load(f)
                cameras_data[cam]['timestamps'] = [int(ts * 1000) for ts in cam_ts_raw]
        else:
            cameras_data[cam]['timestamps'] = []

    print("Loading and transforming LiDAR to Ego-Vispy frame...")
    for i, pkl_file in tqdm(enumerate(pkl_files), total=len(pkl_files)):
        lidar_path = os.path.join(lidar_dir, pkl_file)
        try:
            df = pd.read_pickle(lidar_path)
            pts_world = df[['x', 'y', 'z']].values
        except:
            pts_world = np.zeros((0, 3))
        
        # Transform Points to Ego-Vispy Frame immediately
        if pts_world.shape[0] > 0:
            T_world_veh = poses[i]
            T_veh_world = np.linalg.inv(T_world_veh)
            
            # 1. World -> Vehicle
            pts_veh = (T_veh_world[:3, :3] @ pts_world.T + T_veh_world[:3, 3:4]).T
            
            # 2. Vehicle -> Vispy Ego
            pts_ego = (T_AXIS @ pts_veh.T).T
            points.append(pts_ego)
        else:
            points.append(pts_world)
        
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
        'gps': gps_data,
        'lidar_timestamps': lidar_timestamps,
        'meta_timestamps': meta_timestamps
    }

def process_frame_data(
    pts_ego,
    boxes_ego,
    labels,
    uuids,
    sensors,
    iou_thresh=0.2
):
    # Data is already in Ego-Vispy frame. We only perform NMS here.
    if boxes_ego.shape[0] == 0:
        return pts_ego, boxes_ego, labels

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

    scores = np.array([sensor_scores.get(s, 0.5) for s in sensors])
    boxes_bev = boxes_ego[:, [0, 1, 3, 4, 6]]

    keep_mask = nms_bev_jit(boxes_bev, scores.copy(), iou_thresh)

    return pts_ego, boxes_ego[keep_mask], labels[keep_mask]

def draw_images_grid(cameras_data, pts_ego, boxes_ego, labels_ego, frame_idx, T_world_vehicle):
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
                
                # Calculate T_cam_ego (Ego-Vispy -> Camera)
                # Chain: Cam <- World <- Vehicle <- VispyEgo
                # T_cam_ego = T_cam_world @ T_world_vehicle @ T_vispy_to_vehicle
                
                cam_pose_data = cam_block['poses'][frame_idx]
                T_world_cam = _heading_position_to_mat(cam_pose_data['heading'], cam_pose_data['position'])
                T_cam_world = np.linalg.inv(T_world_cam)
                
                # T_vispy_to_vehicle is T_AXIS_INV_4x4
                T_cam_ego = T_cam_world @ T_world_vehicle @ T_AXIS_INV_4x4
                
                # Project Points (Ego-VisPy -> Cam)
                points_cam = (T_cam_ego[:3, :3] @ pts_ego.T + T_cam_ego[:3, 3:4]).T
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
                
                # Project Boxes (Ego-VisPy -> Cam)
                if boxes_ego is not None and boxes_ego.shape[0] > 0:
                    corners_unit = np.array([
                        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
                    ])
                    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
                    
                    for k in range(boxes_ego.shape[0]):
                        pos = boxes_ego[k, :3]
                        dim = boxes_ego[k, 3:6]
                        yaw = boxes_ego[k, 6]
                        
                        # Corners in Ego-Vispy frame
                        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                        corners_ego = (R_yaw @ (dim * corners_unit).T).T + pos
                        
                        # Transform to Cam
                        corners_cam = (T_cam_ego[:3, :3] @ corners_ego.T + T_cam_ego[:3, 3:4]).T
                        
                        if corners_cam[:, 2].max() < 0.1: continue
                        
                        uv_box = (K @ corners_cam.T).T
                        uv_box[:, 0] /= uv_box[:, 2]
                        uv_box[:, 1] /= uv_box[:, 2]
                        uv_box = uv_box[:, :2].astype(int)
                        
                        for start, end in lines:
                            cv2.line(img, tuple(uv_box[start]), tuple(uv_box[end]), (0, 255, 0), 2)
                        
                        # Label projection
                        top_center_ego = pos + np.array([0, 0, dim[2]/2])
                        top_center_cam = T_cam_ego[:3, :3] @ top_center_ego + T_cam_ego[:3, 3]
                        if top_center_cam[2] > 0.1:
                            uv_label = K @ top_center_cam
                            u_l, v_l = int(uv_label[0]/uv_label[2]), int(uv_label[1]/uv_label[2])
                            cv2.putText(img, labels_ego[k], (u_l - 20, v_l - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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

        # Data is already in Ego-Vispy frame
        pts_ego = data['points'][current_frame_idx]
        boxes_ego_all, labels_all, uuids_all, sensors_all = data['cuboids'][current_frame_idx]
        
        # T_world_vehicle is needed to bridge the gap to cameras
        T_world_vehicle = data['poses'][current_frame_idx]
        
        # Only NMS is performed here
        pts_ego, boxes_ego, labels_ego = process_frame_data(
            pts_ego, boxes_ego_all, labels_all, uuids_all, sensors_all
        )
        
        vis.update(pts_ego, boxes_ego, labels_ego)
        
        # ---- Timing & GPS Info ----
        lidar_ts_ms = data['lidar_timestamps'][current_frame_idx] if current_frame_idx < len(data['lidar_timestamps']) else 0
        
        gps_data = data.get('gps')
        speed_val = 0.0
        if gps_data is not None and current_frame_idx < len(gps_data):
            gps_frame = gps_data[current_frame_idx]
            xvel = gps_frame.get('xvel', 0.0)
            yvel = gps_frame.get('yvel', 0.0)
            v_world = np.array([xvel, yvel, 0.0], dtype=np.float64)
            speed_val = np.linalg.norm(v_world)
            
        print(f"Frame {current_frame_idx:03d}: TS={lidar_ts_ms} ms | Speed = {speed_val:6.2f} m/s")
        # --------------------
        
        # Pass Ego-Vispy data and Vehicle Pose to drawing function
        img_grid = draw_images_grid(
            data['cameras'], pts_ego, boxes_ego, labels_ego, current_frame_idx, T_world_vehicle
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