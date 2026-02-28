import os
import json
import glob
import shutil
import numpy as np
import pandas as pd
import cv2
import transforms3d as t3d
from PIL import Image
from tqdm import tqdm
from nmsjit import nms_bev_jit  # Предполагаем, что у вас есть этот модуль

# --- Константы и Матрицы Преобразований ---

# Матрица перехода из системы PandaSet Vehicle (Y-вперед) в Vispy Ego (X-вперед)
T_AXIS = np.array([
    [ 0,  1,  0],
    [-1,  0,  0],
    [ 0,  0,  1]
], dtype=np.float64)

T_AXIS_INV = T_AXIS.T

T_AXIS_4x4 = np.eye(4, dtype=np.float64)
T_AXIS_4x4[:3, :3] = T_AXIS

T_AXIS_INV_4x4 = np.eye(4, dtype=np.float64)
T_AXIS_INV_4x4[:3, :3] = T_AXIS_INV

# Данные калибровок (как в исходном коде)
CALIBRATIONS_DATA = {
    # ... (вставьте сюда ваш словарь CALIBRATIONS_DATA целиком, он длинный, поэтому опущен для краткости, но он нужен)
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
    # ... остальные камеры ...
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

def fill_missing_calibs():
    # Заполняем недостающие интринсики из файлов, если нужно (здесь упрощенно берем из словаря)
    pass 

def _heading_position_to_mat(heading, position):
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    return t3d.affines.compose(pos, t3d.quaternions.quat2mat(quat), [1.0, 1.0, 1.0])

def get_sensor_scores():
    scores = {
        'main_pandar64': 1.0, 'front_gt': 0.95, 'unknown': 0.5
    }
    for cam in ['front_camera', 'front_left_camera', 'front_right_camera',
                'left_camera', 'right_camera', 'back_camera']:
        scores[cam] = 0.7
    return scores

def process_boxes_nms(boxes_ego, labels, uuids, sensors, iou_thresh=0.2):
    if boxes_ego.shape[0] == 0:
        return boxes_ego, labels, uuids

    sensor_scores = get_sensor_scores()
    scores = np.array([sensor_scores.get(s, 0.5) for s in sensors])
    boxes_bev = boxes_ego[:, [0, 1, 3, 4, 6]]
    keep_mask = nms_bev_jit(boxes_bev, scores.copy(), iou_thresh)
    return boxes_ego[keep_mask], labels[keep_mask], uuids[keep_mask]

def convert_sequence(seq_path, output_root):
    seq_name = os.path.basename(seq_path)
    sweep_dir = os.path.join(output_root, f"sweep_{seq_name}")
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"Processing sequence: {seq_name} -> {sweep_dir}")

    # Пути
    lidar_dir = os.path.join(seq_path, 'lidar')
    camera_root = os.path.join(seq_path, 'camera')
    meta_dir = os.path.join(seq_path, 'meta')
    cuboids_dir = os.path.join(seq_path, 'annotations', 'cuboids')

    # 1. Загрузка базовых метаданных
    with open(os.path.join(lidar_dir, 'poses.json'), 'r') as f:
        poses_dicts = json.load(f)
    poses = [_heading_position_to_mat(p['heading'], p['position']) for p in poses_dicts]

    with open(os.path.join(lidar_dir, 'timestamps.json'), 'r') as f:
        lidar_timestamps = [int(ts * 1000) for ts in json.load(f)]

    with open(os.path.join(meta_dir, 'gps.json'), 'r') as f:
        gps_data = json.load(f)

    # 2. Загрузка и преобразование Cuboids
    cuboid_files = sorted(glob.glob(f'{cuboids_dir}/*.pkl'))
    all_cuboids = []
    
    print("Loading Cuboids...")
    for i, fp in tqdm(enumerate(cuboid_files), total=len(cuboid_files)):
        df = pd.read_pickle(fp)
        if df is None or df.empty:
            all_cuboids.append(None)
            continue
        
        # Координаты World
        boxes_world = np.stack([
            df['position.x'].values, df['position.y'].values, df['position.z'].values,
            df['dimensions.x'].values, df['dimensions.y'].values, df['dimensions.z'].values,
            df['yaw'].values
        ], axis=1)
        labels = df['label'].values
        uuids = df['uuid'].values
        sensor_ids = df['cuboids.sensor_id'].fillna('unknown').values

        # Трансформация в Ego (Vispy)
        T_world_veh = poses[i]
        T_veh_world = np.linalg.inv(T_world_veh)

        pos_world = boxes_world[:, :3]
        pos_veh = (T_veh_world[:3, :3] @ pos_world.T + T_veh_world[:3, 3:4]).T
        pos_ego = (T_AXIS @ pos_veh.T).T

        forward_world = T_world_veh[:3, 1] 
        yaw_vehicle = np.arctan2(forward_world[1], forward_world[0])
        yaws_ego = (boxes_world[:, 6] + np.pi / 2.0) - yaw_vehicle

        boxes_ego = boxes_world.copy()
        boxes_ego[:, :3] = pos_ego
        boxes_ego[:, 6] = yaws_ego
        boxes_ego[:, 3] = boxes_world[:, 4] # Swap dx/dy if needed? (Vispy X-forward usually swaps dims)
        boxes_ego[:, 4] = boxes_world[:, 3]
        
        # NMS
        boxes_nms, labels_nms, uuids_nms = process_boxes_nms(boxes_ego, labels, uuids, sensor_ids)
        
        all_cuboids.append((boxes_nms, labels_nms, uuids_nms))

    # 3. Обработка кадров
    pkl_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.pkl')])
    cam_names = [d for d in os.listdir(camera_root) if os.path.isdir(os.path.join(camera_root, d))] if os.path.exists(camera_root) else []

    # Подготовка данных камер (чтение поз)
    cam_poses_raw = {}
    cam_ts = {}
    for cam in cam_names:
        with open(os.path.join(camera_root, cam, 'poses.json'), 'r') as f:
            cam_poses_raw[cam] = json.load(f)
        with open(os.path.join(camera_root, cam, 'timestamps.json'), 'r') as f:
            cam_ts[cam] = [int(ts * 1000) for ts in json.load(f)]

    print("Processing frames...")
    for i, pkl_file in tqdm(enumerate(pkl_files), total=len(pkl_files)):
        ts = lidar_timestamps[i]
        
        # --- LiDAR ---
        df = pd.read_pickle(os.path.join(lidar_dir, pkl_file))
        pts_world = df[['x', 'y', 'z']].values
        
        # Трансформация Points -> Ego
        T_world_veh = poses[i]
        T_veh_world = np.linalg.inv(T_world_veh)
        pts_veh = (T_veh_world[:3, :3] @ pts_world.T + T_veh_world[:3, 3:4]).T
        pts_ego = (T_AXIS @ pts_veh.T).T
        
        # Ego -> Global matrix для сохранения
        T_ego_global = T_world_veh @ T_AXIS_INV_4x4
        ego2global_rot = t3d.quaternions.mat2quat(T_ego_global[:3, :3]) # w, x, y, z
        ego2global_trans = T_ego_global[:3, 3]
        
        # Sensor2Ego для лидара (тривиально)
        sensor2ego_rot = np.array([1.0, 0.0, 0.0, 0.0])
        sensor2ego_trans = np.zeros(3)
        
        np.savez_compressed(os.path.join(sweep_dir, f"lidarjoined_{ts}.npz"),
                            points=pts_ego.astype(np.float16),
                            ego2global_translation=ego2global_trans,
                            ego2global_rotation=ego2global_rot,
                            sensor2ego_translation=sensor2ego_trans,
                            sensor2ego_rotation=sensor2ego_rot)

        # --- Cuboids ---
        cuboid_data = all_cuboids[i]
        if cuboid_data is not None:
            b_ego, l, u = cuboid_data
            # Сохраняем боксы
            np.savez_compressed(os.path.join(sweep_dir, f"boxes3d_{ts}.npz"),
                                boxes=b_ego.astype(np.float32), # x y z dx dy dz yaw
                                class_names=np.array(l),             # массив строк
                                uuids=np.array(u),             # массив строк
                                ego2global_translation=ego2global_trans,
                                ego2global_rotation=ego2global_rot,
                                sensor2ego_translation=sensor2ego_trans,
                                sensor2ego_rotation=sensor2ego_rot)
        else:
             np.savez_compressed(os.path.join(sweep_dir, f"boxes3d_{ts}.npz"),
                                boxes=np.zeros((0,7)),
                                class_names=np.array([]),
                                uuids=np.array([]),
                                ego2global_translation=ego2global_trans,
                                ego2global_rotation=ego2global_rot,
                                sensor2ego_translation=sensor2ego_trans,
                                sensor2ego_rotation=sensor2ego_rot)

        # --- GPS ---
        if i < len(gps_data):
            g = gps_data[i]
            speed = np.sqrt(g['xvel']**2 + g['yvel']**2)
            np.savez_compressed(os.path.join(sweep_dir, f"gps_{ts}.npz"),
                                lat=g['lat'], long=g['long'], height=g['height'],
                                speed=speed)

        # --- Cameras ---
        for cam in cam_names:
            # Ищем ближайший таймстемп или берем по индексу (PandaSet синхронизирован по индексу, но проверим ts)
            # Используем ts камеры, если нужно, или просто индекс i
            cam_data_list = cam_poses_raw[cam]
            if i >= len(cam_data_list): continue
            
            # Таймстемп камеры (используем для имени файла)
            c_ts = cam_ts[cam][i] if i < len(cam_ts[cam]) else ts
            
            # Картинка
            img_path = os.path.join(camera_root, cam, f"{pkl_file.replace('.pkl', '.jpg')}")
            # PandaSet имена файлов могут отличаться, часто просто индекс
            # Проверим наличие .jpg
            if not os.path.exists(img_path):
                 # Попробуем найти по индексу
                 files = glob.glob(os.path.join(camera_root, cam, '*.jpg'))
                 if len(files) > i:
                     img_path = files[i]
                 else:
                     continue

            # Копируем картинку
            shutil.copy(img_path, os.path.join(sweep_dir, f"{cam}_{c_ts}.jpg"))
            
            # Калибровки и Позу
            # Intrinsics
            intr_dict = CALIBRATIONS_DATA.get(cam, {}).get('intrinsic')
            intr_path = os.path.join(camera_root, cam, 'intrinsics.json')
            if os.path.exists(intr_path):
                with open(intr_path, 'r') as f:
                    intr_dict = json.load(f)
            
            K = np.array([
                [intr_dict['fx'], 0, intr_dict['cx']],
                [0, intr_dict['fy'], intr_dict['cy']],
                [0, 0, 1]
            ], dtype=np.float64)

            # Extrinsic: T_ego_to_cam
            cam_pose_data = cam_data_list[i]
            T_world_cam = _heading_position_to_mat(cam_pose_data['heading'], cam_pose_data['position'])
            T_cam_world = np.linalg.inv(T_world_cam)
            
            T_world_veh = poses[i]
            
            # Chain: Ego -> Veh -> World -> Cam
            T_ego_to_cam = T_cam_world @ T_world_veh @ T_AXIS_INV_4x4
            
            # Нам нужен Sensor2Ego = inv(Ego2Sensor) = inv(T_ego_to_cam)
            T_cam_to_ego = np.linalg.inv(T_ego_to_cam)
            
            sensor2ego_rot = t3d.quaternions.mat2quat(T_cam_to_ego[:3, :3]) # w, x, y, z
            sensor2ego_trans = T_cam_to_ego[:3, 3]
            
            # Сохраняем мету камеры в отдельный npz (так как jpg не хранит массивы)
            np.savez_compressed(os.path.join(sweep_dir, f"{cam}_{c_ts}.npz"),
                                intrinsic=K,
                                ego2global_translation=ego2global_trans,
                                ego2global_rotation=ego2global_rot,
                                sensor2ego_translation=sensor2ego_trans,
                                sensor2ego_rotation=sensor2ego_rot)

if __name__ == "__main__":
    DATA_ROOT = '/mnt/nvme/datasets/pandaset'
    OUTPUT_ROOT = '/mnt/nvme/datasets/pandaset_converted'
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Получаем список всех папок (001, 002, etc)
    sequences = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    for seq in sequences:
        # Пример только для 001 для теста
        # if seq == '001':
        convert_sequence(os.path.join(DATA_ROOT, seq), OUTPUT_ROOT)
            # break