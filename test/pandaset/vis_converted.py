import os
import glob
import numpy as np
import cv2
import transforms3d as t3d
from tqdm import tqdm
from visualizer import LidarVisualizer
import bisect

# Константа максимальной допустимой разницы во времени (в миллисекундах)
MAX_TIME_DIFF_MS = 25  # Увеличил до 50, так как в вашем листинге разница была ~50мс

class Snapshot:
    """
    Контейнер для данных одного кадра.
    Данные загружаются с диска в момент обращения.
    """
    def __init__(self, ts, sweep_path, resolved_cam_paths):
        self.ts = ts
        self.sweep_path = sweep_path
        # Словарь с уже вычисленными путями к файлам камер {'front_camera': '/path/to/file.jpg', ...}
        self._resolved_cam_paths = resolved_cam_paths
        
        self._lidar = None
        self._boxes = None
        self._gps = None
        self._cameras = None

    @property
    def lidar(self):
        if self._lidar is None:
            self._lidar = self._load_lidar()
        return self._lidar

    @property
    def boxes(self):
        if self._boxes is None:
            self._boxes = self._load_boxes()
        return self._boxes

    @property
    def gps(self):
        if self._gps is None:
            self._gps = self._load_gps()
        return self._gps

    @property
    def cameras(self):
        if self._cameras is None:
            self._cameras = self._load_cameras()
        return self._cameras
    
    @property
    def points(self):
        return self.lidar.get('points', np.zeros((0,3), dtype=np.float32))

    def _load_lidar(self):
        path = os.path.join(self.sweep_path, f"lidarjoined_{self.ts}.npz")
        if os.path.exists(path):
            d = np.load(path)
            return {
                'points': d['points'].astype(np.float32),
                'ego2global_translation': d['ego2global_translation'],
                'ego2global_rotation': d['ego2global_rotation']
            }
        return {}

    def _load_boxes(self):
        path = os.path.join(self.sweep_path, f"boxes3d_{self.ts}.npz")
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            return {
                'boxes': d['boxes'],
                'class_names': d['class_names'],
                'uuids': d['uuids']
            }
        return {}

    def _load_gps(self):
        path = os.path.join(self.sweep_path, f"gps_{self.ts}.npz")
        if os.path.exists(path):
            d = np.load(path)
            return {
                'lat': float(d['lat']),
                'long': float(d['long']),
                'height': float(d['height']),
                'speed': float(d['speed'])
            }
        return {}

    def _load_cameras(self):
        cameras = {}
        # Просто идем по заранее вычисленным путям
        for cam_name, cpath in self._resolved_cam_paths.items():
            if not os.path.exists(cpath):
                continue
                
            img = cv2.imread(cpath)
            if img is None:
                continue
            
            meta_path = cpath.replace(".jpg", ".npz")
            cam_data = {'image': img, 'K': None}
            
            if os.path.exists(meta_path):
                d = np.load(meta_path)
                cam_data['K'] = d['intrinsic']
                cam_data['sensor2ego_translation'] = d['sensor2ego_translation']
                cam_data['sensor2ego_rotation'] = d['sensor2ego_rotation']
            cameras[cam_name] = cam_data
            
        return cameras


class Sweep:
    """
    Представляет одну поездку (sweep).
    Индексирует файлы внутри папки, сопоставляет таймстемпы и возвращает Snapshot по индексу.
    """
    def __init__(self, path):
        self.path = path
        self.snapshots = [] # Список объектов Snapshot
        self._index_files()

    def _index_files(self):
        # 1. Сбор всех файлов камер и группировка по именам
        cam_index = {} # { 'front_camera': [ (ts, path), ... ] }
        
        # Сканируем только jpg файлы
        cam_files = glob.glob(os.path.join(self.path, "*.jpg"))
        for cpath in cam_files:
            fname = os.path.basename(cpath)
            # Формат: camera_name_timestamp.jpg
            parts = fname.rsplit('_', 1)
            if len(parts) == 2:
                cam_name = parts[0]
                ts_str = parts[1].replace('.jpg', '')
                try:
                    ts = int(ts_str)
                    if cam_name not in cam_index:
                        cam_index[cam_name] = []
                    cam_index[cam_name].append((ts, cpath))
                except ValueError:
                    continue
        
        # Сортировка таймстемпов камер для бинарного поиска
        sorted_cam_index = {}
        for cam_name, entries in cam_index.items():
            # Сортируем по ts (элемент 0 кортежа)
            entries.sort(key=lambda x: x[0])
            # Оставляем только список ts для bisect
            sorted_cam_index[cam_name] = {
                'timestamps': [e[0] for e in entries],
                'paths': [e[1] for e in entries]
            }

        # 2. Индексация лидарных файлов (основа для кадров)
        lidar_files = sorted(glob.glob(os.path.join(self.path, "lidarjoined_*.npz")))
        
        for f in lidar_files:
            fname = os.path.basename(f)
            ts_str = fname.replace("lidarjoined_", "").replace(".npz", "")
            try:
                ts = int(ts_str)
            except ValueError:
                continue
            
            # 3. Поиск подходящих камер для данного ts лидара
            resolved_cam_paths = {}
            
            for cam_name, data in sorted_cam_index.items():
                cam_ts_list = data['timestamps']
                cam_paths_list = data['paths']
                
                if not cam_ts_list:
                    continue
                
                # Бинарный поиск ближайшего индекса
                idx = bisect.bisect_left(cam_ts_list, ts)
                
                candidates_idx = []
                # Сосед справа (или точное совпадение)
                if idx < len(cam_ts_list):
                    candidates_idx.append(idx)
                # Сосед слева
                if idx > 0:
                    candidates_idx.append(idx - 1)
                
                best_path = None
                min_diff = float('inf')
                
                for cand_idx in candidates_idx:
                    diff = abs(cam_ts_list[cand_idx] - ts)
                    if diff < min_diff:
                        min_diff = diff
                        best_path = cam_paths_list[cand_idx]
                
                # Проверка порога
                if best_path is not None and min_diff <= MAX_TIME_DIFF_MS:
                    resolved_cam_paths[cam_name] = best_path

            # 4. Создание Snapshot с уже известными путями
            self.snapshots.append(Snapshot(ts, self.path, resolved_cam_paths))

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        if idx >= len(self.snapshots):
            raise IndexError("Index out of range")
        return self.snapshots[idx]


class Dataset:
    """
    Главный класс для работы с датасетом.
    Поддерживает режимы полной предзагрузки и ленивой загрузки свипов.
    """
    def __init__(self, root_dir, preindex_all_sweep_files=False):
        self.root_dir = root_dir
        self.sweep_dirs = [] # Список путей к папкам свипов
        self.sweeps = {}     # Кэш загруженных объектов Sweep {idx: Sweep}
        
        # Всегда сканируем папки (это быстро)
        self._scan_sweep_dirs()
        
        if preindex_all_sweep_files:
            self._preload_all_sweeps()

    def _scan_sweep_dirs(self):
        """Быстрое сканирование папок (без парсинга файлов внутри)"""
        print("Scanning sweep directories...")
        dirs = sorted(glob.glob(os.path.join(self.root_dir, "sweep_*")))
        self.sweep_dirs = [d for d in dirs if os.path.isdir(d)]
        print(f"Found {len(self.sweep_dirs)} sweep directories.")

    def _preload_all_sweeps(self):
        """Принудительная инициализация всех Sweep объектов"""
        print("Pre-indexing all sweeps...")
        for i, d in enumerate(tqdm(self.sweep_dirs)):
            self.sweeps[i] = Sweep(d)

    def get_sweep(self, sweep_idx):
        if sweep_idx >= len(self.sweep_dirs):
            raise IndexError("Sweep index out of range")
            
        # Ленивая загрузка, если еще не загружен
        if sweep_idx not in self.sweeps:
            print(f"Initializing sweep {sweep_idx} on demand...")
            self.sweeps[sweep_idx] = Sweep(self.sweep_dirs[sweep_idx])
            
        return self.sweeps[sweep_idx]
    
    def __len__(self):
        return len(self.sweep_dirs)

def draw_snapshot(
    pts_ego,
    boxes_ego,
    labels,
    cameras_data
):
    """
    Отрисовка снэпшота. Принимает объект Snapshot.
    """

    
    display_w, display_h = 640, 360
    grid_imgs = []
    
    # Порядок камер для грида
    cam_order = ['front_left_camera', 'front_camera', 'front_right_camera',
                 'left_camera', 'back_camera', 'right_camera']
    
    for cam_name in cam_order:
        if cam_name not in cameras_data:
            grid_imgs.append(np.zeros((display_h, display_w, 3), dtype=np.uint8))
            continue
            
        cam = cameras_data[cam_name]
        img = cam['image'].copy() if cam['image'] is not None else np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Если есть калибрация и точки
        if cam['K'] is not None and pts_ego.shape[0] > 0:
            K = cam['K']
            t_vec = cam['sensor2ego_translation']
            q_vec = cam['sensor2ego_rotation'] # w, x, y, z
            
            # Трансформация из Ego в Cam
            # sensor2ego: P_ego = R * P_cam + t  =>  P_cam = R.T * (P_ego - t)
            R_mat = t3d.quaternions.quat2mat(q_vec)
            
            # Точки
            pts_cam = (R_mat.T @ (pts_ego - t_vec).T).T
            mask = pts_cam[:, 2] > 0.5
            pts_cam = pts_cam[mask]
            
            if pts_cam.shape[0] > 0:
                uv = (K @ pts_cam.T).T
                uv[:, 0] /= uv[:, 2]
                uv[:, 1] /= uv[:, 2]
                depths = pts_cam[:, 2]
                max_depth = depths.max() + 1e-6
                
                for j in range(len(uv)):
                    u, v = int(uv[j, 0]), int(uv[j, 1])
                    d = depths[j]
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        norm_d = min(d / max_depth, 1.0)
                        color = (int(255 * (1 - norm_d)), 0, int(255 * norm_d))
                        cv2.circle(img, (u, v), 1, color, -1)

            # Боксы
            if boxes_ego.shape[0] > 0:
                corners_unit = np.array([
                    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
                ])
                lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
                
                for k in range(boxes_ego.shape[0]):
                    pos = boxes_ego[k, :3]
                    dim = boxes_ego[k, 3:6]
                    yaw = boxes_ego[k, 6]
                    
                    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                    corners_ego = (R_yaw @ (dim * corners_unit).T).T + pos
                    corners_cam = (R_mat.T @ (corners_ego - t_vec).T).T
                    
                    # --- FIX START ---
                    # Проверяем, что МИНИМАЛЬНАЯ глубина (самая дальняя назад точка) больше порога.
                    # Если хотя бы одна точка сзади (z < 0.2), пропускаем бокс, чтобы избежать артефактов.
                    if corners_cam[:, 2].min() < 0.2:
                        continue
                    # --- FIX END ---
                    
                    uv_box = (K @ corners_cam.T).T
                    uv_box[:, 0] /= uv_box[:, 2]
                    uv_box[:, 1] /= uv_box[:, 2]
                    uv_box = uv_box[:, :2].astype(int)
                    
                    for start, end in lines:
                        cv2.line(img, tuple(uv_box[start]), tuple(uv_box[end]), (0, 255, 0), 2)
                    
                    # Лейбл
                    if len(labels) > k:
                        top_center_ego = pos + np.array([0, 0, dim[2]/2])
                        top_center_cam = R_mat.T @ (top_center_ego - t_vec)
                        if top_center_cam[2] > 0.1:
                            uv_label = K @ top_center_cam
                            u_l, v_l = int(uv_label[0]/uv_label[2]), int(uv_label[1]/uv_label[2])
                            cv2.putText(img, str(labels[k]), (u_l - 20, v_l - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        img_resized = cv2.resize(img, (display_w, display_h))
        grid_imgs.append(img_resized)
        
    # Сборка грида 2x3
    if len(grid_imgs) >= 3:
        row1 = np.hstack(grid_imgs[:3])
    else:
        row1 = np.zeros((display_h, display_w * 3, 3), dtype=np.uint8)
        
    if len(grid_imgs) >= 6:
        row2 = np.hstack(grid_imgs[3:6])
    else:
        # Дозаполняем черным если камер меньше 6
        missing = 6 - len(grid_imgs)
        if len(grid_imgs) > 3:
             existing = np.hstack(grid_imgs[3:])
             pad = np.zeros((display_h, display_w * missing, 3), dtype=np.uint8)
             row2 = np.hstack([existing, pad])
        else:
             row2 = np.zeros((display_h, display_w * 3, 3), dtype=np.uint8)
         
    return np.vstack([row1, row2])


if __name__ == "__main__":
    vis = LidarVisualizer(title="PandaSet 3D LiDAR")
    cv2.namedWindow("Snapshot View", cv2.WINDOW_NORMAL)
    
    
    DATA_ROOT = '/mnt/nvme/datasets/pandaset_converted'
    
    dataset = Dataset(DATA_ROOT, preindex_all_sweep_files=True)

    if len(dataset) > 0:
        for sweep_idx in range(len(dataset)):
            sweep = dataset.get_sweep(sweep_idx)
            
            print(f"Processing Sweep with {len(sweep)} frames...")
            
            for snapshot_idx in tqdm(range(len(sweep))):
                snapshot:Snapshot = sweep[snapshot_idx]
                
                print(snapshot.lidar.keys())
                print(snapshot.boxes.keys())

                img_grid = draw_snapshot(
                    snapshot.lidar['points'],
                    snapshot.boxes['boxes'],
                    snapshot.boxes['class_names'],
                    snapshot.cameras
                )
                vis.update(
                snapshot.lidar['points'],# pts_ego, 
                snapshot.boxes['boxes'],# snapshot.boxes,# boxes_ego, 
                None# labels_ego
                )
                vis.process_events()
                cv2.imshow("Snapshot View", img_grid)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                
                    
        cv2.destroyAllWindows()