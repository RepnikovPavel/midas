import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import bisect

MAX_TIME_DIFF_MS = 25

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


class PandaDatasetConverted:
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