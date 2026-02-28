import os
import glob
import numpy as np
import cv2
import transforms3d as t3d
from tqdm import tqdm
from visualizer import LidarVisualizer
import bisect
import math

# Попытка импорта smopy. Если нет, карта работать не будет, но скрипт не упадет сразу
try:
    import smopy
    SMOpy_AVAILABLE = True
except ImportError:
    SMOpy_AVAILABLE = False
    print("Warning: 'smopy' library not found. Map view will be disabled. Install with: pip install smopy")

# Константа максимальной допустимой разницы во времени (в миллисекундах)
MAX_TIME_DIFF_MS = 25

# Константы для перевода метров в градусы (приближенные)
# 1 градус широты ~= 111139 метров
# 1 градус долготы ~= 111139 * cos(latitude) метров
METERS_PER_DEG_LAT = 111139.0

class OSMMapVisualizer:
    def __init__(self, range_meters=102.4, zoom=17):
        """
        range_meters: радиус отображения вокруг ego-точки (по умолчанию 102.4м)
        zoom: уровень зума карты OSM (чем больше, тем детальнее, но медленнее грузится)
        """
        self.range_meters = range_meters
        self.zoom = zoom
        self.current_map = None
        self.current_bounds = None # (lat_min, lon_min, lat_max, lon_max)
        
        # Порог обновления карты (в метрах), чтобы не грузить тайлы при каждом смещении на 1см
        self.reload_threshold = 50.0 
        self.last_center_lat = None
        self.last_center_lon = None

    def _get_map_bounds(self, lat, lon):
        """Вычисляет границы карты (lat/lon) вокруг точки."""
        # Переводим метры в градусы
        delta_lat = self.range_meters / METERS_PER_DEG_LAT
        delta_lon = self.range_meters / (METERS_PER_DEG_LAT * math.cos(math.radians(lat)))

        lat_min = lat - delta_lat
        lat_max = lat + delta_lat
        lon_min = lon - delta_lon
        lon_max = lon + delta_lon
        
        return lat_min, lon_min, lat_max, lon_max

    def _load_map(self, lat, lon):
        """Загружает новую карту через smopy."""
        if not SMOpy_AVAILABLE:
            return None
            
        lat_min, lon_min, lat_max, lon_max = self._get_map_bounds(lat, lon)
        
        # Smopy принимает (lat_min, lon_min, lat_max, lon_max)
        try:
            # tilesize=512 может дать более красивую картинку, но 256 стандарт
            smap = smopy.Map(lat_min, lon_min, lat_max, lon_max, z=self.zoom, tilesize=256)
            
            # Конвертируем в OpenCV формат (RGB -> BGR)
            # smap.img_pil это PIL изображение
            img = np.array(smap.img_pil) 
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            self.current_map = img
            self.current_bounds = (lat_min, lon_min, lat_max, lon_max)
            self.last_center_lat = lat
            self.last_center_lon = lon
            return img
        except Exception as e:
            print(f"Error loading map: {e}")
            return None

    def _geo_to_pixel(self, lat, lon):
        """Переводит lat/lon в пиксели на текущем изображении карты."""
        if self.current_bounds is None or self.current_map is None:
            return None, None
        
        h, w = self.current_map.shape[:2]
        lat_min, lon_min, lat_max, lon_max = self.current_bounds

        # Нормализуем координаты
        # X (lon): от lon_min до lon_max -> от 0 до w
        # Y (lat): от lat_max до lat_min -> от 0 до h (ось Y в изображении перевернута)
        
        x_ratio = (lon - lon_min) / (lon_max - lon_min)
        y_ratio = (lat_max - lat) / (lat_max - lat_min)

        px = int(x_ratio * w)
        py = int(y_ratio * h)
        
        return px, py

    def update(self, lat, lon, heading_rad=None):
        """
        Обновляет отображение карты.
        lat, lon: координаты ego.
        heading_rad: курс (yaw) в радианах (опционально).
        """
        if not SMOpy_AVAILABLE:
            return np.zeros((400, 400, 3), dtype=np.uint8)

        # Проверяем, нужно ли перезагружать карту (сильно сдвинулись?)
        reload_needed = False
        if self.last_center_lat is None:
            reload_needed = True
        else:
            # Простая проверка дистанции
            dlat = (lat - self.last_center_lat) * METERS_PER_DEG_LAT
            dlon = (lon - self.last_center_lon) * (METERS_PER_DEG_LAT * math.cos(math.radians(lat)))
            dist = math.sqrt(dlat**2 + dlon**2)
            if dist > self.reload_threshold:
                reload_needed = True

        if reload_needed:
            self._load_map(lat, lon)
        
        if self.current_map is None:
            return np.zeros((400, 400, 3), dtype=np.uint8)

        # Копируем карту для рисования
        vis_map = self.current_map.copy()
        
        # Получаем пиксельные координаты ego
        px, py = self._geo_to_pixel(lat, lon)
        
        if px is not None and 0 <= px < vis_map.shape[1] and 0 <= py < vis_map.shape[0]:
            # Рисуем ego точку (синий круг)
            cv2.circle(vis_map, (px, py), 5, (255, 0, 0), -1)
            
            # Рисуем направление (если есть)
            if heading_rad is not None:
                length = 20 # длина стрелки в пикселях
                # heading обычно считается от севера по часовой? 
                # В датасетах часто: 0 - Восток (X), pi/2 - Север (Y).
                # Если в transforms3d используется стандартная логика:
                # x = cos(yaw), y = sin(yaw)
                
                # Для карты (верх = Север):
                # Если heading = 0 это Восток (X), то стрелка вправо.
                # Если heading = 0 это Север, то стрелка вверх.
                # Будем считать, что heading - это угол от оси X (Восток) против часовой.
                
                # Конец стрелки
                # dx = cos(heading), dy = -sin(heading) (ось Y картинки вниз)
                dx = int(length * math.cos(heading_rad))
                dy = int(-length * math.sin(heading_rad))
                
                cv2.arrowedLine(vis_map, (px, py), (px + dx, py + dy), (0, 0, 255), 2, tipLength=0.3)

        # Добавим текст с координатами
        cv2.putText(vis_map, f"Lat: {lat:.6f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(vis_map, f"Lon: {lon:.6f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        return vis_map


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
        
        if cam['K'] is not None and pts_ego.shape[0] > 0:
            K = cam['K']
            t_vec = cam['sensor2ego_translation']
            q_vec = cam['sensor2ego_rotation'] # w, x, y, z
            
            # Трансформация из Ego в Cam
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
                    
                    if corners_cam[:, 2].min() < 0.2:
                        continue
                    
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
    cv2.namedWindow("OSM Map", cv2.WINDOW_NORMAL)
    
    # Инициализируем визуализатор карты
    map_vis = OSMMapVisualizer(range_meters=102.4, zoom=17)
    
    DATA_ROOT = '/mnt/nvme/datasets/pandaset_converted'
    
    dataset = Dataset(DATA_ROOT, preindex_all_sweep_files=True)

    if len(dataset) > 0:
        for sweep_idx in range(len(dataset)):
            sweep = dataset.get_sweep(sweep_idx)
            
            print(f"Processing Sweep with {len(sweep)} frames...")
            
            for snapshot_idx in tqdm(range(len(sweep))):
                snapshot:Snapshot = sweep[snapshot_idx]
                
                # Данные для лидара/боксов
                lidar_data = snapshot.lidar
                boxes_data = snapshot.boxes
                gps_data = snapshot.gps
                
                # Отрисовка камер
                img_grid = draw_snapshot(
                    lidar_data.get('points', np.zeros((0,3))),
                    boxes_data.get('boxes', np.zeros((0,7))),
                    boxes_data.get('class_names', []),
                    snapshot.cameras
                )
                
                # Отрисовка карты
                map_img = np.zeros((400, 400, 3), dtype=np.uint8)
                if gps_data:
                    lat = gps_data['lat']
                    lon = gps_data['long']
                    
                    # Вычисляем курс (yaw) из rotation
                    heading = None
                    if 'ego2global_rotation' in lidar_data:
                        q = lidar_data['ego2global_rotation'] # w, x, y, z
                        # Преобразование кватерниона в Euler angles
                        # t3d возвращает (roll, pitch, yaw) или зависит от последовательности?
                        # Обычно quat2euler возвращает (ai, aj, ak) где 'sxyz' по умолчанию.
                        # Для получения yaw (heading) вокруг Z:
                        euler = t3d.euler.quat2euler(q, axes='sxyz')
                        # euler возвращает (x, y, z) углы?
                        # t3d.euler.quat2euler возвращает углы в радианах.
                        # Порядок осей 'sxyz' -> вращения вокруг x, y, z.
                        # Z - ось вверх. Третий элемент - это Yaw.
                        heading = euler[2] 
                        
                    map_img = map_vis.update(lat, lon, heading)
                
                vis.update(
                    lidar_data.get('points', np.zeros((0,3))), 
                    boxes_data.get('boxes', np.zeros((0,7))), 
                    None
                )
                vis.process_events()
                
                cv2.imshow("Snapshot View", img_grid)
                cv2.imshow("OSM Map", map_img)
                
                key = cv2.waitKey(1)
                if key == 27:
                    break
                
                    
        cv2.destroyAllWindows()