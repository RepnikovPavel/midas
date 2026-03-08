import numpy as np
from numba import njit, typed, types, float64, int64, boolean, void
import math

# --------------------------------------------------------------------------------
# Numba JIT Compiled Functions (Core Logic)
# --------------------------------------------------------------------------------

@njit
def xy2theta(x, y):
    angle = math.atan2(y, x)
    if angle > 0:
        return angle
    else:
        return 2 * math.pi + angle

@njit
def xy2radius(x, y):
    return math.sqrt(x * x + y * y)

@njit
def calc_mean_stdev(vec):
    n = len(vec)
    if n <= 1:
        return 0.0, 0.0
    
    mean = 0.0
    for i in range(n):
        mean += vec[i]
    mean /= n
    
    stdev = 0.0
    for i in range(n):
        diff = vec[i] - mean
        stdev += diff * diff
    stdev /= (n - 1)
    stdev = math.sqrt(stdev)
    
    return mean, stdev

@njit
def estimate_plane(points):
    """
    Оценка плоскости через PCA (SVD).
    points: (N, 4) массив, где columns 0,1,2 - x,y,z. Column 3 - idx.
    Возвращает: normal, d, singular_values, pc_mean
    """
    n_pts = points.shape[0]
    if n_pts == 0:
        return np.zeros(3, dtype=np.float64), 0.0, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    # 1. Вычисление среднего (центроида)
    pc_mean = np.zeros(3, dtype=np.float64)
    for i in range(n_pts):
        pc_mean += points[i, :3]
    pc_mean /= n_pts
    
    # 2. Вычисление ковариационной матрицы
    cov = np.zeros((3, 3), dtype=np.float64)
    for i in range(n_pts):
        p = points[i, :3] - pc_mean
        cov += np.outer(p, p)
    
    if n_pts > 1:
        cov /= (n_pts - 1)
    
    # 3. SVD
    U, S, Vt = np.linalg.svd(cov)
    
    # Нормаль - это последний столбец U
    normal = U[:, 2]
    
    # Ориентация нормали
    if normal[2] < 0:
        normal = -normal
        
    # d = -normal . pc_mean
    d = -np.sum(normal * pc_mean)
    
    return normal, d, S, pc_mean

@njit
def extract_initial_seeds(zone_idx, p_sorted, sensor_height, adaptive_seed_selection_margin, num_lpr, th_seed):
    n_pts = p_sorted.shape[0]
    if n_pts == 0:
        return np.zeros((0, 4), dtype=np.float64)

    init_idx = 0
    if zone_idx == 0:
        threshold = adaptive_seed_selection_margin * sensor_height
        for i in range(n_pts):
            if p_sorted[i, 2] < threshold:
                init_idx += 1
            else:
                break
    
    sum_val = 0.0
    cnt = 0
    limit = min(n_pts, init_idx + num_lpr)
    
    for i in range(init_idx, limit):
        sum_val += p_sorted[i, 2]
        cnt += 1
        
    lpr_height = 0.0
    if cnt != 0:
        lpr_height = sum_val / cnt
        
    seeds_list = typed.List.empty_list(float64[:])
    for i in range(n_pts):
        if p_sorted[i, 2] < lpr_height + th_seed:
            seeds_list.append(p_sorted[i].copy())
            
    seeds_arr = np.zeros((len(seeds_list), 4), dtype=np.float64)
    for k in range(len(seeds_list)):
        seeds_arr[k] = seeds_list[k]
        
    return seeds_arr

@njit
def extract_piecewiseground(zone_idx, src, sensor_height, adaptive_seed_selection_margin, num_lpr, num_iter, 
                            th_seeds_v, th_dist_v, th_seeds, th_dist, uprightness_thr, enable_RVPF):
    
    non_ground_dst_list = typed.List.empty_list(float64[:])
    current_src_list = typed.List.empty_list(float64[:])
    for i in range(src.shape[0]):
        current_src_list.append(src[i])
        
    # R-VPF
    if enable_RVPF:
        for _ in range(num_iter):
            if len(current_src_list) == 0:
                break
                
            temp_arr = np.zeros((len(current_src_list), 4), dtype=np.float64)
            for k in range(len(current_src_list)):
                temp_arr[k] = current_src_list[k]
            
            sort_indices = np.argsort(temp_arr[:, 2])
            p_sorted = temp_arr[sort_indices]
            
            seeds = extract_initial_seeds(zone_idx, p_sorted, sensor_height, adaptive_seed_selection_margin, num_lpr, th_seeds_v)
            
            if seeds.shape[0] == 0:
                break
                
            normal, d, S, pc_mean = estimate_plane(seeds)
            
            if zone_idx == 0 and normal[2] < uprightness_thr:
                next_src_list = typed.List.empty_list(float64[:])
                
                for point in current_src_list:
                    dist = normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2] + d
                    if abs(dist) < th_dist_v:
                        non_ground_dst_list.append(point)
                    else:
                        next_src_list.append(point)
                
                current_src_list = next_src_list
            else:
                break
    
    # R-GPF
    if len(current_src_list) == 0:
        dst_arr = np.zeros((0, 4), dtype=np.float64)
        non_ground_arr = np.zeros((len(non_ground_dst_list), 4), dtype=np.float64)
        for k in range(len(non_ground_dst_list)):
            non_ground_arr[k] = non_ground_dst_list[k]
        return dst_arr, non_ground_arr

    src_arr = np.zeros((len(current_src_list), 4), dtype=np.float64)
    for k in range(len(current_src_list)):
        src_arr[k] = current_src_list[k]
        
    sort_indices = np.argsort(src_arr[:, 2])
    p_sorted = src_arr[sort_indices]
    
    init_seeds = extract_initial_seeds(zone_idx, p_sorted, sensor_height, adaptive_seed_selection_margin, num_lpr, th_seeds)
    
    if init_seeds.shape[0] == 0:
        for p in current_src_list:
            non_ground_dst_list.append(p)
        dst_arr = np.zeros((0, 4), dtype=np.float64)
        non_ground_arr = np.zeros((len(non_ground_dst_list), 4), dtype=np.float64)
        for k in range(len(non_ground_dst_list)):
            non_ground_arr[k] = non_ground_dst_list[k]
        return dst_arr, non_ground_arr

    ground_pc_arr = init_seeds
    
    for i in range(num_iter):
        normal, d, S, pc_mean = estimate_plane(ground_pc_arr)
        
        ground_list = typed.List.empty_list(float64[:])
        
        for point in current_src_list:
            dist = normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2] + d
            
            if i < num_iter - 1:
                if dist < th_dist:
                    ground_list.append(point)
            else:
                if dist < th_dist:
                    ground_list.append(point)
                else:
                    non_ground_dst_list.append(point)
        
        if i < num_iter - 1:
            if len(ground_list) == 0:
                break
            ground_pc_arr = np.zeros((len(ground_list), 4), dtype=np.float64)
            for k in range(len(ground_list)):
                ground_pc_arr[k] = ground_list[k]
        else:
            dst_arr = np.zeros((len(ground_list), 4), dtype=np.float64)
            for k in range(len(ground_list)):
                dst_arr[k] = ground_list[k]
                
    non_ground_arr = np.zeros((len(non_ground_dst_list), 4), dtype=np.float64)
    for k in range(len(non_ground_dst_list)):
        non_ground_arr[k] = non_ground_dst_list[k]
                
    return dst_arr, non_ground_arr

@njit
def temporal_ground_revert(ring_flatness, candidates_flatness, candidates_line_var, 
                           candidates_ground_points_list, th_dist):
    mean_flatness, stdev_flatness = calc_mean_stdev(ring_flatness)
    
    revert_indices = typed.List.empty_list(int64)
    
    for i in range(len(candidates_flatness)):
        c_flatness = candidates_flatness[i]
        c_line_var = candidates_line_var[i]
        c_ground = candidates_ground_points_list[i]
        
        mu_flatness = mean_flatness + 1.5 * stdev_flatness
        
        denom = mu_flatness / 10.0
        if denom == 0: denom = 1e-9
        
        prob_flatness = 1.0 / (1.0 + math.exp((c_flatness - mu_flatness) / denom))
        
        if len(c_ground) > 1500 and c_flatness < th_dist * th_dist:
            prob_flatness = 1.0
            
        prob_line = 1.0
        if c_line_var > 8.0:
            prob_line = 0.0
            
        if prob_line * prob_flatness > 0.5:
            for k in range(c_ground.shape[0]):
                revert_indices.append(int(c_ground[k, 3]))
                
    return revert_indices

@njit
def ground_filter_core(pts, params_tuple, update_elevation_lists, update_flatness_lists):
    (verbose, enable_RNR, enable_RVPF, enable_TGR, num_iter, num_lpr, num_min_pts, 
     num_zones, num_rings_of_interest, RNR_ver_angle_thr, RNR_intensity_thr, 
     sensor_height, th_seeds, th_dist, th_seeds_v, th_dist_v, max_range, min_range, 
     uprightness_thr, adaptive_seed_selection_margin, intensity_thr, 
     num_sectors_each_zone, num_rings_each_zone, elevation_thr_arr, flatness_thr_arr, 
     max_flatness_storage, max_elevation_storage, min_ranges, ring_sizes, sector_sizes) = params_tuple

    n_pts = pts.shape[0]
    
    semantic = np.empty(n_pts, dtype=np.int32)
    semantic.fill(1) 

    # ---------------------------------------------------------
    # 1. Reflected Noise Removal (RNR)
    # ---------------------------------------------------------
    # Выполняем только если enable_RNR=True. 
    # Если pts пришел размером (N, 4) с нулями, проверка intensity пройдет корректно.
    if enable_RNR:
        for i in range(n_pts):
            x, y, z, intensity = pts[i]
            r = math.sqrt(x*x + y*y)
            if r == 0: continue
            ver_angle_in_deg = math.atan2(z, r) * 180.0 / math.pi
            
            if ver_angle_in_deg < RNR_ver_angle_thr and z < -sensor_height - 0.8 and intensity < RNR_intensity_thr:
                semantic[i] = 2

    # ---------------------------------------------------------
    # 2. Concentric Zone Model (CZM) Binning
    # ---------------------------------------------------------
    czm = typed.List()
    for k in range(num_zones):
        zone = typed.List()
        for r in range(num_rings_each_zone[k]):
            ring = typed.List()
            for s in range(num_sectors_each_zone[k]):
                ring.append(typed.List.empty_list(float64[:]))
            zone.append(ring)
        czm.append(zone)

    for i in range(n_pts):
        if semantic[i] == 2:
            continue
            
        x, y, z, intensity = pts[i]
        r = math.sqrt(x*x + y*y)
        
        if not (r <= max_range and r > min_range):
            if semantic[i] != 2:
                semantic[i] = 3
            continue
            
        theta = xy2theta(x, y)
        
        zone_idx = -1
        ring_idx = -1
        sector_idx = -1
        
        if r < min_ranges[1]:
            zone_idx = 0
            ring_idx = int((r - min_ranges[0]) / ring_sizes[0])
            sector_idx = int(theta / sector_sizes[0])
        elif r < min_ranges[2]:
            zone_idx = 1
            ring_idx = int((r - min_ranges[1]) / ring_sizes[1])
            sector_idx = int(theta / sector_sizes[1])
        elif r < min_ranges[3]:
            zone_idx = 2
            ring_idx = int((r - min_ranges[2]) / ring_sizes[2])
            sector_idx = int(theta / sector_sizes[2])
        else:
            zone_idx = 3
            ring_idx = int((r - min_ranges[3]) / ring_sizes[3])
            sector_idx = int(theta / sector_sizes[3])
            
        if ring_idx >= num_rings_each_zone[zone_idx]: ring_idx = num_rings_each_zone[zone_idx] - 1
        if sector_idx >= num_sectors_each_zone[zone_idx]: sector_idx = num_sectors_each_zone[zone_idx] - 1
        if ring_idx < 0: ring_idx = 0
        if sector_idx < 0: sector_idx = 0
        
        pt = np.array([x, y, z, float(i)], dtype=np.float64)
        czm[zone_idx][ring_idx][sector_idx].append(pt)

    # ---------------------------------------------------------
    # 3. Ground Estimation
    # ---------------------------------------------------------
    concentric_idx = 0
    
    candidates_flatness = typed.List.empty_list(float64)
    candidates_line_var = typed.List.empty_list(float64)
    candidates_ground_points = typed.List.empty_list(float64[:,:])
    ringwise_flatness = typed.List.empty_list(float64)

    for zone_idx in range(num_zones):
        for ring_idx in range(num_rings_each_zone[zone_idx]):
            for sector_idx in range(num_sectors_each_zone[zone_idx]):
                patch_list = czm[zone_idx][ring_idx][sector_idx]
                
                if len(patch_list) < num_min_pts:
                    for p in patch_list:
                        semantic[int(p[3])] = 1
                    continue
                
                patch_arr = np.zeros((len(patch_list), 4), dtype=np.float64)
                for k in range(len(patch_list)):
                    patch_arr[k] = patch_list[k]
                
                sort_indices = np.argsort(patch_arr[:, 2])
                p_sorted = patch_arr[sort_indices]
                
                regionwise_ground, regionwise_nonground = extract_piecewiseground(
                    zone_idx, p_sorted, sensor_height, adaptive_seed_selection_margin, num_lpr, num_iter,
                    th_seeds_v, th_dist_v, th_seeds, th_dist, uprightness_thr, enable_RVPF
                )
                
                normal, d, singular_values, pc_mean = estimate_plane(regionwise_ground)
                
                ground_uprightness = normal[2]
                ground_elevation = pc_mean[2]
                ground_flatness = singular_values[2]
                
                line_variable = np.inf
                if singular_values[1] != 0:
                    line_variable = singular_values[0] / singular_values[1]
                
                # Исправление warning: np.sum вместо np.dot для векторов
                heading = np.sum(pc_mean * normal)
                
                is_upright = ground_uprightness > uprightness_thr
                is_near_zone = concentric_idx < num_rings_of_interest
                is_heading_outside = heading < 0.0
                
                is_not_elevated = False
                is_flat = False
                
                if concentric_idx < num_rings_of_interest:
                    is_not_elevated = ground_elevation < elevation_thr_arr[concentric_idx]
                    is_flat = ground_flatness < flatness_thr_arr[concentric_idx]
                
                if is_upright and is_not_elevated and is_near_zone:
                    update_elevation_lists[concentric_idx].append(ground_elevation)
                    update_flatness_lists[concentric_idx].append(ground_flatness)
                    ringwise_flatness.append(ground_flatness)
                
                if not is_upright:
                    for k in range(regionwise_ground.shape[0]):
                        semantic[int(regionwise_ground[k, 3])] = 1
                elif not is_near_zone:
                    for k in range(regionwise_ground.shape[0]):
                        semantic[int(regionwise_ground[k, 3])] = 0
                elif not is_heading_outside:
                    for k in range(regionwise_ground.shape[0]):
                        semantic[int(regionwise_ground[k, 3])] = 1
                elif is_not_elevated or is_flat:
                    for k in range(regionwise_ground.shape[0]):
                        semantic[int(regionwise_ground[k, 3])] = 0
                else:
                    candidates_flatness.append(ground_flatness)
                    candidates_line_var.append(line_variable)
                    candidates_ground_points.append(regionwise_ground)
                
                for k in range(regionwise_nonground.shape[0]):
                    semantic[int(regionwise_nonground[k, 3])] = 1

            if len(candidates_flatness) > 0:
                if enable_TGR:
                    revert_indices = temporal_ground_revert(
                        ringwise_flatness, candidates_flatness, candidates_line_var,
                        candidates_ground_points, th_dist
                    )
                    for idx in revert_indices:
                        semantic[idx] = 0
                
                candidates_flatness.clear()
                candidates_line_var.clear()
                candidates_ground_points.clear()
                ringwise_flatness.clear()
            
            concentric_idx += 1
            
    return semantic

# --------------------------------------------------------------------------------
# Python Wrapper Class
# --------------------------------------------------------------------------------

class PatchWorkpp:
    def __init__(self, params=None):
        if params is None:
            params = {}
            
        self.verbose = params.get('verbose', False)
        self.enable_RNR = params.get('enable_RNR', True)
        self.enable_RVPF = params.get('enable_RVPF', True)
        self.enable_TGR = params.get('enable_TGR', True)
        self.num_iter = params.get('num_iter', 3)
        self.num_lpr = params.get('num_lpr', 20)
        self.num_min_pts = params.get('num_min_pts', 10)
        self.num_zones = params.get('num_zones', 4)
        self.num_rings_of_interest = params.get('num_rings_of_interest', 4)
        
        self.RNR_ver_angle_thr = params.get('RNR_ver_angle_thr', -15.0)
        self.RNR_intensity_thr = params.get('RNR_intensity_thr', 0.2)
        self.sensor_height = params.get('sensor_height', 1.723)
        self.th_seeds = params.get('th_seeds', 0.125)
        self.th_dist = params.get('th_dist', 0.125)
        self.th_seeds_v = params.get('th_seeds_v', 0.25)
        self.th_dist_v = params.get('th_dist_v', 0.1)
        self.max_range = params.get('max_range', 80.0)
        self.min_range = params.get('min_range', 2.7)
        self.uprightness_thr = params.get('uprightness_thr', 0.707)
        self.adaptive_seed_selection_margin = params.get('adaptive_seed_selection_margin', -1.2)
        self.intensity_thr = params.get('intensity_thr', 0.0)
        
        self.num_sectors_each_zone = np.array(params.get('num_sectors_each_zone', [16, 32, 54, 32]), dtype=np.int64)
        self.num_rings_each_zone = np.array(params.get('num_rings_each_zone', [2, 4, 4, 4]), dtype=np.int64)
        
        self.max_flatness_storage = params.get('max_flatness_storage', 1000)
        self.max_elevation_storage = params.get('max_elevation_storage', 1000)
        
        self.elevation_thr = np.array(params.get('elevation_thr', [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        self.flatness_thr = np.array(params.get('flatness_thr', [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        
        min_range_z2 = (7 * self.min_range + self.max_range) / 8.0
        min_range_z3 = (3 * self.min_range + self.max_range) / 4.0
        min_range_z4 = (self.min_range + self.max_range) / 2.0
        self.min_ranges = np.array([self.min_range, min_range_z2, min_range_z3, min_range_z4], dtype=np.float64)
        
        self.ring_sizes = np.array([
            (min_range_z2 - self.min_range) / self.num_rings_each_zone[0],
            (min_range_z3 - min_range_z2) / self.num_rings_each_zone[1],
            (min_range_z4 - min_range_z3) / self.num_rings_each_zone[2],
            (self.max_range - min_range_z4) / self.num_rings_each_zone[3]
        ], dtype=np.float64)
        
        self.sector_sizes = np.array([
            2 * math.pi / self.num_sectors_each_zone[0],
            2 * math.pi / self.num_sectors_each_zone[1],
            2 * math.pi / self.num_sectors_each_zone[2],
            2 * math.pi / self.num_sectors_each_zone[3]
        ], dtype=np.float64)
        
        self.update_elevation_ = [typed.List.empty_list(float64) for _ in range(4)]
        self.update_flatness_ = [typed.List.empty_list(float64) for _ in range(4)]

    def estimateGround(self, cloud_in):
        pts = cloud_in.astype(np.float64)
        
        update_elev_typed = typed.List()
        for lst in self.update_elevation_:
            update_elev_typed.append(lst)
            
        update_flat_typed = typed.List()
        for lst in self.update_flatness_:
            update_flat_typed.append(lst)
            
        params_tuple = (
            self.verbose, self.enable_RNR, self.enable_RVPF, self.enable_TGR,
            self.num_iter, self.num_lpr, self.num_min_pts, self.num_zones,
            self.num_rings_of_interest, self.RNR_ver_angle_thr, self.RNR_intensity_thr,
            self.sensor_height, self.th_seeds, self.th_dist, self.th_seeds_v, self.th_dist_v,
            self.max_range, self.min_range, self.uprightness_thr,
            self.adaptive_seed_selection_margin, self.intensity_thr,
            self.num_sectors_each_zone, self.num_rings_each_zone,
            self.elevation_thr, self.flatness_thr,
            self.max_flatness_storage, self.max_elevation_storage,
            self.min_ranges, self.ring_sizes, self.sector_sizes
        )
        
        semantic = ground_filter_core(pts, params_tuple, update_elev_typed, update_flat_typed)
        
        self.update_elevation_ = [lst for lst in update_elev_typed]
        self.update_flatness_ = [lst for lst in update_flat_typed]
        self._update_thr()
        
        return semantic

    def _update_thr(self):
        for i in range(self.num_rings_of_interest):
            if len(self.update_elevation_[i]) == 0: continue
            vec = np.array(self.update_elevation_[i])
            mean = np.mean(vec)
            stdev = np.std(vec)
            if i == 0:
                self.elevation_thr[i] = mean + 3 * stdev
                self.sensor_height = -mean
            else:
                self.elevation_thr[i] = mean + 2 * stdev
                
        for i in range(self.num_rings_of_interest):
            if len(self.update_flatness_[i]) <= 1: continue
            vec = np.array(self.update_flatness_[i])
            mean = np.mean(vec)
            stdev = np.std(vec)
            self.flatness_thr[i] = mean + stdev

# --------------------------------------------------------------------------------
# Forward Function (Interface)
# --------------------------------------------------------------------------------

def GroundFilterForward(
    pts, 
    verbose=False,
    enable_RNR=True,
    enable_RVPF=True,
    enable_TGR=True,
    num_iter=3,
    num_lpr=20,
    num_min_pts=10,
    num_zones=4,
    num_rings_of_interest=4,
    RNR_ver_angle_thr=-15.0,
    RNR_intensity_thr=0.2,
    sensor_height=1.723,
    th_seeds=0.125,
    th_dist=0.125,
    th_seeds_v=0.25,
    th_dist_v=0.1,
    max_range=80.0,
    min_range=2.7,
    uprightness_thr=0.707,
    adaptive_seed_selection_margin=-1.2,
    intensity_thr=0.0,
    num_sectors_each_zone=None,
    num_rings_each_zone=None,
    elevation_thr=None,
    flatness_thr=None,
    max_flatness_storage=1000,
    max_elevation_storage=1000
):
    if num_sectors_each_zone is None: num_sectors_each_zone = [16, 32, 54, 32]
    if num_rings_each_zone is None: num_rings_each_zone = [2, 4, 4, 4]
    if elevation_thr is None: elevation_thr = [0.0, 0.0, 0.0, 0.0]
    if flatness_thr is None: flatness_thr = [0.0, 0.0, 0.0, 0.0]

    params = {
        'verbose': verbose,
        'enable_RNR': enable_RNR,
        'enable_RVPF': enable_RVPF,
        'enable_TGR': enable_TGR,
        'num_iter': num_iter,
        'num_lpr': num_lpr,
        'num_min_pts': num_min_pts,
        'num_zones': num_zones,
        'num_rings_of_interest': num_rings_of_interest,
        'RNR_ver_angle_thr': RNR_ver_angle_thr,
        'RNR_intensity_thr': RNR_intensity_thr,
        'sensor_height': sensor_height,
        'th_seeds': th_seeds,
        'th_dist': th_dist,
        'th_seeds_v': th_seeds_v,
        'th_dist_v': th_dist_v,
        'max_range': max_range,
        'min_range': min_range,
        'uprightness_thr': uprightness_thr,
        'adaptive_seed_selection_margin': adaptive_seed_selection_margin,
        'intensity_thr': intensity_thr,
        'num_sectors_each_zone': num_sectors_each_zone,
        'num_rings_each_zone': num_rings_each_zone,
        'elevation_thr': elevation_thr,
        'flatness_thr': flatness_thr,
        'max_flatness_storage': max_flatness_storage,
        'max_elevation_storage': max_elevation_storage
    }
    
    # Автоматическое добавление колонки интенсивности, если её нет
    if pts.shape[1] == 3:
        # Создаем массив (N, 4) и заполняем x,y,z, intensity=0
        temp_pts = np.zeros((pts.shape[0], 4), dtype=np.float32)
        temp_pts[:, :3] = pts
        pts = temp_pts
    elif pts.shape[1] == 4:
        pass
    else:
        raise ValueError(f"Input points must have shape [N, 3] or [N, 4], got {pts.shape}")
    
    valid_mask = np.isfinite(pts).all(axis=1)
    clean_pts = pts[valid_mask]
    
    pw = PatchWorkpp(params)
    semantic = pw.estimateGround(clean_pts)
    
    full_semantic = np.empty(len(pts), dtype=np.int32)
    full_semantic.fill(1)
    full_semantic[valid_mask] = semantic
    
    return full_semantic