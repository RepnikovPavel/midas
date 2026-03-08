import numpy as np
from numba import njit, typed, types, float32, int64
import math

# --------------------------------------------------------------------------------
# Numba JIT Compiled Functions (Core Logic - FP32)
# --------------------------------------------------------------------------------

@njit
def xy2theta_fp32(x, y):
    angle = math.atan2(y, x)
    if angle > 0:
        return angle
    else:
        return 2 * math.pi + angle

@njit
def xy2radius_fp32(x, y):
    return math.sqrt(x * x + y * y)

@njit
def calc_mean_stdev_fp32(vec):
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
def eigen_3x3_symmetric_fp32(A):
    # Jacobi Eigenvalue Algorithm for 3x3 symmetric matrices
    # Works in-place on a copy to avoid modifying input
    D = A.copy()
    V = np.eye(3, dtype=np.float32)
    
    # Iterate until convergence (standard Jacobi rotation)
    # For 3x3, usually converges very quickly (< 10 iterations)
    for _ in range(10):
        # Find largest off-diagonal element
        p, q = 0, 1
        max_val = abs(D[0, 1])
        
        if abs(D[0, 2]) > max_val:
            p, q = 0, 2
            max_val = abs(D[0, 2])
        if abs(D[1, 2]) > max_val:
            p, q = 1, 2
            max_val = abs(D[1, 2])
            
        if max_val < 1e-7: # Convergence threshold for float32
            break
            
        # Compute rotation angle
        # theta = 0.5 * atan2(2 * D[p,q], D[q,q] - D[p,p])
        diff = D[q, q] - D[p, p]
        if abs(diff) < 1e-10:
            theta = math.pi / 4.0
            if D[p, q] < 0:
                theta = -theta
        else:
            theta = 0.5 * math.atan2(2.0 * D[p, q], diff)
            
        c = math.cos(theta)
        s = math.sin(theta)
        
        # Apply Jacobi rotation to D (D = R^T * D * R)
        # Updating only the affected rows/cols
        for i in range(3):
            dip = D[i, p]
            diq = D[i, q]
            D[i, p] = c * dip - s * diq
            D[i, q] = s * dip + c * diq
            
        for i in range(3):
            dpi = D[p, i]
            dqi = D[q, i]
            D[p, i] = c * dpi - s * dqi
            D[q, i] = s * dpi + c * dqi
            
        # Update Eigenvector matrix V (V = V * R)
        for i in range(3):
            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq
            
    S = np.array([D[0, 0], D[1, 1], D[2, 2]], dtype=np.float32)
    
    # Sort eigenvalues descending (S[0] >= S[1] >= S[2])
    # Standard SVD returns descending order, logic relies on S[2] being smallest (flatness)
    
    # Bubble sort for 3 elements
    if S[0] < S[1]:
        S[0], S[1] = S[1], S[0]
        for i in range(3): V[i, 0], V[i, 1] = V[i, 1], V[i, 0]
        
    if S[0] < S[2]:
        S[0], S[2] = S[2], S[0]
        for i in range(3): V[i, 0], V[i, 2] = V[i, 2], V[i, 0]
        
    if S[1] < S[2]:
        S[1], S[2] = S[2], S[1]
        for i in range(3): V[i, 1], V[i, 2] = V[i, 2], V[i, 1]
        
    return V, S

@njit
def estimate_plane_fp32(points):
    n_pts = points.shape[0]
    if n_pts == 0:
        return np.zeros(3, dtype=np.float32), 0.0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    pc_mean = np.zeros(3, dtype=np.float32)
    for i in range(n_pts):
        pc_mean += points[i, :3]
    pc_mean /= n_pts
    
    cov = np.zeros((3, 3), dtype=np.float32)
    for i in range(n_pts):
        p = points[i, :3] - pc_mean
        cov += np.outer(p, p)
    
    if n_pts > 1:
        cov /= (n_pts - 1)
    
    # Custom SVD / Eigenvalue decomposition replacement
    U, S = eigen_3x3_symmetric_fp32(cov)
    
    # Normal is the eigenvector corresponding to the smallest eigenvalue (last column)
    normal = U[:, 2]
    
    if normal[2] < 0:
        normal = -normal
        
    d = -np.sum(normal * pc_mean)
    
    return normal, d, S, pc_mean

@njit
def extract_initial_seeds_fp32(zone_idx, p_sorted, sensor_height, adaptive_seed_selection_margin, num_lpr, th_seed):
    n_pts = p_sorted.shape[0]
    if n_pts == 0:
        return np.zeros((0, 4), dtype=np.float32)

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
        
    seeds_list = typed.List.empty_list(float32[:])
    for i in range(n_pts):
        if p_sorted[i, 2] < lpr_height + th_seed:
            seeds_list.append(p_sorted[i].copy())
            
    seeds_arr = np.zeros((len(seeds_list), 4), dtype=np.float32)
    for k in range(len(seeds_list)):
        seeds_arr[k] = seeds_list[k]
        
    return seeds_arr

@njit
def extract_piecewiseground_fp32(zone_idx, src, sensor_height, adaptive_seed_selection_margin, num_lpr, num_iter, 
                            th_seeds_v, th_dist_v, th_seeds, th_dist, uprightness_thr, enable_RVPF):
    
    non_ground_dst_list = typed.List.empty_list(float32[:])
    current_src_list = typed.List.empty_list(float32[:])
    for i in range(src.shape[0]):
        current_src_list.append(src[i])
        
    if enable_RVPF:
        for _ in range(num_iter):
            if len(current_src_list) == 0:
                break
                
            temp_arr = np.zeros((len(current_src_list), 4), dtype=np.float32)
            for k in range(len(current_src_list)):
                temp_arr[k] = current_src_list[k]
            
            sort_indices = np.argsort(temp_arr[:, 2])
            p_sorted = temp_arr[sort_indices]
            
            seeds = extract_initial_seeds_fp32(zone_idx, p_sorted, sensor_height, adaptive_seed_selection_margin, num_lpr, th_seeds_v)
            
            if seeds.shape[0] == 0:
                break
                
            normal, d, S, pc_mean = estimate_plane_fp32(seeds)
            
            if zone_idx == 0 and normal[2] < uprightness_thr:
                next_src_list = typed.List.empty_list(float32[:])
                
                for point in current_src_list:
                    dist = normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2] + d
                    if abs(dist) < th_dist_v:
                        non_ground_dst_list.append(point)
                    else:
                        next_src_list.append(point)
                
                current_src_list = next_src_list
            else:
                break
    
    if len(current_src_list) == 0:
        dst_arr = np.zeros((0, 4), dtype=np.float32)
        non_ground_arr = np.zeros((len(non_ground_dst_list), 4), dtype=np.float32)
        for k in range(len(non_ground_dst_list)):
            non_ground_arr[k] = non_ground_dst_list[k]
        return dst_arr, non_ground_arr

    src_arr = np.zeros((len(current_src_list), 4), dtype=np.float32)
    for k in range(len(current_src_list)):
        src_arr[k] = current_src_list[k]
        
    sort_indices = np.argsort(src_arr[:, 2])
    p_sorted = src_arr[sort_indices]
    
    init_seeds = extract_initial_seeds_fp32(zone_idx, p_sorted, sensor_height, adaptive_seed_selection_margin, num_lpr, th_seeds)
    
    if init_seeds.shape[0] == 0:
        for p in current_src_list:
            non_ground_dst_list.append(p)
        dst_arr = np.zeros((0, 4), dtype=np.float32)
        non_ground_arr = np.zeros((len(non_ground_dst_list), 4), dtype=np.float32)
        for k in range(len(non_ground_dst_list)):
            non_ground_arr[k] = non_ground_dst_list[k]
        return dst_arr, non_ground_arr

    ground_pc_arr = init_seeds
    
    for i in range(num_iter):
        normal, d, S, pc_mean = estimate_plane_fp32(ground_pc_arr)
        
        ground_list = typed.List.empty_list(float32[:])
        
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
            ground_pc_arr = np.zeros((len(ground_list), 4), dtype=np.float32)
            for k in range(len(ground_list)):
                ground_pc_arr[k] = ground_list[k]
        else:
            dst_arr = np.zeros((len(ground_list), 4), dtype=np.float32)
            for k in range(len(ground_list)):
                dst_arr[k] = ground_list[k]
                
    non_ground_arr = np.zeros((len(non_ground_dst_list), 4), dtype=np.float32)
    for k in range(len(non_ground_dst_list)):
        non_ground_arr[k] = non_ground_dst_list[k]
                
    return dst_arr, non_ground_arr

@njit
def temporal_ground_revert_fp32(ring_flatness, candidates_flatness, candidates_line_var, 
                           candidates_ground_points_list, th_dist):
    mean_flatness, stdev_flatness = calc_mean_stdev_fp32(ring_flatness)
    
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
def update_thresholds_core_fp32(update_elevation, update_flatness, elevation_thr_arr, flatness_thr_arr, 
                           sensor_height_arr, max_elevation_storage, max_flatness_storage):
    # Update Elevation
    for i in range(len(update_elevation)):
        if len(update_elevation[i]) == 0: continue
        
        mean, stdev = calc_mean_stdev_fp32(update_elevation[i])
        
        if i == 0:
            elevation_thr_arr[i] = mean + 3 * stdev
            sensor_height_arr[0] = -mean
        else:
            elevation_thr_arr[i] = mean + 2 * stdev
            
        exceed_num = len(update_elevation[i]) - max_elevation_storage
        if exceed_num > 0:
            for _ in range(exceed_num):
                update_elevation[i].pop(0)
                
    # Update Flatness
    for i in range(len(update_flatness)):
        if len(update_flatness[i]) <= 1: continue
        
        mean, stdev = calc_mean_stdev_fp32(update_flatness[i])
        flatness_thr_arr[i] = mean + stdev
        
        exceed_num = len(update_flatness[i]) - max_flatness_storage
        if exceed_num > 0:
            for _ in range(exceed_num):
                update_flatness[i].pop(0)

@njit
def ground_filter_core_stateful_fp32(pts, params_tuple, update_elevation, update_flatness, 
                                elevation_thr_arr, flatness_thr_arr, sensor_height_arr):
    (verbose, enable_RVPF, enable_TGR, num_iter, num_lpr, num_min_pts, 
     num_zones, num_rings_of_interest,  
     sensor_height_init, th_seeds, th_dist, th_seeds_v, th_dist_v, max_range, min_range, 
     uprightness_thr, adaptive_seed_selection_margin, 
     num_sectors_each_zone, num_rings_each_zone, 
     max_flatness_storage, max_elevation_storage, min_ranges, ring_sizes, sector_sizes) = params_tuple

    n_pts = pts.shape[0]
    
    semantic = np.empty(n_pts, dtype=np.int32)
    semantic.fill(1) 

    current_sensor_height = sensor_height_arr[0]

    # CZM
    czm = typed.List()
    for k in range(num_zones):
        zone = typed.List()
        for r in range(num_rings_each_zone[k]):
            ring = typed.List()
            for s in range(num_sectors_each_zone[k]):
                ring.append(typed.List.empty_list(float32[:]))
            zone.append(ring)
        czm.append(zone)

    for i in range(n_pts):
        # RNR removed
        
        x, y, z = pts[i] # Only XYZ now
        r = math.sqrt(x*x + y*y)
        
        if not (r <= max_range and r > min_range):
            if semantic[i] != 2: # Keep logic consistent though RNR is gone
                semantic[i] = 3
            continue
            
        theta = xy2theta_fp32(x, y)
        
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
        
        # Store point with index
        pt = np.array([x, y, z, float(i)], dtype=np.float32)
        czm[zone_idx][ring_idx][sector_idx].append(pt)

    # Ground Estimation
    concentric_idx = 0
    
    candidates_flatness = typed.List.empty_list(float32)
    candidates_line_var = typed.List.empty_list(float32)
    candidates_ground_points = typed.List.empty_list(float32[:,:])
    ringwise_flatness = typed.List.empty_list(float32)

    for zone_idx in range(num_zones):
        for ring_idx in range(num_rings_each_zone[zone_idx]):
            for sector_idx in range(num_sectors_each_zone[zone_idx]):
                patch_list = czm[zone_idx][ring_idx][sector_idx]
                
                if len(patch_list) < num_min_pts:
                    for p in patch_list:
                        semantic[int(p[3])] = 1
                    continue
                
                patch_arr = np.zeros((len(patch_list), 4), dtype=np.float32)
                for k in range(len(patch_list)):
                    patch_arr[k] = patch_list[k]
                
                sort_indices = np.argsort(patch_arr[:, 2])
                p_sorted = patch_arr[sort_indices]
                
                regionwise_ground, regionwise_nonground = extract_piecewiseground_fp32(
                    zone_idx, p_sorted, current_sensor_height, adaptive_seed_selection_margin, num_lpr, num_iter,
                    th_seeds_v, th_dist_v, th_seeds, th_dist, uprightness_thr, enable_RVPF
                )
                
                normal, d, singular_values, pc_mean = estimate_plane_fp32(regionwise_ground)
                
                ground_uprightness = normal[2]
                ground_elevation = pc_mean[2]
                ground_flatness = singular_values[2]
                
                line_variable = np.inf
                if singular_values[1] != 0:
                    line_variable = singular_values[0] / singular_values[1]
                
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
                    update_elevation[concentric_idx].append(ground_elevation)
                    update_flatness[concentric_idx].append(ground_flatness)
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
                    revert_indices = temporal_ground_revert_fp32(
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
            
    update_thresholds_core_fp32(update_elevation, update_flatness, elevation_thr_arr, flatness_thr_arr, 
                           sensor_height_arr, max_elevation_storage, max_flatness_storage)
            
    return semantic

# --------------------------------------------------------------------------------
# Python Wrapper Class (Stateful - FP32)
# --------------------------------------------------------------------------------

class PatchWorkpp_fp32:
    def __init__(self, params=None):
        if params is None:
            params = {}
            
        self.verbose = params.get('verbose', False)
        # RNR removed
        self.enable_RVPF = params.get('enable_RVPF', True)
        self.enable_TGR = params.get('enable_TGR', True)
        self.num_iter = params.get('num_iter', 3)
        self.num_lpr = params.get('num_lpr', 20)
        self.num_min_pts = params.get('num_min_pts', 10)
        self.num_zones = params.get('num_zones', 4)
        self.num_rings_of_interest = params.get('num_rings_of_interest', 4)
        
        # RNR parameters removed
        
        self.initial_sensor_height = params.get('sensor_height', 1.723)
        self.th_seeds = params.get('th_seeds', 0.125)
        self.th_dist = params.get('th_dist', 0.125)
        self.th_seeds_v = params.get('th_seeds_v', 0.25)
        self.th_dist_v = params.get('th_dist_v', 0.1)
        self.max_range = params.get('max_range', 80.0)
        self.min_range = params.get('min_range', 2.7)
        self.uprightness_thr = params.get('uprightness_thr', 0.707)
        self.adaptive_seed_selection_margin = params.get('adaptive_seed_selection_margin', -1.2)
        # Intensity thr removed
        
        self.num_sectors_each_zone = np.array(params.get('num_sectors_each_zone', [16, 32, 54, 32]), dtype=np.int64)
        self.num_rings_each_zone = np.array(params.get('num_rings_each_zone', [2, 4, 4, 4]), dtype=np.int64)
        
        self.max_flatness_storage = params.get('max_flatness_storage', 1000)
        self.max_elevation_storage = params.get('max_elevation_storage', 1000)
        
        # STATE INITIALIZATION (FP32)
        self.update_elevation_ = typed.List()
        self.update_flatness_ = typed.List()
        for _ in range(4):
            self.update_elevation_.append(typed.List.empty_list(float32))
            self.update_flatness_.append(typed.List.empty_list(float32))
        
        self.elevation_thr = np.array(params.get('elevation_thr', [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.flatness_thr = np.array(params.get('flatness_thr', [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        self.sensor_height_ = np.array([self.initial_sensor_height], dtype=np.float32)
        
        min_range_z2 = (7 * self.min_range + self.max_range) / 8.0
        min_range_z3 = (3 * self.min_range + self.max_range) / 4.0
        min_range_z4 = (self.min_range + self.max_range) / 2.0
        self.min_ranges = np.array([self.min_range, min_range_z2, min_range_z3, min_range_z4], dtype=np.float32)
        
        self.ring_sizes = np.array([
            (min_range_z2 - self.min_range) / self.num_rings_each_zone[0],
            (min_range_z3 - min_range_z2) / self.num_rings_each_zone[1],
            (min_range_z4 - min_range_z3) / self.num_rings_each_zone[2],
            (self.max_range - min_range_z4) / self.num_rings_each_zone[3]
        ], dtype=np.float32)
        
        self.sector_sizes = np.array([
            2 * math.pi / self.num_sectors_each_zone[0],
            2 * math.pi / self.num_sectors_each_zone[1],
            2 * math.pi / self.num_sectors_each_zone[2],
            2 * math.pi / self.num_sectors_each_zone[3]
        ], dtype=np.float32)

    @property
    def sensor_height(self):
        return self.sensor_height_[0]

    def forward(self, cloud_in, verbose=False):
        # Ensure FP32
        pts = cloud_in.astype(np.float32)
        
        # Handle XYZ input (N, 3) or (N, 4). If N,4, we strip the intensity just in case.
        if pts.shape[1] == 4:
            # Take only XYZ
            temp_pts = np.zeros((pts.shape[0], 3), dtype=np.float32)
            temp_pts[:, :3] = pts[:, :3]
            pts = temp_pts
        elif pts.shape[1] == 3:
            # Use as is
            pass
        else:
             raise ValueError(f"Input points must have shape [N, 3] or [N, 4], got {pts.shape}")

        valid_mask = np.isfinite(pts).all(axis=1)
        clean_pts = pts[valid_mask]
        
        params_tuple = (
            self.verbose, self.enable_RVPF, self.enable_TGR,
            self.num_iter, self.num_lpr, self.num_min_pts, self.num_zones,
            self.num_rings_of_interest, 
            self.initial_sensor_height,
            self.th_seeds, self.th_dist, self.th_seeds_v, self.th_dist_v,
            self.max_range, self.min_range, self.uprightness_thr,
            self.adaptive_seed_selection_margin, 
            self.num_sectors_each_zone, self.num_rings_each_zone,
            self.max_flatness_storage, self.max_elevation_storage,
            self.min_ranges, self.ring_sizes, self.sector_sizes
        )
        
        semantic = ground_filter_core_stateful_fp32(
            clean_pts, params_tuple, 
            self.update_elevation_, self.update_flatness_, 
            self.elevation_thr, self.flatness_thr, self.sensor_height_
        )
        
        if verbose:
            print(f"PatchWork++ (FP32) State Update sensor_height: {self.sensor_height_[0]:.4f}")
            print(f"PatchWork++ (FP32) State Update elevation_thr: {self.elevation_thr}")
            print(f"PatchWork++ (FP32) State Update flatness_thr:  {self.flatness_thr}")

        full_semantic = np.empty(len(cloud_in), dtype=np.int32)
        full_semantic.fill(1)
        full_semantic[valid_mask] = semantic
        
        return full_semantic

# --------------------------------------------------------------------------------
# Stateless Function Interface (FP32)
# --------------------------------------------------------------------------------

def GroundFilterForward_fp32(
    pts, 
    verbose=False,
    enable_RVPF=True,
    enable_TGR=True,
    num_iter=3,
    num_lpr=20,
    num_min_pts=10,
    num_zones=4,
    num_rings_of_interest=4,
    sensor_height=1.723,
    th_seeds=0.125,
    th_dist=0.125,
    th_seeds_v=0.25,
    th_dist_v=0.1,
    max_range=80.0,
    min_range=2.7,
    uprightness_thr=0.707,
    adaptive_seed_selection_margin=-1.2,
    num_sectors_each_zone=None,
    num_rings_each_zone=None,
    elevation_thr=None,
    flatness_thr=None,
    max_flatness_storage=1000,
    max_elevation_storage=1000
):
    # 1. Defaults
    if num_sectors_each_zone is None: num_sectors_each_zone = [16, 32, 54, 32]
    if num_rings_each_zone is None: num_rings_each_zone = [2, 4, 4, 4]
    if elevation_thr is None: elevation_thr = [0.0, 0.0, 0.0, 0.0]
    if flatness_thr is None: flatness_thr = [0.0, 0.0, 0.0, 0.0]

    # 2. Input processing
    pts = pts.astype(np.float32)
    if pts.shape[1] == 4:
        # Drop intensity
        temp_pts = np.zeros((pts.shape[0], 3), dtype=np.float32)
        temp_pts[:, :3] = pts[:, :3]
        pts = temp_pts
    elif pts.shape[1] == 3:
        pass
    else:
         raise ValueError(f"Input points must have shape [N, 3] or [N, 4], got {pts.shape}")

    valid_mask = np.isfinite(pts).all(axis=1)
    clean_pts = pts[valid_mask]

    # 3. Initialize Temp State
    update_elevation = typed.List()
    update_flatness = typed.List()
    for _ in range(4):
        update_elevation.append(typed.List.empty_list(float32))
        update_flatness.append(typed.List.empty_list(float32))
    
    elevation_thr_arr = np.array(elevation_thr, dtype=np.float32)
    flatness_thr_arr = np.array(flatness_thr, dtype=np.float32)
    sensor_height_arr = np.array([sensor_height], dtype=np.float32)
    
    num_sectors_each_zone_np = np.array(num_sectors_each_zone, dtype=np.int64)
    num_rings_each_zone_np = np.array(num_rings_each_zone, dtype=np.int64)
    
    min_range_z2 = (7 * min_range + max_range) / 8.0
    min_range_z3 = (3 * min_range + max_range) / 4.0
    min_range_z4 = (min_range + max_range) / 2.0
    min_ranges = np.array([min_range, min_range_z2, min_range_z3, min_range_z4], dtype=np.float32)
    
    ring_sizes = np.array([
        (min_range_z2 - min_range) / num_rings_each_zone_np[0],
        (min_range_z3 - min_range_z2) / num_rings_each_zone_np[1],
        (min_range_z4 - min_range_z3) / num_rings_each_zone_np[2],
        (max_range - min_range_z4) / num_rings_each_zone_np[3]
    ], dtype=np.float32)
    
    sector_sizes = np.array([
        2 * math.pi / num_sectors_each_zone_np[0],
        2 * math.pi / num_sectors_each_zone_np[1],
        2 * math.pi / num_sectors_each_zone_np[2],
        2 * math.pi / num_sectors_each_zone_np[3]
    ], dtype=np.float32)

    # 4. Params Tuple
    params_tuple = (
        verbose, enable_RVPF, enable_TGR,
        num_iter, num_lpr, num_min_pts, num_zones,
        num_rings_of_interest, 
        sensor_height, # initial
        th_seeds, th_dist, th_seeds_v, th_dist_v,
        max_range, min_range, uprightness_thr,
        adaptive_seed_selection_margin, 
        num_sectors_each_zone_np, num_rings_each_zone_np,
        max_flatness_storage, max_elevation_storage,
        min_ranges, ring_sizes, sector_sizes
    )

    # 5. Core Execution
    semantic = ground_filter_core_stateful_fp32(
        clean_pts, params_tuple, 
        update_elevation, update_flatness, 
        elevation_thr_arr, flatness_thr_arr, sensor_height_arr
    )
    
    # 6. Output
    full_semantic = np.empty(len(pts), dtype=np.int32)
    full_semantic.fill(1)
    full_semantic[valid_mask] = semantic
    
    return full_semantic