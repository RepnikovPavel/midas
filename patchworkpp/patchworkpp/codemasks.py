import numpy as np
from numba import njit, float32, int64, bool_
import math

# --------------------------------------------------------------------------------
# Numba JIT Compiled Functions (Core Logic - FP32, Array/Mask based)
# --------------------------------------------------------------------------------

@njit
def xy2theta_fp32(x, y):
    angle = math.atan2(y, x)
    if angle > 0:
        return angle
    else:
        return 2 * math.pi + angle

@njit
def calc_mean_stdev_slice_fp32(arr, n):
    """Calculate mean and stdev for the first n elements of arr."""
    if n <= 1:
        return 0.0, 0.0
    
    mean = 0.0
    for i in range(n):
        mean += arr[i]
    mean /= n
    
    stdev = 0.0
    for i in range(n):
        diff = arr[i] - mean
        stdev += diff * diff
    stdev /= (n - 1)
    stdev = math.sqrt(stdev)
    
    return mean, stdev

@njit
def calc_mean_stdev_circular_fp32(arr, head, count, max_size):
    """Mean and stdev for a circular buffer."""
    if count <= 1:
        return 0.0, 0.0
    
    mean = 0.0
    start_pos = (head - count + max_size) % max_size
    
    for i in range(count):
        idx = (start_pos + i) % max_size
        mean += arr[idx]
    mean /= count
    
    stdev = 0.0
    for i in range(count):
        idx = (start_pos + i) % max_size
        diff = arr[idx] - mean
        stdev += diff * diff
    
    stdev /= (count - 1)
    stdev = math.sqrt(stdev)
    
    return mean, stdev

@njit
def eigen_3x3_symmetric_fp32(A):
    D = A.copy()
    V = np.eye(3, dtype=np.float32)
    
    for _ in range(10):
        p, q = 0, 1
        max_val = abs(D[0, 1])
        
        if abs(D[0, 2]) > max_val:
            p, q = 0, 2
            max_val = abs(D[0, 2])
        if abs(D[1, 2]) > max_val:
            p, q = 1, 2
            max_val = abs(D[1, 2])
            
        if max_val < 1e-7:
            break
            
        diff = D[q, q] - D[p, p]
        if abs(diff) < 1e-10:
            theta = math.pi / 4.0
            if D[p, q] < 0:
                theta = -theta
        else:
            theta = 0.5 * math.atan2(2.0 * D[p, q], diff)
            
        c = math.cos(theta)
        s = math.sin(theta)
        
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
            
        for i in range(3):
            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq
            
    S = np.array([D[0, 0], D[1, 1], D[2, 2]], dtype=np.float32)
    
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
    
    U, S = eigen_3x3_symmetric_fp32(cov)
    
    normal = U[:, 2]
    
    if normal[2] < 0:
        normal = -normal
        
    d = -np.sum(normal * pc_mean)
    
    return normal, d, S, pc_mean

@njit
def extract_piecewiseground_mask_fp32(zone_idx, src_sorted, sensor_height, adaptive_seed_selection_margin, num_lpr, num_iter, 
                            th_seeds_v, th_dist_v, th_seeds, th_dist, uprightness_thr, enable_RVPF):
    
    n_pts = src_sorted.shape[0]
    if n_pts == 0:
        return np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    # Replaces 'current_src_list'
    is_candidate = np.ones(n_pts, dtype=np.bool_)
    
    # RVPF Loop
    if enable_RVPF:
        for _ in range(num_iter):
            # Count valid candidates
            cnt_candidate = 0
            for i in range(n_pts):
                if is_candidate[i]: cnt_candidate += 1
            
            if cnt_candidate == 0:
                break
            
            # We need to pick seeds. Seeds are based on lowest points.
            # src_sorted is sorted by Z. The lowest candidates are the first ones in the array that are still valid.
            # Logic: iterate src_sorted, take first num_lpr valid points for LPR.
            
            sum_val = 0.0
            lpr_cnt = 0
            for i in range(n_pts):
                if is_candidate[i]:
                    sum_val += src_sorted[i, 2]
                    lpr_cnt += 1
                    if lpr_cnt >= num_lpr: break
            
            if lpr_cnt == 0: break
            lpr_h = sum_val / lpr_cnt
            
            # Identify seeds for this RVPF step
            is_seed = np.zeros(n_pts, dtype=np.bool_)
            seed_cnt = 0
            for i in range(n_pts):
                if is_candidate[i]:
                    if src_sorted[i, 2] < lpr_h + th_seeds_v:
                        is_seed[i] = True
                        seed_cnt += 1
            
            if seed_cnt == 0: break
            
            # Estimate Plane
            seeds = src_sorted[is_seed]
            normal, d, S, pc_mean = estimate_plane_fp32(seeds)
            
            if zone_idx == 0 and normal[2] < uprightness_thr:
                # Vertical removal logic
                # Points with dist < th_dist_v are removed from candidates (marked non-ground effectively)
                for i in range(n_pts):
                    if is_candidate[i]:
                        pt = src_sorted[i]
                        dist = normal[0]*pt[0] + normal[1]*pt[1] + normal[2]*pt[2] + d
                        if abs(dist) < th_dist_v:
                            is_candidate[i] = False
            else:
                break
    
    # Main Ground Estimation Loop
    
    # Calculate LPR for main seeds
    sum_val = 0.0
    lpr_cnt = 0
    for i in range(n_pts):
        if is_candidate[i]:
            sum_val += src_sorted[i, 2]
            lpr_cnt += 1
            if lpr_cnt >= num_lpr: break
            
    lpr_h = 0.0
    if lpr_cnt > 0: lpr_h = sum_val / lpr_cnt
    
    is_ground = np.zeros(n_pts, dtype=np.bool_)
    
    # Initial Seeds identification
    for i in range(n_pts):
        if is_candidate[i]:
            if src_sorted[i, 2] < lpr_h + th_seeds:
                is_ground[i] = True
                
    ground_cnt = 0
    for i in range(n_pts):
        if is_ground[i]: ground_cnt += 1
        
    if ground_cnt == 0:
        # No seeds found
        return np.zeros(n_pts, dtype=np.bool_), is_candidate, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    for i in range(num_iter):
        ground_pts = src_sorted[is_ground]
        normal, d, S, pc_mean = estimate_plane_fp32(ground_pts)
        
        new_ground = np.zeros(n_pts, dtype=np.bool_)
        
        for k in range(n_pts):
            if is_candidate[k]:
                pt = src_sorted[k]
                dist = normal[0]*pt[0] + normal[1]*pt[1] + normal[2]*pt[2] + d
                
                if dist < th_dist:
                    new_ground[k] = True
        
        if i < num_iter - 1:
            is_ground = new_ground
        else:
            is_ground = new_ground
            
    # Determine non-ground
    # Original logic: 
    # RVPF removed points -> non-ground
    # Last iter rejected points -> non-ground
    is_nonground = np.zeros(n_pts, dtype=np.bool_)
    
    for k in range(n_pts):
        if not is_candidate[k]:
            is_nonground[k] = True
        elif not is_ground[k]:
            is_nonground[k] = True
            
    return is_ground, is_nonground, normal, S, pc_mean

@njit
def temporal_ground_revert_core_fp32(ring_flatness_arr, ring_flatness_cnt, 
                                     cand_flatness_arr, cand_line_var_arr, 
                                     cand_points_idx_buf, cand_points_start, cand_points_end,
                                     cand_cnt, th_dist, semantic):
    mean_f, stdev_f = calc_mean_stdev_slice_fp32(ring_flatness_arr, ring_flatness_cnt)
    
    for i in range(cand_cnt):
        c_flatness = cand_flatness_arr[i]
        c_line_var = cand_line_var_arr[i]
        
        mu_flatness = mean_f + 1.5 * stdev_f
        
        denom = mu_flatness / 10.0
        if denom == 0: denom = 1e-9
        
        prob_flatness = 1.0 / (1.0 + math.exp((c_flatness - mu_flatness) / denom))
        
        num_pts_in_candidate = cand_points_end[i] - cand_points_start[i]
        
        if num_pts_in_candidate > 1500 and c_flatness < th_dist * th_dist:
            prob_flatness = 1.0
            
        prob_line = 1.0
        if c_line_var > 8.0:
            prob_line = 0.0
            
        if prob_line * prob_flatness > 0.5:
            start = cand_points_start[i]
            end = cand_points_end[i]
            for k in range(start, end):
                orig_idx = cand_points_idx_buf[k]
                semantic[orig_idx] = 0

@njit
def update_thresholds_core_fp32(update_elevation_arrs, update_flatness_arrs, 
                                elevation_counts, flatness_counts,
                                elevation_heads, flatness_heads,
                                elevation_thr_arr, flatness_thr_arr, 
                                sensor_height_arr, max_elevation_storage, max_flatness_storage):
    
    for i in range(4):
        cnt = elevation_counts[i]
        if cnt <= 1: continue
        
        mean, stdev = calc_mean_stdev_circular_fp32(update_elevation_arrs[i], elevation_heads[i], cnt, max_elevation_storage)
        
        if i == 0:
            elevation_thr_arr[i] = mean + 3 * stdev
            sensor_height_arr[0] = -mean
        else:
            elevation_thr_arr[i] = mean + 2 * stdev
            
    for i in range(4):
        cnt = flatness_counts[i]
        if cnt <= 1: continue
        
        mean, stdev = calc_mean_stdev_circular_fp32(update_flatness_arrs[i], flatness_heads[i], cnt, max_flatness_storage)
        flatness_thr_arr[i] = mean + stdev

@njit
def ground_filter_core_stateful_fp32(pts, params_tuple, 
                                update_elevation_arrs, update_flatness_arrs, 
                                elevation_counts, flatness_counts,
                                elevation_heads, flatness_heads,
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

    # 1. Compute Bin Indices for all points (Flattened CZM)
    bin_ids = np.empty(n_pts, dtype=np.int64)
    
    # Pre-calculate offsets for linear indexing
    zone_bin_counts = np.zeros(4, dtype=np.int64)
    zone_bin_starts = np.zeros(4, dtype=np.int64)
    total_bins = 0
    for z in range(num_zones):
        zone_bin_counts[z] = num_rings_each_zone[z] * num_sectors_each_zone[z]
        zone_bin_starts[z] = total_bins
        total_bins += zone_bin_counts[z]
        
    for i in range(n_pts):
        x, y, z = pts[i]
        r = math.sqrt(x*x + y*y)
        
        if not (r <= max_range and r > min_range):
            bin_ids[i] = -1
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
        
        bin_id = zone_bin_starts[zone_idx] + ring_idx * num_sectors_each_zone[zone_idx] + sector_idx
        bin_ids[i] = bin_id

    # 2. Sort points by Bin ID
    sort_indices = np.argsort(bin_ids)
    
    # Buffers for TGR (Temporal Ground Revert)
    # We allocate max possible size to avoid dynamic lists
    ring_flatness_arr = np.zeros(total_bins, dtype=np.float32)
    ring_flatness_cnt = 0
    
    # Candidates buffers
    cand_flatness = np.zeros(total_bins, dtype=np.float32)
    cand_line_var = np.zeros(total_bins, dtype=np.float32)
    cand_points_idx_buf = np.zeros(n_pts, dtype=np.int64) # Stores original indices of candidate ground points
    cand_points_start = np.zeros(total_bins, dtype=np.int64)
    cand_points_end = np.zeros(total_bins, dtype=np.int64)
    cand_cnt = 0
    cand_buf_ptr = 0
    
    concentric_idx = 0
    last_ring_idx = -1
    
    # 3. Iterate through sorted points
    i = 0
    while i < n_pts:
        bin_id = bin_ids[sort_indices[i]]
        
        if bin_id == -1:
            i += 1
            continue
            
        # Determine zone/ring/sector from bin_id
        current_zone = 0
        for z in range(num_zones):
            if bin_id >= zone_bin_starts[z] and bin_id < zone_bin_starts[z] + zone_bin_counts[z]:
                current_zone = z
                break
        
        # Calculate ring index global to the sequence
        current_ring_idx = 0
        for z in range(current_zone):
            current_ring_idx += num_rings_each_zone[z]
        
        local_ring_idx = (bin_id - zone_bin_starts[current_zone]) // num_sectors_each_zone[current_zone]
        current_ring_idx += local_ring_idx
        
        # Check for Ring Transition to trigger TGR
        if current_ring_idx != last_ring_idx:
            if last_ring_idx != -1:
                # Process TGR for the finished ring
                if enable_TGR and cand_cnt > 0:
                    temporal_ground_revert_core_fp32(
                        ring_flatness_arr, ring_flatness_cnt,
                        cand_flatness, cand_line_var,
                        cand_points_idx_buf, cand_points_start, cand_points_end,
                        cand_cnt, th_dist, semantic
                    )
                
                # Reset buffers
                ring_flatness_cnt = 0
                cand_cnt = 0
                cand_buf_ptr = 0
            
            last_ring_idx = current_ring_idx
        
        # Find range of current bin
        start_idx = i
        while i < n_pts and bin_ids[sort_indices[i]] == bin_id:
            i += 1
        end_idx = i
        
        count_in_bin = end_idx - start_idx
        
        if count_in_bin < num_min_pts:
            continue
            
        # Prepare sorted points for extraction
        bin_global_indices = sort_indices[start_idx:end_idx]
        z_vals = np.zeros(count_in_bin, dtype=np.float32)
        for k in range(count_in_bin):
            z_vals[k] = pts[bin_global_indices[k], 2]
        z_sort_order = np.argsort(z_vals)
        
        src_sorted = np.zeros((count_in_bin, 4), dtype=np.float32)
        for k in range(count_in_bin):
            orig_idx = bin_global_indices[z_sort_order[k]]
            src_sorted[k, 0] = pts[orig_idx, 0]
            src_sorted[k, 1] = pts[orig_idx, 1]
            src_sorted[k, 2] = pts[orig_idx, 2]
            src_sorted[k, 3] = float(orig_idx)
            
        is_ground_mask, is_nonground_mask, normal, S, pc_mean = extract_piecewiseground_mask_fp32(
            current_zone, src_sorted, current_sensor_height, adaptive_seed_selection_margin, num_lpr, num_iter,
            th_seeds_v, th_dist_v, th_seeds, th_dist, uprightness_thr, enable_RVPF
        )
        
        ground_uprightness = normal[2]
        ground_elevation = pc_mean[2]
        ground_flatness = S[2]
        
        line_variable = np.inf
        if S[1] != 0:
            line_variable = S[0] / S[1]
            
        heading = np.sum(pc_mean * normal)
        
        is_upright = ground_uprightness > uprightness_thr
        is_near_zone = concentric_idx < num_rings_of_interest
        is_heading_outside = heading < 0.0
        
        is_not_elevated = False
        is_flat = False
        
        if concentric_idx < num_rings_of_interest:
            is_not_elevated = ground_elevation < elevation_thr_arr[concentric_idx]
            is_flat = ground_flatness < flatness_thr_arr[concentric_idx]
            
        # Update State (Ring Buffers)
        if is_upright and is_not_elevated and is_near_zone:
            head = elevation_heads[concentric_idx]
            cnt = elevation_counts[concentric_idx]
            update_elevation_arrs[concentric_idx][head] = ground_elevation
            elevation_heads[concentric_idx] = (head + 1) % max_elevation_storage
            if cnt < max_elevation_storage:
                elevation_counts[concentric_idx] += 1
                
            head_f = flatness_heads[concentric_idx]
            cnt_f = flatness_counts[concentric_idx]
            update_flatness_arrs[concentric_idx][head_f] = ground_flatness
            flatness_heads[concentric_idx] = (head_f + 1) % max_flatness_storage
            if cnt_f < max_flatness_storage:
                flatness_counts[concentric_idx] += 1
            
            ring_flatness_arr[ring_flatness_cnt] = ground_flatness
            ring_flatness_cnt += 1
            
        # Apply Semantic Labels
        if not is_upright:
            for k in range(count_in_bin):
                semantic[int(src_sorted[k, 3])] = 1
        elif not is_near_zone:
            for k in range(count_in_bin):
                semantic[int(src_sorted[k, 3])] = 0
        elif not is_heading_outside:
            for k in range(count_in_bin):
                semantic[int(src_sorted[k, 3])] = 1
        elif is_not_elevated or is_flat:
            for k in range(count_in_bin):
                if is_ground_mask[k]:
                    semantic[int(src_sorted[k, 3])] = 0
                else:
                    semantic[int(src_sorted[k, 3])] = 1
        else:
            # Candidate for TGR
            # Store metadata
            cand_flatness[cand_cnt] = ground_flatness
            cand_line_var[cand_cnt] = line_variable
            
            # Store indices of ground points
            cand_points_start[cand_cnt] = cand_buf_ptr
            for k in range(count_in_bin):
                if is_ground_mask[k]:
                    if cand_buf_ptr < n_pts:
                        cand_points_idx_buf[cand_buf_ptr] = int(src_sorted[k, 3])
                        cand_buf_ptr += 1
            cand_points_end[cand_cnt] = cand_buf_ptr
            cand_cnt += 1
            
        # Apply non-ground labels
        for k in range(count_in_bin):
            if is_nonground_mask[k]:
                semantic[int(src_sorted[k, 3])] = 1
        
        concentric_idx += 1

    # Process TGR for the very last ring
    if enable_TGR and cand_cnt > 0:
        temporal_ground_revert_core_fp32(
            ring_flatness_arr, ring_flatness_cnt,
            cand_flatness, cand_line_var,
            cand_points_idx_buf, cand_points_start, cand_points_end,
            cand_cnt, th_dist, semantic
        )
            
    update_thresholds_core_fp32(update_elevation_arrs, update_flatness_arrs, 
                                elevation_counts, flatness_counts,
                                elevation_heads, flatness_heads,
                                elevation_thr_arr, flatness_thr_arr, 
                                sensor_height_arr, max_elevation_storage, max_flatness_storage)
            
    return semantic

# --------------------------------------------------------------------------------
# Python Wrapper Class (Stateful - FP32)
# --------------------------------------------------------------------------------

class PatchWorkpp_fp32_masks:
    def __init__(self, params=None):
        if params is None:
            params = {}
            
        self.verbose = params.get('verbose', False)
        self.enable_RVPF = params.get('enable_RVPF', True)
        self.enable_TGR = params.get('enable_TGR', True)
        self.num_iter = params.get('num_iter', 3)
        self.num_lpr = params.get('num_lpr', 20)
        self.num_min_pts = params.get('num_min_pts', 10)
        self.num_zones = params.get('num_zones', 4)
        self.num_rings_of_interest = params.get('num_rings_of_interest', 4)
        
        self.initial_sensor_height = params.get('sensor_height', 1.723)
        self.th_seeds = params.get('th_seeds', 0.125)
        self.th_dist = params.get('th_dist', 0.125)
        self.th_seeds_v = params.get('th_seeds_v', 0.25)
        self.th_dist_v = params.get('th_dist_v', 0.1)
        self.max_range = params.get('max_range', 80.0)
        self.min_range = params.get('min_range', 2.7)
        self.uprightness_thr = params.get('uprightness_thr', 0.707)
        self.adaptive_seed_selection_margin = params.get('adaptive_seed_selection_margin', -1.2)
        
        self.num_sectors_each_zone = np.array(params.get('num_sectors_each_zone', [16, 32, 54, 32]), dtype=np.int64)
        self.num_rings_each_zone = np.array(params.get('num_rings_each_zone', [2, 4, 4, 4]), dtype=np.int64)
        
        self.max_flatness_storage = params.get('max_flatness_storage', 1000)
        self.max_elevation_storage = params.get('max_elevation_storage', 1000)
        
        # STATE INITIALIZATION (Fixed Size Arrays)
        self.elevation_thr = np.array(params.get('elevation_thr', [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.flatness_thr = np.array(params.get('flatness_thr', [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        self.sensor_height_ = np.array([self.initial_sensor_height], dtype=np.float32)
        
        # Ring Buffers
        self.update_elevation_arrs = np.zeros((4, self.max_elevation_storage), dtype=np.float32)
        self.update_flatness_arrs = np.zeros((4, self.max_flatness_storage), dtype=np.float32)
        
        self.elevation_counts = np.zeros(4, dtype=np.int64)
        self.flatness_counts = np.zeros(4, dtype=np.int64)
        
        self.elevation_heads = np.zeros(4, dtype=np.int64)
        self.flatness_heads = np.zeros(4, dtype=np.int64)
        
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
        pts = cloud_in.astype(np.float32)
        
        if pts.shape[1] == 4:
            temp_pts = np.zeros((pts.shape[0], 3), dtype=np.float32)
            temp_pts[:, :3] = pts[:, :3]
            pts = temp_pts
        elif pts.shape[1] == 3:
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
            self.update_elevation_arrs, self.update_flatness_arrs, 
            self.elevation_counts, self.flatness_counts,
            self.elevation_heads, self.flatness_heads,
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

def GroundFilterForward_fp32_masks(pts, **kwargs):
    gf = PatchWorkpp_fp32_masks(kwargs)
    return gf.forward(pts, verbose=kwargs.get('verbose', False))