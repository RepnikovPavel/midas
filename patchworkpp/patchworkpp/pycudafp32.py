import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import math

# --------------------------------------------------------------------------------
# CUDA C++ Code
# --------------------------------------------------------------------------------

cuda_kernel_code = """
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846f
// Max points allowed in Shared Memory for sorting/fitting per bin
#define SMEM_CAP 4096 

__device__ float xy2theta_device(float x, float y) {
    float angle = atan2f(y, x);
    if (angle > 0) return angle;
    else return 2 * PI + angle;
}

__device__ void eigen_3x3_symmetric_device(float* A, float* V, float* S) {
    float D[9];
    for(int i=0; i<9; ++i) D[i] = A[i];
    
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            V[i*3+j] = (i == j) ? 1.0f : 0.0f;

    for(int iter=0; iter<10; ++iter) {
        int p=0, q=1;
        float max_val = fabsf(D[1]);
        if (fabsf(D[2]) > max_val) { p=0; q=2; max_val = fabsf(D[2]); }
        if (fabsf(D[5]) > max_val) { p=1; q=2; max_val = fabsf(D[5]); }
        
        if (max_val < 1e-7f) break;
        
        float diff = D[q*3+q] - D[p*3+p];
        float theta;
        if (fabsf(diff) < 1e-10f) {
            theta = PI / 4.0f;
            if (D[p*3+q] < 0) theta = -theta;
        } else {
            theta = 0.5f * atan2f(2.0f * D[p*3+q], diff);
        }
        
        float c = cosf(theta);
        float s = sinf(theta);
        
        for(int i=0; i<3; ++i) {
            float dip = D[i*3+p];
            float diq = D[i*3+q];
            D[i*3+p] = c * dip - s * diq;
            D[i*3+q] = s * dip + c * diq;
        }
        for(int i=0; i<3; ++i) {
            float dpi = D[p*3+i];
            float dqi = D[q*3+i];
            D[p*3+i] = c * dpi - s * dqi;
            D[q*3+i] = s * dpi + c * dqi;
        }
        for(int i=0; i<3; ++i) {
            float vip = V[i*3+p];
            float viq = V[i*3+q];
            V[i*3+p] = c * vip - s * viq;
            V[i*3+q] = s * vip + c * viq;
        }
    }
    
    S[0] = D[0]; S[1] = D[4]; S[2] = D[8];
    
    if (S[0] < S[1]) { float t=S[0]; S[0]=S[1]; S[1]=t; for(int i=0;i<3;i++){float tmp=V[i*3+0]; V[i*3+0]=V[i*3+1]; V[i*3+1]=tmp;} }
    if (S[0] < S[2]) { float t=S[0]; S[0]=S[2]; S[2]=t; for(int i=0;i<3;i++){float tmp=V[i*3+0]; V[i*3+0]=V[i*3+2]; V[i*3+2]=tmp;} }
    if (S[1] < S[2]) { float t=S[1]; S[1]=S[2]; S[2]=t; for(int i=0;i<3;i++){float tmp=V[i*3+1]; V[i*3+1]=V[i*3+2]; V[i*3+2]=tmp;} }
}

__device__ void calc_mean_stdev_circular_device(const float* arr, int head, int count, int max_size, float& mean, float& stdev) {
    if (count <= 1) { mean = 0.0f; stdev = 0.0f; return; }
    mean = 0.0f;
    int start_pos = (head - count + max_size) % max_size;
    for (int i = 0; i < count; ++i) {
        mean += arr[(start_pos + i) % max_size];
    }
    mean /= count;
    
    stdev = 0.0f;
    for (int i = 0; i < count; ++i) {
        float diff = arr[(start_pos + i) % max_size] - mean;
        stdev += diff * diff;
    }
    stdev = sqrtf(stdev / (count - 1));
}

__global__ void kernel_assign_bins(const float* pts, long long* bin_ids, int n_pts, 
                                   float min_range, float max_range, int num_zones,
                                   const long long* num_rings_each_zone, const long long* num_sectors_each_zone,
                                   const float* min_ranges, const float* ring_sizes, const float* sector_sizes,
                                   const long long* zone_bin_starts) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pts) return;
    
    float x = pts[idx * 3 + 0];
    float y = pts[idx * 3 + 1];
    float r = sqrtf(x*x + y*y);
    
    if (!(r <= max_range && r > min_range)) { bin_ids[idx] = -1; return; }
    
    float theta = xy2theta_device(x, y);
    
    int zone_idx = -1;
    int ring_idx = -1;
    int sector_idx = -1;
    
    if (r < min_ranges[1]) { zone_idx = 0; ring_idx = (int)((r - min_ranges[0]) / ring_sizes[0]); sector_idx = (int)(theta / sector_sizes[0]); }
    else if (r < min_ranges[2]) { zone_idx = 1; ring_idx = (int)((r - min_ranges[1]) / ring_sizes[1]); sector_idx = (int)(theta / sector_sizes[1]); }
    else if (r < min_ranges[3]) { zone_idx = 2; ring_idx = (int)((r - min_ranges[2]) / ring_sizes[2]); sector_idx = (int)(theta / sector_sizes[2]); }
    else { zone_idx = 3; ring_idx = (int)((r - min_ranges[3]) / ring_sizes[3]); sector_idx = (int)(theta / sector_sizes[3]); }
    
    if (ring_idx >= num_rings_each_zone[zone_idx]) ring_idx = num_rings_each_zone[zone_idx] - 1;
    if (sector_idx >= num_sectors_each_zone[zone_idx]) sector_idx = num_sectors_each_zone[zone_idx] - 1;
    if (ring_idx < 0) ring_idx = 0;
    if (sector_idx < 0) sector_idx = 0;
    
    bin_ids[idx] = zone_bin_starts[zone_idx] + ring_idx * num_sectors_each_zone[zone_idx] + sector_idx;
}

__global__ void kernel_count_bins(const long long* bin_ids, int* bin_counts, int n_pts, int total_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pts) return;
    long long bid = bin_ids[idx];
    if (bid >= 0 && bid < total_bins) atomicAdd(&bin_counts[bid], 1);
}

__global__ void kernel_scatter_indices(const long long* bin_ids, int* bin_write_heads, long long* sorted_indices, int n_pts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pts) return;
    long long bid = bin_ids[idx];
    if (bid >= 0) {
        int pos = atomicAdd(&bin_write_heads[bid], 1);
        sorted_indices[pos] = idx;
    }
}

__global__ void kernel_ground_filter_stateful(
    const float* pts, const long long* sorted_indices, const int* bin_counts_scan, 
    const int* valid_bin_ids_list, int* semantic,
    float* update_elevation_arrs, float* update_flatness_arrs, 
    int* elevation_counts, int* flatness_counts,
    int* elevation_heads, int* flatness_heads,
    float* elevation_thr_arr, float* flatness_thr_arr, float* sensor_height_arr,
    int num_iter, int num_lpr, int num_min_pts, int num_rings_of_interest,
    float th_seeds, float th_dist, float uprightness_thr, float adaptive_seed_selection_margin,
    int max_flatness_storage, int max_elevation_storage,
    const long long* zone_bin_starts) 
{
    int logical_bid = blockIdx.x; 
    int bid = valid_bin_ids_list[logical_bid];
    int concentric_idx = logical_bid;
    
    int tid = threadIdx.x;
    
    int start_idx = bin_counts_scan[bid];
    int end_idx = bin_counts_scan[bid+1];
    int count_in_bin = end_idx - start_idx;
    
    int zone_idx = 0;
    if (bid >= zone_bin_starts[3]) zone_idx = 3;
    else if (bid >= zone_bin_starts[2]) zone_idx = 2;
    else if (bid >= zone_bin_starts[1]) zone_idx = 1;
    
    extern __shared__ char smem_raw[];
    float* s_z = (float*)smem_raw;
    int* s_orig_idx = (int*)&s_z[SMEM_CAP];
    
    int process_count = min(count_in_bin, SMEM_CAP);
    
    // Load & Sort only the first process_count points (lowest Z assumption holds after sort)
    for (int k = tid; k < process_count; k += blockDim.x) {
        long long pt_idx = sorted_indices[start_idx + k];
        s_z[k] = pts[pt_idx * 3 + 2];
        s_orig_idx[k] = (int)pt_idx;
    }
    __syncthreads();
    
    // Sort by Z
    for (int i = 0; i < process_count; i++) {
        int start = i % 2;
        for (int j = start + tid * 2; j < process_count - 1; j += blockDim.x * 2) {
            if (s_z[j] > s_z[j+1]) {
                float tz = s_z[j]; s_z[j] = s_z[j+1]; s_z[j+1] = tz;
                int ti = s_orig_idx[j]; s_orig_idx[j] = s_orig_idx[j+1]; s_orig_idx[j+1] = ti;
            }
        }
        __syncthreads();
    }
    
    float sensor_height = sensor_height_arr[0];
    
    int init_idx = 0;
    if (zone_idx == 0) {
        float threshold = adaptive_seed_selection_margin * sensor_height;
        for (int k = 0; k < process_count; ++k) {
            if (s_z[k] < threshold) init_idx++;
            else break;
        }
    }
    
    float sum_val = 0.0f;
    int lpr_cnt = 0;
    for (int k = init_idx; k < process_count; ++k) {
        sum_val += s_z[k];
        lpr_cnt++;
        if (lpr_cnt >= num_lpr) break;
    }
    
    float lpr_h = (lpr_cnt > 0) ? sum_val / lpr_cnt : 0.0f;
    
    float normal[3] = {0,0,1};
    float d = 0;
    float S[3] = {0,0,0};
    float pc_mean[3] = {0,0,0};
    
    for (int iter = 0; iter < num_iter; ++iter) {
        float mean[3] = {0,0,0};
        int g_count = 0;
        
        for (int k = 0; k < process_count; ++k) {
            bool is_g = false;
            if (iter == 0) {
                if (s_z[k] < lpr_h + th_seeds) is_g = true;
            } else {
                int pt_idx = s_orig_idx[k];
                float dist = normal[0]*pts[pt_idx*3+0] + normal[1]*pts[pt_idx*3+1] + normal[2]*pts[pt_idx*3+2] + d;
                if (dist < th_dist) is_g = true;
            }
            
            if (is_g) {
                int pt_idx = s_orig_idx[k];
                mean[0] += pts[pt_idx*3+0];
                mean[1] += pts[pt_idx*3+1];
                mean[2] += pts[pt_idx*3+2];
                g_count++;
            }
        }
        
        if (g_count < 3) break;
        
        mean[0] /= g_count; mean[1] /= g_count; mean[2] /= g_count;
        
        float cov[9] = {0,0,0, 0,0,0, 0,0,0};
        for (int k = 0; k < process_count; ++k) {
            bool is_g = false;
            if (iter == 0) {
                if (s_z[k] < lpr_h + th_seeds) is_g = true;
            } else {
                int pt_idx = s_orig_idx[k];
                float dist = normal[0]*pts[pt_idx*3+0] + normal[1]*pts[pt_idx*3+1] + normal[2]*pts[pt_idx*3+2] + d;
                if (dist < th_dist) is_g = true;
            }
            
            if (is_g) {
                int pt_idx = s_orig_idx[k];
                float p[3] = {pts[pt_idx*3+0] - mean[0], pts[pt_idx*3+1] - mean[1], pts[pt_idx*3+2] - mean[2]};
                cov[0] += p[0]*p[0]; cov[1] += p[0]*p[1]; cov[2] += p[0]*p[2];
                cov[4] += p[1]*p[1]; cov[5] += p[1]*p[2];
                cov[8] += p[2]*p[2];
            }
        }
        cov[3] = cov[1]; cov[6] = cov[2]; cov[7] = cov[5];
        
        if (g_count > 1) {
            float div = 1.0f / (g_count - 1);
            for(int i=0; i<9; ++i) cov[i] *= div;
        }
        
        float V[9];
        eigen_3x3_symmetric_device(cov, V, S);
        
        normal[0] = V[6]; normal[1] = V[7]; normal[2] = V[8];
        if (normal[2] < 0) {
            normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2];
        }
        d = -(normal[0]*mean[0] + normal[1]*mean[1] + normal[2]*mean[2]);
        
        pc_mean[0] = mean[0]; pc_mean[1] = mean[1]; pc_mean[2] = mean[2];
    }
    
    float ground_uprightness = normal[2];
    float ground_elevation = pc_mean[2];
    float ground_flatness = S[2];
    
    float heading = pc_mean[0]*normal[0] + pc_mean[1]*normal[1] + pc_mean[2]*normal[2];
    
    bool is_upright = ground_uprightness > uprightness_thr;
    bool is_near_zone = concentric_idx < num_rings_of_interest;
    bool is_heading_outside = heading < 0.0f;
    
    bool is_not_elevated = false;
    bool is_flat = false;
    
    if (concentric_idx < num_rings_of_interest) {
        is_not_elevated = ground_elevation < elevation_thr_arr[concentric_idx];
        is_flat = ground_flatness < flatness_thr_arr[concentric_idx];
        
        if (is_upright && is_not_elevated && is_near_zone) {
            int head = atomicAdd(&elevation_heads[concentric_idx], 1);
            update_elevation_arrs[concentric_idx * max_elevation_storage + (head % max_elevation_storage)] = ground_elevation;
            if (head < max_elevation_storage) atomicAdd(&elevation_counts[concentric_idx], 1);
            
            int head_f = atomicAdd(&flatness_heads[concentric_idx], 1);
            update_flatness_arrs[concentric_idx * max_flatness_storage + (head_f % max_flatness_storage)] = ground_flatness;
            if (head_f < max_flatness_storage) atomicAdd(&flatness_counts[concentric_idx], 1);
        }
    }
    
    // Final labeling - Iterate over ALL points in bin (count_in_bin), not just process_count
    for (int k = tid; k < count_in_bin; k += blockDim.x) {
        int pt_idx;
        float dist;
        
        // If within the sorted range, use shared memory (fast)
        if (k < process_count) {
            pt_idx = s_orig_idx[k];
            // X and Y are needed for dist, Z is in s_z. 
            // We still read from global pts to avoid complex register usage, 
            // but s_orig_idx is the important cached part.
        } else {
            // If outside sorted range, read from global memory
            pt_idx = sorted_indices[start_idx + k];
        }
        
        // Calculate distance
        dist = normal[0]*pts[pt_idx*3+0] + normal[1]*pts[pt_idx*3+1] + normal[2]*pts[pt_idx*3+2] + d;
        
        bool is_nonground_mask = (dist >= th_dist);
        int label = 1; 

        if (!is_upright) {
            label = 1;
        }
        else if (!is_near_zone) {
            label = is_nonground_mask ? 1 : 0;
        }
        else if (!is_heading_outside) {
            label = 1;
        }
        else if (is_not_elevated || is_flat) {
            label = is_nonground_mask ? 1 : 0;
        }
        else {
            label = 1;
        }
        
        // Apply Numba logic override
        if (is_nonground_mask) {
            label = 1;
        }
        
        semantic[pt_idx] = label;
    }
}

__global__ void kernel_update_thresholds(float* update_elevation_arrs, float* update_flatness_arrs, 
                                         int* elevation_counts, int* flatness_counts,
                                         int* elevation_heads, int* flatness_heads,
                                         float* elevation_thr_arr, float* flatness_thr_arr, 
                                         float* sensor_height_arr, int max_elevation_storage, int max_flatness_storage) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4) return;
    
    int cnt = elevation_counts[i];
    if (cnt > 1) {
        float mean, stdev;
        calc_mean_stdev_circular_device(&update_elevation_arrs[i*max_elevation_storage], elevation_heads[i], cnt, max_elevation_storage, mean, stdev);
        if (i == 0) {
            elevation_thr_arr[i] = mean + 3 * stdev;
            sensor_height_arr[0] = -mean;
        } else {
            elevation_thr_arr[i] = mean + 2 * stdev;
        }
    }
    
    cnt = flatness_counts[i];
    if (cnt > 1) {
        float mean, stdev;
        calc_mean_stdev_circular_device(&update_flatness_arrs[i*max_flatness_storage], flatness_heads[i], cnt, max_flatness_storage, mean, stdev);
        flatness_thr_arr[i] = mean + stdev;
    }
}
"""

mod = SourceModule(cuda_kernel_code)

kernel_assign_bins = mod.get_function("kernel_assign_bins")
kernel_count_bins = mod.get_function("kernel_count_bins")
kernel_scatter_indices = mod.get_function("kernel_scatter_indices")
kernel_ground_filter = mod.get_function("kernel_ground_filter_stateful")
kernel_update_thresholds = mod.get_function("kernel_update_thresholds")


class PatchWorkpp_fp32_light_pycuda:
    def __init__(self, params=None):
        if params is None: params = {}
            
        self.verbose = params.get('verbose', False)
        self.num_iter = params.get('num_iter', 3)
        self.num_lpr = params.get('num_lpr', 20)
        self.num_min_pts = params.get('num_min_pts', 10)
        self.num_zones = params.get('num_zones', 4)
        self.num_rings_of_interest = params.get('num_rings_of_interest', 4)
        
        self.initial_sensor_height = params.get('sensor_height', 1.723)
        self.th_seeds = params.get('th_seeds', 0.125)
        self.th_dist = params.get('th_dist', 0.125)
        self.max_range = params.get('max_range', 80.0)
        self.min_range = params.get('min_range', 2.7)
        self.uprightness_thr = params.get('uprightness_thr', 0.707)
        self.adaptive_seed_selection_margin = params.get('adaptive_seed_selection_margin', -1.2)
        
        self.num_sectors_each_zone = np.array(params.get('num_sectors_each_zone', [16, 32, 54, 32]), dtype=np.int64)
        self.num_rings_each_zone = np.array(params.get('num_rings_each_zone', [2, 4, 4, 4]), dtype=np.int64)
        
        self.max_flatness_storage = params.get('max_flatness_storage', 1000)
        self.max_elevation_storage = params.get('max_elevation_storage', 1000)
        
        self.zone_bin_counts = self.num_rings_each_zone * self.num_sectors_each_zone
        self.zone_bin_starts = np.zeros(4, dtype=np.int64)
        self.zone_bin_starts[1] = self.zone_bin_counts[0]
        self.zone_bin_starts[2] = self.zone_bin_starts[1] + self.zone_bin_counts[1]
        self.zone_bin_starts[3] = self.zone_bin_starts[2] + self.zone_bin_counts[2]
        self.total_bins = int(np.sum(self.zone_bin_counts))
        
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
        
        # GPU State
        self.d_update_elevation_arrs = cuda.mem_alloc(4 * self.max_elevation_storage * 4)
        self.d_update_flatness_arrs = cuda.mem_alloc(4 * self.max_flatness_storage * 4)
        self.d_elevation_counts = cuda.mem_alloc(4 * 4)
        self.d_flatness_counts = cuda.mem_alloc(4 * 4)
        self.d_elevation_heads = cuda.mem_alloc(4 * 4)
        self.d_flatness_heads = cuda.mem_alloc(4 * 4)
        self.d_elevation_thr = cuda.mem_alloc(4 * 4)
        self.d_flatness_thr = cuda.mem_alloc(4 * 4)
        self.d_sensor_height = cuda.mem_alloc(4)
        
        h_zero_int = np.zeros(4, dtype=np.int32)
        h_zero_float = np.zeros(4, dtype=np.float32)
        h_init_sh = np.array([self.initial_sensor_height], dtype=np.float32)
        
        cuda.memset_d8(self.d_update_elevation_arrs, 0, 4 * self.max_elevation_storage * 4)
        cuda.memset_d8(self.d_update_flatness_arrs, 0, 4 * self.max_flatness_storage * 4)
        cuda.memcpy_htod(self.d_elevation_counts, h_zero_int)
        cuda.memcpy_htod(self.d_flatness_counts, h_zero_int)
        cuda.memcpy_htod(self.d_elevation_heads, h_zero_int)
        cuda.memcpy_htod(self.d_flatness_heads, h_zero_int)
        cuda.memcpy_htod(self.d_elevation_thr, h_zero_float)
        cuda.memcpy_htod(self.d_flatness_thr, h_zero_float)
        cuda.memcpy_htod(self.d_sensor_height, h_init_sh)
        
        self.d_num_rings_each_zone = cuda.to_device(self.num_rings_each_zone)
        self.d_num_sectors_each_zone = cuda.to_device(self.num_sectors_each_zone)
        self.d_min_ranges = cuda.to_device(self.min_ranges)
        self.d_ring_sizes = cuda.to_device(self.ring_sizes)
        self.d_sector_sizes = cuda.to_device(self.sector_sizes)
        self.d_zone_bin_starts = cuda.to_device(self.zone_bin_starts)
        
        self.block_size = 256

    @property
    def sensor_height(self):
        h_sh = np.empty(1, dtype=np.float32)
        cuda.memcpy_dtoh(h_sh, self.d_sensor_height)
        return h_sh[0]

    def forward(self, cloud_in, verbose=False):
        pts = cloud_in.astype(np.float32)
        n_pts = pts.shape[0]
        
        d_pts = cuda.to_device(pts)
        d_bin_ids = cuda.mem_alloc(n_pts * 8) 
        d_sorted_indices = cuda.mem_alloc(n_pts * 8)
        
        grid = ((n_pts + self.block_size - 1) // self.block_size, 1, 1)
        block = (self.block_size, 1, 1)
        
        kernel_assign_bins(d_pts, d_bin_ids, np.int32(n_pts), 
                           np.float32(self.min_range), np.float32(self.max_range), np.int32(self.num_zones),
                           self.d_num_rings_each_zone, self.d_num_sectors_each_zone,
                           self.d_min_ranges, self.d_ring_sizes, self.d_sector_sizes,
                           self.d_zone_bin_starts,
                           grid=grid, block=block)
                           
        d_bin_counts = cuda.mem_alloc(self.total_bins * 4)
        cuda.memset_d8(d_bin_counts, 0, self.total_bins * 4)
        
        kernel_count_bins(d_bin_ids, d_bin_counts, np.int32(n_pts), np.int32(self.total_bins), 
                          grid=grid, block=block)
        
        h_bin_counts = np.empty(self.total_bins, dtype=np.int32)
        cuda.memcpy_dtoh(h_bin_counts, d_bin_counts)
        
        h_bin_scan = np.zeros(self.total_bins + 1, dtype=np.int32)
        cumsum = 0
        for i in range(self.total_bins):
            h_bin_scan[i] = cumsum
            cumsum += h_bin_counts[i]
        h_bin_scan[self.total_bins] = cumsum
        
        valid_bin_ids_list = []
        for i in range(self.total_bins):
            if h_bin_counts[i] >= self.num_min_pts:
                valid_bin_ids_list.append(i)
        num_valid_bins = len(valid_bin_ids_list)
        
        d_bin_scan = cuda.to_device(h_bin_scan)
        d_valid_bin_ids_list = cuda.to_device(np.array(valid_bin_ids_list, dtype=np.int32))
        d_write_heads = cuda.to_device(h_bin_scan[:-1])
        
        kernel_scatter_indices(d_bin_ids, d_write_heads, d_sorted_indices, np.int32(n_pts), 
                               grid=grid, block=block)
        
        d_semantic = cuda.mem_alloc(n_pts * 4)
        d_semantic_int = np.empty(n_pts, dtype=np.int32)
        d_semantic_int.fill(1)
        cuda.memcpy_htod(d_semantic, d_semantic_int)
        
        shared_mem_size = 4096 * 4 + 4096 * 4
        
        grid_bins = (num_valid_bins, 1, 1)
        
        if num_valid_bins > 0:
            kernel_ground_filter(d_pts, d_sorted_indices, d_bin_scan, d_valid_bin_ids_list, d_semantic,
                                 self.d_update_elevation_arrs, self.d_update_flatness_arrs,
                                 self.d_elevation_counts, self.d_flatness_counts,
                                 self.d_elevation_heads, self.d_flatness_heads,
                                 self.d_elevation_thr, self.d_flatness_thr, self.d_sensor_height,
                                 np.int32(self.num_iter), np.int32(self.num_lpr), np.int32(self.num_min_pts), np.int32(self.num_rings_of_interest),
                                 np.float32(self.th_seeds), np.float32(self.th_dist), np.float32(self.uprightness_thr), np.float32(self.adaptive_seed_selection_margin),
                                 np.int32(self.max_flatness_storage), np.int32(self.max_elevation_storage),
                                 self.d_zone_bin_starts,
                                 grid=grid_bins, block=block, shared=shared_mem_size)
        
        kernel_update_thresholds(self.d_update_elevation_arrs, self.d_update_flatness_arrs,
                                 self.d_elevation_counts, self.d_flatness_counts,
                                 self.d_elevation_heads, self.d_flatness_heads,
                                 self.d_elevation_thr, self.d_flatness_thr, self.d_sensor_height,
                                 np.int32(self.max_elevation_storage), np.int32(self.max_flatness_storage),
                                 grid=(1,1,1), block=(4,1,1))
                                 
        semantic = np.empty(n_pts, dtype=np.int32)
        cuda.memcpy_dtoh(semantic, d_semantic)
        
        if verbose:
            h_elev = np.empty(4, dtype=np.float32)
            h_flat = np.empty(4, dtype=np.float32)
            h_sh = np.empty(1, dtype=np.float32)
            cuda.memcpy_dtoh(h_elev, self.d_elevation_thr)
            cuda.memcpy_dtoh(h_flat, self.d_flatness_thr)
            cuda.memcpy_dtoh(h_sh, self.d_sensor_height)
            print(f"PatchWork++ Light (PyCUDA) State Update sensor_height: {h_sh[0]:.4f}")
            print(f"PatchWork++ Light (PyCUDA) State Update elevation_thr: {h_elev}")
            print(f"PatchWork++ Light (PyCUDA) State Update flatness_thr:  {h_flat}")
            
        return semantic

def GroundFilterForward_fp32_light_pycuda(pts, **kwargs):
    gf = PatchWorkpp_fp32_light_pycuda(kwargs)
    return gf.forward(pts, verbose=kwargs.get('verbose', False))
