#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>

#define PI 3.14159265358979323846f
#define SMEM_CAP 4096 

// --------------------------------------------------------------------------------
// Device Helpers
// --------------------------------------------------------------------------------

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

// --------------------------------------------------------------------------------
// Kernels
// --------------------------------------------------------------------------------

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
    const long long* zone_bin_starts,
    const long long* num_rings_each_zone,
    const long long* num_sectors_each_zone) 
{
    int logical_bid = blockIdx.x; 
    int bid = valid_bin_ids_list[logical_bid];
    
    int tid = threadIdx.x;
    
    int start_idx = bin_counts_scan[bid];
    int end_idx = bin_counts_scan[bid+1];
    int count_in_bin = end_idx - start_idx;
    
    int zone_idx = 0;
    if (bid >= zone_bin_starts[3]) zone_idx = 3;
    else if (bid >= zone_bin_starts[2]) zone_idx = 2;
    else if (bid >= zone_bin_starts[1]) zone_idx = 1;

    long long bid_in_zone = bid - zone_bin_starts[zone_idx];
    long long ring_in_zone = bid_in_zone / num_sectors_each_zone[zone_idx];
    
    long long rings_before = 0;
    for(int z=0; z<zone_idx; ++z) rings_before += num_rings_each_zone[z];
    
    int concentric_idx = (int)(rings_before + ring_in_zone);
    
    extern __shared__ char smem_raw[];
    float* s_z = (float*)smem_raw;
    int* s_orig_idx = (int*)&s_z[SMEM_CAP];
    
    int process_count = min(count_in_bin, SMEM_CAP);
    
    for (int k = tid; k < process_count; k += blockDim.x) {
        long long pt_idx = sorted_indices[start_idx + k];
        s_z[k] = pts[pt_idx * 3 + 2];
        s_orig_idx[k] = (int)pt_idx;
    }
    __syncthreads();
    
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
    
    for (int k = tid; k < count_in_bin; k += blockDim.x) {
        int pt_idx;
        float dist;
        
        if (k < process_count) {
            pt_idx = s_orig_idx[k];
        } else {
            pt_idx = sorted_indices[start_idx + k];
        }
        
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


// --------------------------------------------------------------------------------
// C++ Interface
// --------------------------------------------------------------------------------

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> ground_filter_forward(
    torch::Tensor pts,
    torch::Tensor update_elevation_arrs, torch::Tensor update_flatness_arrs,
    torch::Tensor elevation_counts, torch::Tensor flatness_counts,
    torch::Tensor elevation_heads, torch::Tensor flatness_heads,
    torch::Tensor elevation_thr_arr, torch::Tensor flatness_thr_arr,
    torch::Tensor sensor_height_arr,
    torch::Tensor num_rings_each_zone, torch::Tensor num_sectors_each_zone,
    torch::Tensor min_ranges, torch::Tensor ring_sizes, torch::Tensor sector_sizes,
    torch::Tensor zone_bin_starts,
    int num_iter, int num_lpr, int num_min_pts, int num_rings_of_interest,
    float th_seeds, float th_dist, float uprightness_thr, float adaptive_seed_selection_margin,
    int max_flatness_storage, int max_elevation_storage,
    float min_range, float max_range
) {
    CHECK_INPUT(pts);
    
    int n_pts = pts.size(0);
    
    // Calculate total_bins using raw pointer access (CPU side)
    auto num_rings_cpu = num_rings_each_zone.to(torch::kCPU);
    auto num_sectors_cpu = num_sectors_each_zone.to(torch::kCPU);
    
    const long long* rings_ptr = static_cast<const long long*>(num_rings_cpu.data_ptr());
    const long long* sectors_ptr = static_cast<const long long*>(num_sectors_cpu.data_ptr());
    
    long long total_bins_ll = 0;
    for(int i=0; i<4; ++i) total_bins_ll += rings_ptr[i] * sectors_ptr[i];
    int total_bins = (int)total_bins_ll;

    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(pts.device());
    auto options_int64 = torch::TensorOptions().dtype(torch::kInt64).device(pts.device());

    auto bin_ids = torch::empty({n_pts}, options_int64);
    auto bin_counts = torch::zeros({total_bins}, options_int32);
    
    int block_size = 256;
    int grid_size = (n_pts + block_size - 1) / block_size;

    // Kernel Call Corrected
    kernel_assign_bins<<<grid_size, block_size>>>(
        static_cast<const float*>(pts.data_ptr()), 
        static_cast<long long*>(bin_ids.data_ptr()), 
        n_pts,
        min_range, max_range, 4,
        static_cast<const long long*>(num_rings_each_zone.data_ptr()), 
        static_cast<const long long*>(num_sectors_each_zone.data_ptr()),
        static_cast<const float*>(min_ranges.data_ptr()), 
        static_cast<const float*>(ring_sizes.data_ptr()), 
        static_cast<const float*>(sector_sizes.data_ptr()),
        static_cast<const long long*>(zone_bin_starts.data_ptr())
    );

    kernel_count_bins<<<grid_size, block_size>>>(
        static_cast<const long long*>(bin_ids.data_ptr()), 
        static_cast<int*>(bin_counts.data_ptr()), 
        n_pts, total_bins
    );

    auto bin_counts_cpu = bin_counts.to(torch::kCPU);
    const int* counts_ptr = static_cast<const int*>(bin_counts_cpu.data_ptr());
    
    std::vector<int> h_bin_scan_vec(total_bins + 1, 0);
    std::vector<int> valid_bin_ids_vec;
    int cumsum = 0;
    for(int i=0; i<total_bins; ++i) {
        h_bin_scan_vec[i] = cumsum;
        int cnt = counts_ptr[i];
        cumsum += cnt;
        if(cnt >= num_min_pts) valid_bin_ids_vec.push_back(i);
    }
    h_bin_scan_vec[total_bins] = cumsum;

    int num_valid_bins = valid_bin_ids_vec.size();
    auto bin_scan = torch::from_blob(h_bin_scan_vec.data(), {total_bins + 1}, torch::kInt32).clone().to(pts.device());
    auto valid_bin_ids_tensor = torch::from_blob(valid_bin_ids_vec.data(), {num_valid_bins}, torch::kInt32).clone().to(pts.device());

    auto sorted_indices = torch::empty({n_pts}, options_int64);
    auto write_heads = bin_scan.clone();
    
    kernel_scatter_indices<<<grid_size, block_size>>>(
        static_cast<const long long*>(bin_ids.data_ptr()), 
        static_cast<int*>(write_heads.data_ptr()), 
        static_cast<long long*>(sorted_indices.data_ptr()), 
        n_pts
    );

    auto semantic = torch::ones({n_pts}, options_int32);
    
    size_t shared_mem_size = SMEM_CAP * 4 + SMEM_CAP * 4;

    if(num_valid_bins > 0) {
        kernel_ground_filter_stateful<<<num_valid_bins, block_size, shared_mem_size>>>(
            static_cast<const float*>(pts.data_ptr()), 
            static_cast<const long long*>(sorted_indices.data_ptr()), 
            static_cast<const int*>(bin_scan.data_ptr()),
            static_cast<const int*>(valid_bin_ids_tensor.data_ptr()), 
            static_cast<int*>(semantic.data_ptr()),
            static_cast<float*>(update_elevation_arrs.data_ptr()), 
            static_cast<float*>(update_flatness_arrs.data_ptr()),
            static_cast<int*>(elevation_counts.data_ptr()), 
            static_cast<int*>(flatness_counts.data_ptr()),
            static_cast<int*>(elevation_heads.data_ptr()), 
            static_cast<int*>(flatness_heads.data_ptr()),
            static_cast<float*>(elevation_thr_arr.data_ptr()), 
            static_cast<float*>(flatness_thr_arr.data_ptr()), 
            static_cast<float*>(sensor_height_arr.data_ptr()),
            num_iter, num_lpr, num_min_pts, num_rings_of_interest,
            th_seeds, th_dist, uprightness_thr, adaptive_seed_selection_margin,
            max_flatness_storage, max_elevation_storage,
            static_cast<const long long*>(zone_bin_starts.data_ptr()),
            static_cast<const long long*>(num_rings_each_zone.data_ptr()),
            static_cast<const long long*>(num_sectors_each_zone.data_ptr())
        );
    }

    kernel_update_thresholds<<<1, 4>>>(
        static_cast<float*>(update_elevation_arrs.data_ptr()), 
        static_cast<float*>(update_flatness_arrs.data_ptr()),
        static_cast<int*>(elevation_counts.data_ptr()), 
        static_cast<int*>(flatness_counts.data_ptr()),
        static_cast<int*>(elevation_heads.data_ptr()), 
        static_cast<int*>(flatness_heads.data_ptr()),
        static_cast<float*>(elevation_thr_arr.data_ptr()), 
        static_cast<float*>(flatness_thr_arr.data_ptr()),
        static_cast<float*>(sensor_height_arr.data_ptr()),
        max_elevation_storage, max_flatness_storage
    );

    return {semantic};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ground_filter_forward, "PatchWork Ground Filter forward (CUDA)");
}