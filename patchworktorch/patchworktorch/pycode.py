import torch
import math
import numpy as np
import patchworktorch_backend as _C

class patchworktorch_class:
    def __init__(self, params=None, device='cuda'):
        if params is None: params = {}
            
        self.verbose = params.get('verbose', False)
        self.device = device
        
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
        
        self.max_flatness_storage = params.get('max_flatness_storage', 1000)
        self.max_elevation_storage = params.get('max_elevation_storage', 1000)
        
        num_sectors_each_zone = np.array(params.get('num_sectors_each_zone', [16, 32, 54, 32]), dtype=np.int64)
        num_rings_each_zone = np.array(params.get('num_rings_each_zone', [2, 4, 4, 4]), dtype=np.int64)
        
        self.zone_bin_counts = num_rings_each_zone * num_sectors_each_zone
        zone_bin_starts = np.zeros(4, dtype=np.int64)
        zone_bin_starts[1] = self.zone_bin_counts[0]
        zone_bin_starts[2] = zone_bin_starts[1] + self.zone_bin_counts[1]
        zone_bin_starts[3] = zone_bin_starts[2] + self.zone_bin_counts[2]
        self.total_bins = int(np.sum(self.zone_bin_counts))
        
        min_range_z2 = (7 * self.min_range + self.max_range) / 8.0
        min_range_z3 = (3 * self.min_range + self.max_range) / 4.0
        min_range_z4 = (self.min_range + self.max_range) / 2.0
        min_ranges = np.array([self.min_range, min_range_z2, min_range_z3, min_range_z4], dtype=np.float32)
        
        ring_sizes = np.array([
            (min_range_z2 - self.min_range) / num_rings_each_zone[0],
            (min_range_z3 - min_range_z2) / num_rings_each_zone[1],
            (min_range_z4 - min_range_z3) / num_rings_each_zone[2],
            (self.max_range - min_range_z4) / num_rings_each_zone[3]
        ], dtype=np.float32)
        
        sector_sizes = np.array([
            2 * math.pi / num_sectors_each_zone[0],
            2 * math.pi / num_sectors_each_zone[1],
            2 * math.pi / num_sectors_each_zone[2],
            2 * math.pi / num_sectors_each_zone[3]
        ], dtype=np.float32)
        
        # Instantiate tensors on device
        self.num_rings_each_zone = torch.from_numpy(num_rings_each_zone).to(device)
        self.num_sectors_each_zone = torch.from_numpy(num_sectors_each_zone).to(device)
        self.zone_bin_starts = torch.from_numpy(zone_bin_starts).to(device)
        self.min_ranges = torch.from_numpy(min_ranges).to(device)
        self.ring_sizes = torch.from_numpy(ring_sizes).to(device)
        self.sector_sizes = torch.from_numpy(sector_sizes).to(device)
        
        # State Tensors
        self.update_elevation_arrs = torch.zeros(4, self.max_elevation_storage, device=device, dtype=torch.float32)
        self.update_flatness_arrs = torch.zeros(4, self.max_flatness_storage, device=device, dtype=torch.float32)
        self.elevation_counts = torch.zeros(4, device=device, dtype=torch.int32)
        self.flatness_counts = torch.zeros(4, device=device, dtype=torch.int32)
        self.elevation_heads = torch.zeros(4, device=device, dtype=torch.int32)
        self.flatness_heads = torch.zeros(4, device=device, dtype=torch.int32)
        self.elevation_thr_arr = torch.zeros(4, device=device, dtype=torch.float32)
        self.flatness_thr_arr = torch.zeros(4, device=device, dtype=torch.float32)
        self.sensor_height_arr = torch.tensor([self.initial_sensor_height], device=device, dtype=torch.float32)

    @property
    def sensor_height(self):
        return self.sensor_height_arr.item()

    def forward(self, cloud_in):
        # cloud_in must be (N, 3) float32 tensor on correct device
        if not cloud_in.is_cuda:
            cloud_in = cloud_in.to(self.device)
        
        # Call C++ extension
        results = _C.forward(
            cloud_in,
            self.update_elevation_arrs, self.update_flatness_arrs,
            self.elevation_counts, self.flatness_counts,
            self.elevation_heads, self.flatness_heads,
            self.elevation_thr_arr, self.flatness_thr_arr,
            self.sensor_height_arr,
            self.num_rings_each_zone, self.num_sectors_each_zone,
            self.min_ranges, self.ring_sizes, self.sector_sizes,
            self.zone_bin_starts,
            self.num_iter, self.num_lpr, self.num_min_pts, self.num_rings_of_interest,
            self.th_seeds, self.th_dist, self.uprightness_thr, self.adaptive_seed_selection_margin,
            self.max_flatness_storage, self.max_elevation_storage,
            self.min_range, self.max_range
        )
        
        semantic = results[0]
        
        if self.verbose:
            print(f"PatchWork++ Torch State Update sensor_height: {self.sensor_height:.4f}")
            print(f"PatchWork++ Torch State Update elevation_thr: {self.elevation_thr_arr.cpu().numpy()}")
            print(f"PatchWork++ Torch State Update flatness_thr:  {self.flatness_thr_arr.cpu().numpy()}")
            
        return semantic