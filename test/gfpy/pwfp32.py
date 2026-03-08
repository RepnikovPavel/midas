import os
import glob
import numpy as np
import cv2
import transforms3d as t3d
from tqdm import tqdm
from visualizer import LidarVisualizer
import bisect
from pandaset import PandaDatasetConverted,Sweep,Snapshot
import time
from cuml.cluster import DBSCAN
import cudf
import torch
import cupy
import patchworkpp

def colorize_point_cloud(
    pts, 
    labels,
    target_label=0,
    target_color=[255, 0, 0],
    ):
    """
    Раскрашивает облако точек:
    - label == 0: Красный цвет
    - label != 0: Карта 'jet' по оси Z
    
    Args:
        pts (np.ndarray): Массив точек (N, 3).
        labels (np.ndarray or cupy.ndarray): Массив меток (N,).
        
    Returns:
        np.ndarray: Массив цветов (N, 3) dtype=uint8 (RGB).
    """
    # Если labels находятся на GPU (cupy), переносим их в CPU (numpy)
        
    n_points = pts.shape[0]
    colors = np.zeros((n_points, 3), dtype=np.uint8)
    
    # --- 1. Обработка label == 0 (Красный цвет) ---
    mask_label_0 = (labels == target_label)
    colors[mask_label_0] = target_color # R, G, B
    
    # --- 2. Обработка label != 0 (Jet по высоте Z) ---
    mask_other = ~mask_label_0
    colors[mask_other] = (127,127,127)
    
    return colors


if __name__ == "__main__":
    vis = LidarVisualizer(title="PandaSet 3D LiDAR")
    
    DATA_ROOT = '/mnt/nvme/datasets/pandaset_converted'
    
    dataset = PandaDatasetConverted(DATA_ROOT, preindex_all_sweep_files=True)



    if len(dataset) > 0:
        for sweep_idx in range(len(dataset)):
            sweep = dataset.get_sweep(sweep_idx)
            for snapshot_idx in tqdm(range(len(sweep))):
                snapshot:Snapshot = sweep[snapshot_idx]
                pts_ego = snapshot.lidar['points'] # N,3 xyz
                print(f"pts shape {pts_ego.shape}")
                boxes_3d_ego = snapshot.boxes['boxes']
                vis.process_events()
                t1 = time.perf_counter_ns()
                labels = patchworkpp.GroundFilterForward_fp32(
                    # pts=np.concatenate((pts_ego,np.ones(shape=(len(pts_ego),1),dtype=np.float32)),axis=1), #: np.ndarray,
                    pts=pts_ego, #: np.ndarray,
                    verbose=False, #: bool = False,
                    enable_RVPF=True,#: bool = True,
                    enable_TGR=True,#: bool = True,
                    sensor_height=-100.0,#: float = 1.723,
                    # num_iter: int = 3,
                    # num_lpr: int = 20,
                    # num_min_pts: int = 10,
                    # num_zones: int = 4,
                    # num_rings_of_interest: int = 4,
                    # RNR_ver_angle_thr: float = -15.0,
                    # RNR_intensity_thr: float = 0.2,
                    # th_seeds: float = 0.125,
                    # th_dist: float = 0.125,
                    # th_seeds_v: float = 0.25,
                    # th_dist_v: float = 0.1,
                    max_range=80.0,#: float = 80.0,
                    min_range=2.7,#: float = 2.7,
                    # uprightness_thr: float = 0.707,
                    adaptive_seed_selection_margin=-1000.0#: float = -1.2,
                    # num_sectors_each_zone: np.ndarray = None,
                    # num_rings_each_zone: np.ndarray = None,
                    # elevation_thr: np.ndarray = None,
                    # flatness_thr: np.ndarray = None,
                    # max_flatness_storage: int = 1000,
                    # max_elevation_storage: int = 1000
                )
                t2 = time.perf_counter_ns()
                print(f"gfpy forward {(t2-t1)/1e6:.2f} ms")
                print('labels',np.unique(labels,return_counts=True))
                vis.process_events()

                pts_colors = colorize_point_cloud(pts_ego,labels)
                # if label==0 make red color for this points
                # else: just jet along z

                """
                {'lat': 37.7747430157756, 'long': -122.40097178666713, 'height': 3.0745996995937364, 'speed': 9.005294706004106}
                """
                print(snapshot.gps) 
                vis.process_events()

                vis.update(
                    [
                        {
                            'plot_type':'points',
                            'data':pts_ego,
                            'colors': pts_colors
                        },
                        {
                            'plot_type':'boxes',
                            'data':boxes_3d_ego
                        }
                    ]
                )
                vis.process_events()
