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
# from cuml.cluster import DBSCAN
# import cudf
# import torch
# import cupy
import patchworkpp
import torch
from convexhullpy import get_convex_hull
from tqdm import tqdm

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

def points_in_boxes(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    Определяет, находятся ли точки внутри 3D боксов.
    
    Args:
        points: (N, 3) тензор координат точек (x, y, z).
        boxes: (M, 7) тензор параметров боксов (x, y, z, dx, dy, dz, yaw).
               x, y, z - центр бокса.
               dx, dy, dz - размеры бокса.
               yaw - угол поворота вокруг оси Z (в радианах).
               
    Returns:
        mask: (N, M) булев тензор, где True означает, что точка i находится внутри бокса j.
    """
    # Проверка на пустые входы для избежания ошибок размерности
    if points.shape[0] == 0 or boxes.shape[0] == 0:
        return torch.zeros((points.shape[0], boxes.shape[0]), 
                           dtype=torch.bool, device=points.device)

    # 1. Вычисляем смещение точек относительно центров боксов
    # points: (N, 1, 3), boxes[:, :3]: (1, M, 3) -> offsets: (N, M, 3)
    offsets = points.unsqueeze(1) - boxes[:, :3].unsqueeze(0)
    
    # 2. Поворачиваем смещения в локальную систему координат бокса
    # Используем матрицу поворота R_yaw^T (обратный поворот)
    # cos(-yaw) = cos(yaw), sin(-yaw) = -sin(yaw)
    yaws = boxes[:, 6] # (M,)
    cos_yaw = torch.cos(yaws).unsqueeze(0) # (1, M)
    sin_yaw = torch.sin(yaws).unsqueeze(0) # (1, M)
    
    # offsets[..., 0] это dx, offsets[..., 1] это dy
    # Локальные координаты:
    # local_x =  dx * cos(yaw) + dy * sin(yaw)
    # local_y = -dx * sin(yaw) + dy * cos(yaw)
    # local_z = dz (без изменений, так как yaw вращает вокруг Z)
    
    local_x = offsets[..., 0] * cos_yaw + offsets[..., 1] * sin_yaw
    local_y = -offsets[..., 0] * sin_yaw + offsets[..., 1] * cos_yaw
    local_z = offsets[..., 2]
    
    # 3. Проверяем попадание в границы половинных размеров
    # half_dims: (1, M, 3)
    half_dims = boxes[:, 3:6].unsqueeze(0) / 2.0
    
    # Сравниваем модуль локальных координат с половиной размера бокса
    in_x = local_x.abs() <= half_dims[:, :, 0]
    in_y = local_y.abs() <= half_dims[:, :, 1]
    in_z = local_z.abs() <= half_dims[:, :, 2]
    
    # Точка внутри, если она внутри по всем трем осям
    mask = in_x & in_y & in_z
    
    return mask

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
                pts_torch = torch.from_numpy(pts_ego).to(torch.float16).contiguous().to('cuda')
                boxes_torch = torch.from_numpy(boxes_3d_ego).to(torch.float16).contiguous().to('cuda')
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                mask_PB = points_in_boxes(pts_torch,boxes_torch)
                torch.cuda.synchronize()
                t2 = time.perf_counter_ns()
                print(f'points in boxes call {(t2-t1)/1e6:.2f} ms')
                is_inside_at_least_one_box = torch.any(mask_PB,dim=1)
                labels_np = is_inside_at_least_one_box.cpu().detach().numpy()
                pts_colors = colorize_point_cloud(pts_ego,labels_np,target_label=1)

                is_box_have_at_least_N_points = (torch.sum(mask_PB,dim=0)>1).cpu().detach().numpy().astype(np.bool_) 
                mask_PB_cpu = mask_PB.cpu().detach().numpy().astype(np.bool_)
                
                hulls_data = [] # Список кортежей (vertices, faces)
                t1 = time.perf_counter_ns()
                for box_idx in range(mask_PB_cpu.shape[1]):
                    if not is_box_have_at_least_N_points[box_idx]:
                        continue
                    points_in_this_box_mask =  mask_PB_cpu[:,box_idx]
                    points_in_this_box = pts_ego[points_in_this_box_mask]
                    
                    # Функция возвращает (vertices, faces), а не индексы
                    hull_vertices, hull_faces = get_convex_hull(points_in_this_box.astype(np.float32))
                    
                    if hull_vertices.shape[0] > 0:
                        hulls_data.append((hull_vertices, hull_faces))
                        
                t2 = time.perf_counter_ns()
                print(f'cpu hulls {(t2-t1)/1e6:.2f} ms')


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
                        },
                        {
                            'plot_type':'hulls',
                            'data': hulls_data,
                            'color': (0.0, 1.0, 0.0, 0.3) # Зеленый, полупрозрачный
                        }
                    ]
                )
                vis.process_events()
