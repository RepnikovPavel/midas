import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from visualizer import LidarVisualizer
from pandaset import PandaDatasetConverted, Sweep, Snapshot
import time
import torch
from scipy.spatial import ConvexHull

# --- Вспомогательные функции (оставляем без изменений) ---

def colorize_point_cloud(pts, labels, target_label=0, target_color=[255, 0, 0]):
    n_points = pts.shape[0]
    colors = np.zeros((n_points, 3), dtype=np.uint8)
    mask_label_0 = (labels == target_label)
    colors[mask_label_0] = target_color
    mask_other = ~mask_label_0
    colors[mask_other] = (127, 127, 127)
    return colors

def points_in_boxes(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if points.shape[0] == 0 or boxes.shape[0] == 0:
        return torch.zeros((points.shape[0], boxes.shape[0]), dtype=torch.bool, device=points.device)

    offsets = points.unsqueeze(1) - boxes[:, :3].unsqueeze(0)
    yaws = boxes[:, 6]
    cos_yaw = torch.cos(yaws).unsqueeze(0)
    sin_yaw = torch.sin(yaws).unsqueeze(0)
    
    local_x = offsets[..., 0] * cos_yaw + offsets[..., 1] * sin_yaw
    local_y = -offsets[..., 0] * sin_yaw + offsets[..., 1] * cos_yaw
    local_z = offsets[..., 2]
    
    half_dims = boxes[:, 3:6].unsqueeze(0) / 2.0
    
    in_x = local_x.abs() <= half_dims[:, :, 0]
    in_y = local_y.abs() <= half_dims[:, :, 1]
    in_z = local_z.abs() <= half_dims[:, :, 2]
    
    mask = in_x & in_y & in_z
    return mask

# --- Новая функция для получения Hulls от SciPy ---

def get_scipy_hull(points):
    """
    Строит выпуклую оболочку через scipy.spatial.ConvexHull.
    Возвращает вершины и грани в формате, совместимом с визуализатором:
    (vertices, faces)
    """
    # Scipy требует минимум 4 точки для 3D hull
    if points.shape[0] < 4:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    
    try:
        hull = ConvexHull(points)
        
        # hull.simplices содержит индексы точек в массиве points.
        # hull.vertices содержит индексы точек, составляющих оболочку.
        
        # Нам нужно вернуть только вершины оболочки и перенумерованные грани.
        # В простейшем случае можно вернуть все points, но лучше вернуть только уникальные вершины оболочки.
        
        # Получаем уникальные индексы вершин, участвующих в гранях
        # (SciPy иногда выдает hull.vertices, но надежнее взять уникальные из simplices,
        # или просто использовать points как есть, если точек немного).
        # Для визуализации "как есть" проще вернуть сам массив points и simplices.
        
        vertices = points
        faces = hull.simplices
        
        return vertices, faces
        
    except Exception as e:
        # QhullError случается для вырожденных случаев (все точки на одной плоскости)
        # print(f"Scipy hull error: {e}")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

if __name__ == "__main__":
    vis = LidarVisualizer(title="PandaSet SciPy Hull Verification")
    
    DATA_ROOT = '/mnt/nvme/datasets/pandaset_converted'
    dataset = PandaDatasetConverted(DATA_ROOT, preindex_all_sweep_files=True)

    if len(dataset) > 0:
        for sweep_idx in range(len(dataset)):
            sweep = dataset.get_sweep(sweep_idx)
            for snapshot_idx in tqdm(range(len(sweep))):
                snapshot: Snapshot = sweep[snapshot_idx]
                pts_ego = snapshot.lidar['points']
                boxes_3d_ego = snapshot.boxes['boxes']
                
                vis.process_events()
                
                # 1. Определяем точки внутри боксов
                pts_torch = torch.from_numpy(pts_ego).to(torch.float16).contiguous().to('cuda')
                boxes_torch = torch.from_numpy(boxes_3d_ego).to(torch.float16).contiguous().to('cuda')
                
                mask_PB = points_in_boxes(pts_torch, boxes_torch)
                is_inside_at_least_one_box = torch.any(mask_PB, dim=1)
                labels_np = is_inside_at_least_one_box.cpu().detach().numpy()
                pts_colors = colorize_point_cloud(pts_ego, labels_np, target_label=1)
                
                # Фильтр боксов, где есть минимум 4 точки
                is_box_have_at_least_N_points = (torch.sum(mask_PB, dim=0) > 3).cpu().detach().numpy()
                mask_PB_cpu = mask_PB.cpu().detach().numpy().astype(np.bool_)
                
                hulls_data = []
                
                t1 = time.perf_counter_ns()
                # 2. Строим Hulls через SciPy
                for box_idx in range(mask_PB_cpu.shape[1]):
                    if not is_box_have_at_least_N_points[box_idx]:
                        continue
                    
                    points_in_this_box = pts_ego[mask_PB_cpu[:, box_idx]]
                    
                    # Используем SciPy
                    hull_vertices, hull_faces = get_scipy_hull(points_in_this_box)
                    
                    if hull_vertices.shape[0] > 0:
                        hulls_data.append((hull_vertices, hull_faces))
                t2 = time.perf_counter_ns()
                print(f"scipy hulls {(t2-t1)/1e6:.2f} ms")
                # 3. Визуализация
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
                            'color': (0.0, 0.0, 1.0, 0.3) # Синий цвет для SciPy
                        }
                    ]
                )
                vis.process_events()