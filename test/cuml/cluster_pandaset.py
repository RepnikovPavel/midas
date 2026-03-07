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


def get_cluster_colors_rgb(labels: np.ndarray) -> np.ndarray:
    """
    Генерирует цвета для меток кластеров.
    - Шум (-1) окрашивается в серый цвет.
    - Кластеры окрашиваются перемешанной палитрой Jet для высокой контрастности.

    Args:
        labels (np.ndarray): Массив меток (N,), тип int32. -1 означает шум.

    Returns:
        np.ndarray: Массив цветов (N, 3), тип uint8 (RGB).
    """
    labels = np.asarray(labels, dtype=np.int32)
    n_points = labels.shape[0]
    
    # Результат по умолчанию — серый цвет (для шума и фона)
    colors = np.full((n_points, 3), 128, dtype=np.uint8)
    
    # Получаем список уникальных ID кластеров (исключаем -1)
    unique_labels = np.unique(labels)
    cluster_ids = unique_labels[unique_labels != -1]
    
    if len(cluster_ids) == 0:
        return colors
        
    # 1. Генерируем палитру Jet (256 цветов)
    # cv2.applyColorMap ожидает изображение (H, W), создаем градиент 1x256
    spectrum = np.arange(256, dtype=np.uint8).reshape(1, 256)
    # cv2 выдает BGR, конвертируем в RGB
    jet_palette = cv2.applyColorMap(spectrum, cv2.COLORMAP_JET)
    jet_palette = cv2.cvtColor(jet_palette, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    
    num_clusters = len(cluster_ids)
    
    # 2. Равномерно выбираем индексы цветов из палитры
    #.linspace гарантирует, что мы охватим весь спектр от синего до красного
    color_indices = np.linspace(0, 255, num_clusters, dtype=np.int32)
    
    # 3. Перемешиваем индексы, чтобы соседние кластеры имели разные цвета
    np.random.shuffle(color_indices)
    
    # 4. Создаем Look-Up Table (LUT) для быстрого присвоения цвета
    # Размер таблицы = максимальный ID кластера + 1
    max_label_id = cluster_ids.max()
    lut = np.zeros((max_label_id + 1, 3), dtype=np.uint8)
    
    # Заполняем таблицу перемешанными цветами
    lut[cluster_ids] = jet_palette[color_indices]
    
    # 5. Применяем цвета к точкам
    # Маска для всех точек, которые не являются шумом
    clustered_mask = labels != -1
    
    # Присваиваем цвета, используя метки как индексы в LUT
    colors[clustered_mask] = lut[labels[clustered_mask]]
    
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
                pts_ego = snapshot.lidar['points']
                print(f"pts shape {pts_ego.shape}")
                boxes_3d_ego = snapshot.boxes['boxes']
                cudf_ = cudf.DataFrame()
                cudf_['x'] = pts_ego[:,0]
                cudf_['y'] = pts_ego[:,1]
                cudf_['z'] = pts_ego[:,2]
                
                dbscan_float = DBSCAN(eps=0.32, min_samples=20)
                
                torch.cuda.synchronize()
                t1 = time.perf_counter_ns()
                dbscan_float.fit(cudf_)
                torch.cuda.synchronize()
                t2 = time.perf_counter_ns()
                cluster_labels_cupy = dbscan_float.labels_.values
                cluster_labels_np = cupy.asnumpy(dbscan_float.labels_.values)
                cluster_colors_np = get_cluster_colors_rgb(cluster_labels_np)

                """
                unique [ -1   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
                17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34
                35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52
                53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70
                71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88
                89  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106
                107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124
                125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142
                143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160
                161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178
                179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196
                197 198 199 200 201 202 203 204 205 206 207 208 209 210] 
                """
                
                lvs_,lcnt_ = cupy.unique(dbscan_float.labels_.values,return_counts=True)
                unclustered_points_mask_np = cupy.asnumpy(cluster_labels_cupy==-1).astype(np.bool_)
                clustered_points_mask_np = cupy.asnumpy(cluster_labels_cupy!=-1).astype(np.bool_)

                print(f"cuml clustering time {(t2-t1)/1e6:.2f} ms")
                """
                {'lat': 37.7747430157756, 'long': -122.40097178666713, 'height': 3.0745996995937364, 'speed': 9.005294706004106}
                """
                print(snapshot.gps) 

                vis.update(
                    [
                        {
                            'plot_type':'points',
                            'data':pts_ego,
                            'colors':cluster_colors_np,
                        },
                        {
                            'plot_type':'boxes',
                            'data':boxes_3d_ego
                        }
                    ]
                )

                vis.process_events()