import os
import glob
import numpy as np
import cv2
import transforms3d as t3d
from tqdm import tqdm
from visualizer import LidarVisualizer
import bisect
from pandaset import PandaDatasetConverted,Sweep,Snapshot

def draw_snapshot(
    pts_ego,
    boxes_ego,
    labels,
    cameras_data
):
    """
    Отрисовка снэпшота. Принимает объект Snapshot.
    """

    
    display_w, display_h = 640, 360
    grid_imgs = []
    
    # Порядок камер для грида
    cam_order = ['front_left_camera', 'front_camera', 'front_right_camera',
                 'left_camera', 'back_camera', 'right_camera']
    
    for cam_name in cam_order:
        if cam_name not in cameras_data:
            grid_imgs.append(np.zeros((display_h, display_w, 3), dtype=np.uint8))
            continue
            
        cam = cameras_data[cam_name]
        img = cam['image'].copy() if cam['image'] is not None else np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        if cam['K'] is not None and pts_ego.shape[0] > 0:
            K = cam['K']
            t_vec = cam['sensor2ego_translation']
            q_vec = cam['sensor2ego_rotation'] # w, x, y, z
            
            # Трансформация из Ego в Cam
            # sensor2ego: P_ego = R * P_cam + t  =>  P_cam = R.T * (P_ego - t)
            R_mat = t3d.quaternions.quat2mat(q_vec)
            
            # Точки
            pts_cam = (R_mat.T @ (pts_ego - t_vec).T).T
            mask = pts_cam[:, 2] > 0.5
            pts_cam = pts_cam[mask]
            
            if pts_cam.shape[0] > 0:
                uv = (K @ pts_cam.T).T
                uv[:, 0] /= uv[:, 2]
                uv[:, 1] /= uv[:, 2]
                depths = pts_cam[:, 2]
                max_depth = depths.max() + 1e-6
                
                for j in range(len(uv)):
                    u, v = int(uv[j, 0]), int(uv[j, 1])
                    d = depths[j]
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        norm_d = min(d / max_depth, 1.0)
                        color = (int(255 * (1 - norm_d)), 0, int(255 * norm_d))
                        cv2.circle(img, (u, v), 1, color, -1)

            if boxes_ego.shape[0] > 0:
                corners_unit = np.array([
                    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
                ])
                lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
                
                for k in range(boxes_ego.shape[0]):
                    pos = boxes_ego[k, :3]
                    dim = boxes_ego[k, 3:6]
                    yaw = boxes_ego[k, 6]
                    
                    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                    corners_ego = (R_yaw @ (dim * corners_unit).T).T + pos
                    corners_cam = (R_mat.T @ (corners_ego - t_vec).T).T
                    
                    if corners_cam[:, 2].min() < 0.2:
                        continue
                    
                    uv_box = (K @ corners_cam.T).T
                    uv_box[:, 0] /= uv_box[:, 2]
                    uv_box[:, 1] /= uv_box[:, 2]
                    uv_box = uv_box[:, :2].astype(int)
                    
                    for start, end in lines:
                        cv2.line(img, tuple(uv_box[start]), tuple(uv_box[end]), (0, 255, 0), 2)
                    
                    # Лейбл
                    if len(labels) > k:
                        top_center_ego = pos + np.array([0, 0, dim[2]/2])
                        top_center_cam = R_mat.T @ (top_center_ego - t_vec)
                        if top_center_cam[2] > 0.1:
                            uv_label = K @ top_center_cam
                            u_l, v_l = int(uv_label[0]/uv_label[2]), int(uv_label[1]/uv_label[2])
                            cv2.putText(img, str(labels[k]), (u_l - 20, v_l - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        img_resized = cv2.resize(img, (display_w, display_h))
        grid_imgs.append(img_resized)
        
    # Сборка грида 2x3
    if len(grid_imgs) >= 3:
        row1 = np.hstack(grid_imgs[:3])
    else:
        row1 = np.zeros((display_h, display_w * 3, 3), dtype=np.uint8)
        
    if len(grid_imgs) >= 6:
        row2 = np.hstack(grid_imgs[3:6])
    else:
        # Дозаполняем черным если камер меньше 6
        missing = 6 - len(grid_imgs)
        if len(grid_imgs) > 3:
             existing = np.hstack(grid_imgs[3:])
             pad = np.zeros((display_h, display_w * missing, 3), dtype=np.uint8)
             row2 = np.hstack([existing, pad])
        else:
             row2 = np.zeros((display_h, display_w * 3, 3), dtype=np.uint8)
         
    return np.vstack([row1, row2])


if __name__ == "__main__":
    vis = LidarVisualizer(title="PandaSet 3D LiDAR")
    cv2.namedWindow("Snapshot View", cv2.WINDOW_NORMAL)
    
    
    DATA_ROOT = '/mnt/nvme/datasets/pandaset_converted'
    
    dataset = PandaDatasetConverted(DATA_ROOT, preindex_all_sweep_files=True)

    if len(dataset) > 0:
        for sweep_idx in range(len(dataset)):
            sweep = dataset.get_sweep(sweep_idx)
            for snapshot_idx in tqdm(range(len(sweep))):
                snapshot:Snapshot = sweep[snapshot_idx]
                # {'lat': 37.7747430157756, 'long': -122.40097178666713, 'height': 3.0745996995937364, 'speed': 9.005294706004106}
                print(snapshot.gps)
                img_grid = draw_snapshot(
                    snapshot.lidar['points'],
                    snapshot.boxes['boxes'],
                    snapshot.boxes['class_names'],
                    snapshot.cameras
                )
                vis.update(
                snapshot.lidar['points'],# pts_ego, 
                snapshot.boxes['boxes'],# snapshot.boxes,# boxes_ego, 
                None# labels_ego
                )
                vis.process_events()
                cv2.imshow("Snapshot View", img_grid)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                
                    
        cv2.destroyAllWindows()