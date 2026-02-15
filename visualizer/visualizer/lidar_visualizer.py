import os
import json
import vispy 

vispy.use('glfw')

import numpy as np
import pandas as pd
import cv2
import transforms3d as t3d
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, TypeVar, overload
from abc import ABCMeta, abstractmethod

# Импорты Vispy
from vispy import app, scene
from vispy.scene import visuals
from vispy.color import Colormap



class LidarVisualizer:
    def __init__(self, title="LiDAR 3D View"):
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), show=True, title=title)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.fov = 45
        self.view.camera.distance = 50
        
        visuals.XYZAxis(parent=self.view.scene)
        self.scatter = visuals.Markers(parent=self.view.scene)
        self.scatter.set_data(np.zeros((1, 3)), face_color='white', size=2)
        
        # Линии для границ боксов
        self.lines = visuals.Line(parent=self.view.scene, color='green', width=2, connect='segments')
        
        # <<< НОВОЕ: Линии для стрелок направления >>>
        # Делаем их красными и толстыми (width=4)
        self.arrows = visuals.Line(parent=self.view.scene, color='red', width=4, connect='segments')
        
        self.cmap = Colormap(['blue', 'cyan', 'green', 'yellow', 'red'])

    def update(self, points, boxes, labels=None):
        # --- Отрисовка точек ---
        if points.shape[0] > 0:
            z_vals = points[:, 2]
            z_min, z_max = -3.0, 3.0
            z_norm = (z_vals - z_min) / (z_max - z_min)
            z_norm = np.clip(z_norm, 0, 1)
            colors = self.cmap.map(z_norm)
            self.scatter.set_data(points, face_color=colors, edge_color=None, size=2, edge_width=0)
        else:
            self.scatter.set_data(np.zeros((0, 3)))

        # --- Отрисовка боксов и стрелок ---
        if boxes.shape[0] > 0:
            # Отрисовка граней боксов
            line_data, text_pos, text_texts = self._process_boxes_for_vis(boxes, labels)
            if line_data.shape[0] > 0:
                self.lines.set_data(line_data, color=(0, 1, 0, 0.8))
            
            # <<< НОВОЕ: Отрисовка стрелок направления >>>
            arrow_data = self._process_arrows_for_vis(boxes)
            if arrow_data.shape[0] > 0:
                # Цвет красный, полупрозрачный (1, 0, 0, 0.9)
                self.arrows.set_data(arrow_data, color=(1, 0, 0, 0.9))
        else:
            self.lines.set_data(np.zeros((0, 3)))
            self.arrows.set_data(np.zeros((0, 3)))

    def _process_boxes_for_vis(self, boxes, labels=None):
        lines = []
        txt_positions = []
        txt_labels = []
        corners_unit = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5]
        ])
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        for i in range(boxes.shape[0]):
            x, y, z, dx, dy, dz, yaw = boxes[i]
            pos = np.array([x, y, z])
            dim = np.array([dx, dy, dz])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            corners = (R @ (dim * corners_unit).T).T + pos
            for start, end in edges:
                lines.append(corners[start])
                lines.append(corners[end])
            txt_positions.append(pos + np.array([0, 0, dim[2] / 2]))
            if labels is not None:
                txt_labels.append(labels[i])
        return np.array(lines) if lines else np.zeros((0, 3)), np.array(txt_positions), txt_labels

    def _process_arrows_for_vis(self, boxes):
        """
        Создает линии для отображения направления (стрелок).
        Стрелка идет от центра к передней грани бокса по оси X (длина dx).
        """
        arrow_segments = []
        for i in range(boxes.shape[0]):
            x, y, z, dx, dy, dz, yaw = boxes[i]
            
            # Центр бокса
            start_point = np.array([x, y, z])
            
            # Направляющий вектор (ось X бокса) с длиной dx/2
            # Это приведет стрелку ровно к краю бокса
            # dx - это длина бокса, делим на 2, чтобы дойти от центра до грани
            half_length = dx / 2.0
            
            # Вектор направления в глобальных координатах
            # При yaw=0, X направлен вдоль глобального X.
            # Умножаем длину на единичный вектор направления (cos, sin)
            dir_x = half_length * np.cos(yaw)
            dir_y = half_length * np.sin(yaw)
            
            # Конец стрелки (передняя грань по центру)
            end_point = np.array([x + dir_x, y + dir_y, z])
            
            arrow_segments.append(start_point)
            arrow_segments.append(end_point)
            
        return np.array(arrow_segments) if arrow_segments else np.zeros((0, 3))