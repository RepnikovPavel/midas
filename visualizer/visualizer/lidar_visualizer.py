import numpy as np
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
        
        # Линии для стрелок направления
        self.arrows = visuals.Line(parent=self.view.scene, color='red', width=4, connect='segments')
        
        self.cmap = Colormap(['blue', 'cyan', 'green', 'yellow', 'red'])

        # Оптимизация: предвычисление геометрии куба
        self.corners_unit = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5]
        ])
        self.edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

    def process_events(self):
        """
        Неблокирующий метод обновления GUI.
        Нужно вызывать в каждом шаге цикла после update().
        """
        app.process_events()

    def update(self, points, boxes=None, labels=None):
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

        if boxes is not None and boxes.shape[0] > 0:
            # Отрисовка граней боксов
            line_data = self._process_boxes_for_vis(boxes)
            if line_data.shape[0] > 0:
                self.lines.set_data(line_data, color=(0, 1, 0, 0.8))
            
            # Отрисовка стрелок направления
            arrow_data = self._process_arrows_for_vis(boxes)
            if arrow_data.shape[0] > 0:
                self.arrows.set_data(arrow_data, color=(1, 0, 0, 0.9))
        else:
            self.lines.set_data(np.zeros((0, 3)))
            self.arrows.set_data(np.zeros((0, 3)))

    def _process_boxes_for_vis(self, boxes):
        """Оптимированная генерация линий для боксов."""
        N = boxes.shape[0]
        # Результирующий массив: N боксов * 12 ребер * 2 точки
        all_lines = np.zeros((N * len(self.edges), 2, 3))
        
        for i in range(N):
            x, y, z, dx, dy, dz, yaw = boxes[i]
            pos = np.array([x, y, z])
            dim = np.array([dx, dy, dz])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            # Вычисляем углы
            corners = (R @ (dim * self.corners_unit).T).T + pos
            
            # Заполняем линии для текущего бокса
            for j, (start_idx, end_idx) in enumerate(self.edges):
                all_lines[i * len(self.edges) + j, 0] = corners[start_idx]
                all_lines[i * len(self.edges) + j, 1] = corners[end_idx]
        
        # Reshape для формата vispy (N*24, 3)
        return all_lines.reshape(-1, 3)

    def _process_arrows_for_vis(self, boxes):
        """Создает линии для отображения направления (стрелок)."""
        N = boxes.shape[0]
        arrow_segments = np.zeros((N * 2, 3)) # 2 точки на бокс (начало и конец)
        
        for i in range(N):
            x, y, z, dx, dy, dz, yaw = boxes[i]
            
            # Центр бокса
            start_point = np.array([x, y, z])
            
            # Вектор направления (передняя грань)
            half_length = dx / 2.0
            dir_x = half_length * np.cos(yaw)
            dir_y = half_length * np.sin(yaw)
            
            end_point = np.array([x + dir_x, y + dir_y, z])
            
            # Vispy Line с connect='segments' ожидает пары точек подряд
            arrow_segments[i*2] = start_point
            arrow_segments[i*2 + 1] = end_point
            
        return arrow_segments