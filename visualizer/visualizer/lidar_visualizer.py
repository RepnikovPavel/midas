import numpy as np
from vispy import app, scene
from vispy.scene import visuals
from vispy.color import Colormap
import numba

# --- Numba-optimized geometry calculations ---

@numba.njit
def calculate_box_lines(boxes):
    """
    Вычисляет вершины линий для всех боксов одним массивом.
    Возвращает массив формы (N * 24, 3).
    """
    n_boxes = boxes.shape[0]
    # 12 ребер * 2 точки на ребро = 24 вершины
    n_verts = n_boxes * 24
    lines = np.empty((n_verts, 3), dtype=np.float32)
    
    # Индексы ребер для куба с центром в (0,0,0) и размером 1
    # (соответствует логике self.edges)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0], # Нижняя грань
        [4, 5], [5, 6], [6, 7], [7, 4], # Верхняя грань
        [0, 4], [1, 5], [2, 6], [3, 7]  # Боковые ребра
    ])
    
    # Локальные координаты углов единичного куба (от -0.5 до 0.5)
    corners_unit = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5]
    ])

    for i in range(n_boxes):
        x, y, z, dx, dy, dz, yaw = boxes[i]
        
        # Матрица поворота
        c = np.cos(yaw)
        s = np.sin(yaw)
        
        # Вычисляем 8 углов реального бокса
        # Сначала масштабируем единичный куб, потом поворачиваем, потом сдвигаем
        corners = np.empty((8, 3))
        for j in range(8):
            # Масштабирование
            lx = corners_unit[j, 0] * dx
            ly = corners_unit[j, 1] * dy
            lz = corners_unit[j, 2] * dz
            
            # Поворот и сдвиг
            corners[j, 0] = c * lx - s * ly + x
            corners[j, 1] = s * lx + c * ly + y
            corners[j, 2] = lz + z
            
        # Заполняем вершины линий для текущего бокса
        base_idx = i * 24
        for k in range(12):
            p1_idx = edges[k, 0]
            p2_idx = edges[k, 1]
            
            # Точка начала ребра
            lines[base_idx + k*2, 0] = corners[p1_idx, 0]
            lines[base_idx + k*2, 1] = corners[p1_idx, 1]
            lines[base_idx + k*2, 2] = corners[p1_idx, 2]
            
            # Точка конца ребра
            lines[base_idx + k*2 + 1, 0] = corners[p2_idx, 0]
            lines[base_idx + k*2 + 1, 1] = corners[p2_idx, 1]
            lines[base_idx + k*2 + 1, 2] = corners[p2_idx, 2]
            
    return lines

@numba.njit
def calculate_arrows(boxes):
    """
    Вычисляет вершины линий для стрелок направления.
    Возвращает массив формы (N * 2, 3).
    """
    n_boxes = boxes.shape[0]
    arrows = np.empty((n_boxes * 2, 3), dtype=np.float32)
    
    for i in range(n_boxes):
        x, y, z, dx, dy, dz, yaw = boxes[i]
        
        half_l = dx / 2.0
        c = np.cos(yaw)
        s = np.sin(yaw)
        
        # Центр
        arrows[i*2, 0] = x
        arrows[i*2, 1] = y
        arrows[i*2, 2] = z
        
        # Передняя точка
        arrows[i*2 + 1, 0] = x + half_l * c
        arrows[i*2 + 1, 1] = y + half_l * s
        arrows[i*2 + 1, 2] = z
        
    return arrows


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

    def process_events(self):
        app.process_events()

    def update(self, data_list):
        """
        Обновляет визуализацию на основе списка словарей.
        
        Args:
            data_list (list): Список словарей с ключами:
                - 'plot_type': 'points' или 'boxes'
                - 'data': np.ndarray (N, 3) для точек или (N, 7) для боксов
                - 'colors': (опционально) np.ndarray uint8 RGB
        """
        points_accum = []
        points_colors_accum = []
        
        boxes_accum = []
        boxes_colors_accum = []

        for item in data_list:
            plot_type = item.get('plot_type')
            data = item.get('data')
            colors = item.get('colors')
            
            if data is None or data.shape[0] == 0:
                continue

            if plot_type == 'points':
                points_accum.append(data)
                
                if colors is not None:
                    # Конвертация uint8 RGB (0-255) в float RGBA (0-1)
                    c = colors.astype(np.float32) / 255.0
                    # Добавляем альфа-канал
                    c_rgba = np.hstack((c, np.ones((c.shape[0], 1), dtype=np.float32)))
                    points_colors_accum.append(c_rgba)
                else:
                    # Используем карту высот (Z-gradient)
                    z_vals = data[:, 2]
                    z_min, z_max = -3.0, 3.0
                    z_norm = (z_vals - z_min) / (z_max - z_min)
                    z_norm = np.clip(z_norm, 0, 1)
                    points_colors_accum.append(self.cmap.map(z_norm))

            elif plot_type == 'boxes':
                boxes_accum.append(data)
                
                if colors is not None:
                    c = colors.astype(np.float32) / 255.0
                    c_rgba = np.hstack((c, np.ones((c.shape[0], 1), dtype=np.float32)))
                    boxes_colors_accum.append(c_rgba)
                else:
                    # Дефолтный зеленый цвет для всех боксов в пачке
                    # (N, 4)
                    default_box_color = np.tile(np.array([[0, 1, 0, 0.8]], dtype=np.float32), (data.shape[0], 1))
                    boxes_colors_accum.append(default_box_color)

        # --- Отрисовка точек ---
        if len(points_accum) > 0:
            all_points = np.vstack(points_accum)
            all_point_colors = np.vstack(points_colors_accum)
            self.scatter.set_data(all_points, face_color=all_point_colors, edge_color=None, size=2, edge_width=0)
        else:
            self.scatter.set_data(np.zeros((0, 3)))

        # --- Отрисовка боксов ---
        if len(boxes_accum) > 0:
            all_boxes = np.vstack(boxes_accum)
            
            # Вычисляем геометрию через Numba
            line_pos = calculate_box_lines(all_boxes)
            arrow_pos = calculate_arrows(all_boxes)
            
            # Подготовка цветов для линий
            # Нам нужно повторить цвет каждого бокса для 24 вершин линий
            all_box_colors = np.vstack(boxes_colors_accum)
            line_colors = np.repeat(all_box_colors, 24, axis=0)
            
            self.lines.set_data(line_pos, color=line_colors)
            self.arrows.set_data(arrow_pos, color='red') # Стрелки оставляем красными для контраста
        else:
            self.lines.set_data(np.zeros((0, 3)))
            self.arrows.set_data(np.zeros((0, 3)))