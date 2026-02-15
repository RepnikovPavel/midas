from numba import njit
import numpy as np
from numba.typed import List as NumbaList


@njit
def get_corners_numba(box):
    x, y, w, l, yaw = box
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    vx = np.array([cos_yaw * w / 2, sin_yaw * w / 2])
    vy = np.array([-sin_yaw * l / 2, cos_yaw * l / 2])
    corners = np.empty((4, 2))
    # CCW order
    corners[0] = np.array([x, y]) + vx + vy 
    corners[1] = np.array([x, y]) - vx + vy 
    corners[2] = np.array([x, y]) - vx - vy 
    corners[3] = np.array([x, y]) + vx - vy 
    return corners

@njit
def area_polygon(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i, 0] * corners[j, 1]
        area -= corners[j, 0] * corners[i, 1]
    return 0.5 * abs(area)

@njit
def is_inside_edge(p, edge_start, edge_end):
    return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) >= \
           (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])

@njit
def intersection_line(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return p1 
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

@njit
def polygon_intersection_numba(poly_subject, poly_clip):
    output = NumbaList()
    for i in range(poly_subject.shape[0]):
        output.append(poly_subject[i])
    for i in range(poly_clip.shape[0]):
        if len(output) == 0:
            return np.empty((0, 2))
        input_list = output
        output = NumbaList()
        edge_start = poly_clip[i]
        edge_end = poly_clip[(i + 1) % poly_clip.shape[0]]
        for j in range(len(input_list)):
            current_point = input_list[j]
            prev_point = input_list[j - 1]
            cur_inside = is_inside_edge(current_point, edge_start, edge_end)
            prev_inside = is_inside_edge(prev_point, edge_start, edge_end)
            if cur_inside:
                if not prev_inside:
                    inter = intersection_line(prev_point, current_point, edge_start, edge_end)
                    output.append(inter)
                output.append(current_point)
            elif prev_inside:
                inter = intersection_line(prev_point, current_point, edge_start, edge_end)
                output.append(inter)
    if len(output) == 0:
        return np.empty((0, 2))
    res = np.empty((len(output), 2))
    for i in range(len(output)):
        res[i] = output[i]
    return res

@njit
def iou_2d_rotated(box_a, box_b):
    corners_a = get_corners_numba(box_a)
    corners_b = get_corners_numba(box_b)
    area_a = area_polygon(corners_a)
    area_b = area_polygon(corners_b)
    if area_a < 1e-7 or area_b < 1e-7:
        return 0.0
    inter_poly = polygon_intersection_numba(corners_a, corners_b)
    if inter_poly.shape[0] < 3:
        return 0.0
    intersection_area = area_polygon(inter_poly)
    union = area_a + area_b - intersection_area
    if union < 1e-7:
        return 0.0
    return intersection_area / union

@njit
def nms_bev_jit(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.bool_)
    order = np.argsort(-scores) 
    keep = np.zeros(len(boxes), dtype=np.bool_)
    for i in range(len(boxes)):
        idx = order[i]
        if scores[idx] == -1: 
            continue
        keep[idx] = True
        for j in range(i + 1, len(boxes)):
            next_idx = order[j]
            if scores[next_idx] == -1:
                continue
            iou = iou_2d_rotated(boxes[idx], boxes[next_idx])
            if iou > iou_threshold:
                scores[next_idx] = -1 
    return keep