import numpy as np
from numba import njit, float32, int64, boolean

# ---------------------------------------------------------------------------
# Математические утилиты
# ---------------------------------------------------------------------------

@njit
def vec3_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@njit
def vec3_len_sq(a):
    return a[0]*a[0] + a[1]*a[1] + a[2]*a[2]

@njit
def vec3_cross(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ], dtype=np.float32)

@njit
def get_signed_distance_to_plane(q, normal, D):
    return vec3_dot(normal, q) + D

@njit
def get_triangle_normal(a, b, c):
    # (b-a) x (c-a)
    ab = b - a
    ac = c - a
    return vec3_cross(ab, ac)

# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

@njit
def get_convex_hull(points):
    """
    Вычисляет выпуклую оболочку, используя структуру граней с соседями
    (аналогично предоставленному C++ коду).
    """
    N = points.shape[0]
    
    if N < 4:
        verts = points[:min(N, 3)].copy()
        faces = np.zeros((1, 3), dtype=np.int64)
        if N > 0: faces[0, 0] = 0
        if N > 1: faces[0, 1] = 1
        if N > 2: faces[0, 2] = 2
        return verts, faces

    # -----------------------------------------------------------
    # 1. Инициализация памяти
    # -----------------------------------------------------------
    # Используем подход C++ кода: Грань хранит вершины и соседей
    MAX_FACES = 4 * N + 100 # Запас
    
    # Хранилище граней
    # face_v[i, k] - k-я вершина i-й грани (0..2)
    face_v = np.full((MAX_FACES, 3), -1, dtype=np.int64)
    # face_n[i, k] - индекс соседней грани напротив k-й вершины
    face_n = np.full((MAX_FACES, 3), -1, dtype=np.int64)
    # Геометрия
    face_normal = np.zeros((MAX_FACES, 3), dtype=np.float32)
    face_D = np.zeros(MAX_FACES, dtype=np.float32)
    face_active = np.zeros(MAX_FACES, dtype=boolean)
    
    # Пулы и счетчики
    face_count = 0
    free_face_stack = np.zeros(MAX_FACES, dtype=np.int64)
    free_face_count = 0
    
    # Очередь граней для обработки (FIFO или стек)
    process_stack = np.zeros(MAX_FACES, dtype=np.int64)
    process_count = 0
    
    # Связь точка -> грань (для accelerated partitioning)
    point_face = np.full(N, -1, dtype=np.int64)
    
    # Временные буферы для DFS видимых граней
    # Используем статические массивы для избежания аллокаций в njit
    visible_list = np.zeros(MAX_FACES, dtype=np.int64)
    visible_count = 0
    visited_mark = np.zeros(MAX_FACES, dtype=boolean) # Отметки посещения для текущего DFS
    
    # Буфер для новых граней (horizon processing)
    # В 3D новый конус создает граней столько же, сколько ребер горизонта
    new_facet_data = np.zeros((MAX_FACES, 4), dtype=np.int64) # [v1, v2, neighbor_idx, -1]
    new_facet_count = 0
    
    # -----------------------------------------------------------
    # Вспомогательные функции
    # -----------------------------------------------------------
    def add_face(v0, v1, v2, n0, n1, n2):
        nonlocal face_count, free_face_count
        idx = -1
        if free_face_count > 0:
            idx = free_face_stack[free_face_count - 1]
            free_face_count -= 1
        else:
            idx = face_count
            face_count += 1
        
        face_v[idx, 0] = v0
        face_v[idx, 1] = v1
        face_v[idx, 2] = v2
        face_n[idx, 0] = n0
        face_n[idx, 1] = n1
        face_n[idx, 2] = n2
        face_active[idx] = True
        return idx

    def remove_face(idx):
        nonlocal free_face_count
        face_active[idx] = False
        free_face_stack[free_face_count] = idx
        free_face_count += 1

    def set_plane(idx):
        v0, v1, v2 = face_v[idx, 0], face_v[idx, 1], face_v[idx, 2]
        p0, p1, p2 = points[v0], points[v1], points[v2]
        n = get_triangle_normal(p0, p1, p2)
        face_normal[idx][:] = n
        face_D[idx] = -vec3_dot(n, p0)

    # -----------------------------------------------------------
    # 2. Начальный тетраэдр (аналогично C++ create_initial_simplex)
    # -----------------------------------------------------------
    # Находим 4 точки
    extremes = np.zeros(6, dtype=np.int64)
    vals = np.array([points[0,0], points[0,0], points[0,1], points[0,1], points[0,2], points[0,2]], dtype=np.float32)
    for i in range(1, N):
        p = points[i]
        if p[0] > vals[0]: vals[0] = p[0]; extremes[0] = i
        elif p[0] < vals[1]: vals[1] = p[0]; extremes[1] = i
        if p[1] > vals[2]: vals[2] = p[1]; extremes[2] = i
        elif p[1] < vals[3]: vals[3] = p[1]; extremes[3] = i
        if p[2] > vals[4]: vals[4] = p[2]; extremes[4] = i
        elif p[2] < vals[5]: vals[5] = p[2]; extremes[5] = i

    max_d = -1.0
    p0, p1 = 0, 0
    for i in range(6):
        for j in range(i+1, 6):
            d = vec3_len_sq(points[extremes[i]] - points[extremes[j]])
            if d > max_d:
                max_d = d
                p0 = extremes[i]
                p1 = extremes[j]

    s = points[p0]
    v = points[p1] - points[p0]
    v_len_sq = vec3_len_sq(v)
    v_inv_len_sq = 1.0 / v_len_sq if v_len_sq != 0 else 0.0
    
    max_d = -1.0
    p2 = 0
    for i in range(N):
        d = vec3_len_sq(points[i] - s) - (vec3_dot(points[i] - s, v)**2) * v_inv_len_sq
        if d > max_d:
            max_d = d
            p2 = i
            
    n_base = get_triangle_normal(points[p0], points[p1], points[p2])
    d_base = -vec3_dot(n_base, points[p0])
    
    max_d = -1.0
    p3 = 0
    for i in range(N):
        dist = abs(get_signed_distance_to_plane(points[i], n_base, d_base))
        if dist > max_d:
            max_d = dist
            p3 = i

    # Ориентация
    if get_signed_distance_to_plane(points[p3], n_base, d_base) > 0:
        p0, p1 = p1, p0 # Переворот основания

    # Создаем 4 грани тетраэдра
    # Грань 0: p0, p1, p2. Соседи: 3, 1, 2
    f0 = add_face(p0, p1, p2, 3, 1, 2)
    set_plane(f0)
    
    # Грань 1: p0, p2, p3. Соседи: 3, 2, 0
    f1 = add_face(p0, p2, p3, 3, 2, 0)
    set_plane(f1)
    
    # Грань 2: p0, p3, p1. Соседи: 3, 0, 1
    f2 = add_face(p0, p3, p1, 3, 0, 1)
    set_plane(f2)
    
    # Грань 3: p1, p3, p2. Соседи: 1, 2, 0
    f3 = add_face(p1, p3, p2, 1, 2, 0)
    set_plane(f3)

    # Распределение точек (Partitioning)
    for i in range(N):
        if i == p0 or i == p1 or i == p2 or i == p3: continue
        for f_idx in range(4):
            if get_signed_distance_to_plane(points[i], face_normal[f_idx], face_D[f_idx]) > 1e-6:
                point_face[i] = f_idx
                break # Точка принадлежит первой же грани с положительной стороной
    
    # Добавляем грани в очередь
    for f_idx in range(4):
        process_stack[process_count] = f_idx
        process_count += 1

    # -----------------------------------------------------------
    # 3. Основной цикл (аналог C++ create_convex_hull)
    # -----------------------------------------------------------
    while process_count > 0:
        process_count -= 1
        current_f = process_stack[process_count]
        
        if not face_active[current_f]: continue
        
        # Находим самую дальнюю точку
        apex = -1
        max_dist = 0.0
        for i in range(N):
            if point_face[i] == current_f:
                dist = get_signed_distance_to_plane(points[i], face_normal[current_f], face_D[current_f])
                if dist > max_dist:
                    max_dist = dist
                    apex = i
        
        if apex == -1: continue # Нет точек снаружи
        
        # --- Поиск видимых граней (DFS из C++ process_visibles) ---
        visible_count = 0
        stack_ptr = 0
        dfs_stack = np.zeros(MAX_FACES, dtype=np.int64)
        
        dfs_stack[stack_ptr] = current_f
        stack_ptr += 1
        
        # Сброс отметок visited
        # В C++ используется 'visited_' set. Тут обнуляем массив (дорого, но безопасно) или используем поколение (сложно)
        # Для скорости можно просто инкрементировать глобальный счетчик поколений, но numba простых типов...
        # Проще очистить список после использования.
        
        new_facet_count = 0
        
        while stack_ptr > 0:
            stack_ptr -= 1
            f_idx = dfs_stack[stack_ptr]
            
            if visited_mark[f_idx]: continue
            visited_mark[f_idx] = True
            
            # Проверяем видимость
            if get_signed_distance_to_plane(points[apex], face_normal[f_idx], face_D[f_idx]) > 1e-9:
                visible_list[visible_count] = f_idx
                visible_count += 1
                
                # Соседи
                for i in range(3):
                    neigh = face_n[f_idx, i]
                    if neigh != -1:
                        dfs_stack[stack_ptr] = neigh
                        stack_ptr += 1
        
        # --- Обработка горизонта (Horizon processing) ---
        # Для каждой видимой грани проверяем её соседей
        # Если сосед не виден -> это граница горизонта
        # Создаем новые грани
        
        temp_new_faces = [] # (vA, vB, neighbor_face_idx)
        
        # Проходим по всем видимым граням
        for vi in range(visible_count):
            v_f_idx = visible_list[vi]
            
            # Перераспределяем точки видимой грани
            # В C++: outside_.splice(std::cend(outside_), std::move(facet_.outside_));
            # Тут мы просто сбросим point_face для этих точек ниже
            
            for i in range(3):
                neighbor_idx = face_n[v_f_idx, i]
                
                # Если сосед не существует или не видим -> ребро горизонта
                is_horizon = False
                if neighbor_idx == -1:
                    is_horizon = True
                else:
                    if not visited_mark[neighbor_idx]:
                        is_horizon = True
                
                if is_horizon:
                    # Ребро горизонта.
                    # Ребро определяется вершинами грани v_f_idx, ИСКЛЮЧАЯ вершину напротив neighbor_idx
                    # Напротив neighbor_idx (который i) лежит вершина face_v[v_f_idx, i]
                    # Ребро: face_v[v_f_idx, (i+1)%3] -> face_v[v_f_idx, (i+2)%3]
                    # Порядок важен! Нормаль видимой грани смотрит на нас. Ребро должно быть против часовой стрелки от вершины?
                    # Нет, новая грань должна быть (apex, edge_start, edge_end).
                    # Чтобы нормаль смотрела наружу:
                    # Смотрим снаружи на видимую грань. Ребро идет по кругу.
                    # Новая грань "вырастает" из ребра к apex.
                    # Вершины новой грани: (apex, edge_start, edge_end).
                    
                    edge_idx1 = (i + 1) % 3
                    edge_idx2 = (i + 2) % 3
                    
                    v_start = face_v[v_f_idx, edge_idx1]
                    v_end = face_v[v_f_idx, edge_idx2]
                    
                    # Создаем заготовку для новой грани
                    # Нам нужно запомнить: v_start, v_end, neighbor_idx
                    # Позже мы свяжем их между собой
                    
                    # Сохраняем в temp массив
                    # new_facet_data: [v_start, v_end, neighbor_idx, -1]
                    if new_facet_count < MAX_FACES:
                        new_facet_data[new_facet_count, 0] = v_start
                        new_facet_data[new_facet_count, 1] = v_end
                        new_facet_data[new_facet_count, 2] = neighbor_idx
                        new_facet_data[new_facet_count, 3] = v_f_idx # Для связывания соседей (старая видимая грань)
                        new_facet_count += 1
        
        # Удаление видимых граней
        for vi in range(visible_count):
            remove_face(visible_list[vi])
            visited_mark[visible_list[vi]] = False # Сброс
            
        # Сброс точек (чтобы переназначить)
        # Точки, принадлежавшие видимым граням, должны быть переназначены новым
        # В упрощенной реализации (accelerated partitioning) мы сбрасываем связь
        # и переназначаем ниже
        for i in range(N):
            for vi in range(visible_count):
                if point_face[i] == visible_list[vi]:
                    point_face[i] = -1
                    break

        # Создание новых граней
        created_faces_indices = np.zeros(new_facet_count, dtype=np.int64)
        
        # Сначала создаем грани
        for k in range(new_facet_count):
            v1 = new_facet_data[k, 0]
            v2 = new_facet_data[k, 1]
            # neighbor_old = new_facet_data[k, 2] # Сосед с другой стороны горизонта
            
            # Создаем грань (apex, v1, v2)
            # Соседи: 0 (напротив apex) - это neighbor_old. Остальные 1 и 2 пока неизвестны
            n_idx = add_face(apex, v1, v2, -1, -1, -1)
            set_plane(n_idx)
            created_faces_indices[k] = n_idx
            
            # Обновляем старого соседа (neighbor_old), чтобы он указывал на новую грань
            neighbor_old = new_facet_data[k, 2]
            if neighbor_old != -1 and face_active[neighbor_old]:
                # Находим в neighbor_old индекс грани, который указывал на visible_face
                visible_face_old = new_facet_data[k, 3]
                
                # Ищем индекс в face_n[neighbor_old], который равен visible_face_old
                slot = -1
                for t in range(3):
                    if face_n[neighbor_old, t] == visible_face_old:
                        slot = t
                        break
                
                if slot != -1:
                    face_n[neighbor_old, slot] = n_idx
                    # У новой грани сосед 0 (напротив apex) - это neighbor_old
                    face_n[n_idx, 0] = neighbor_old
            
        # Связывание новых граней между собой (Adjacency update)
        # Новые грани образуют "веер" вокруг apex.
        # Грань A(v1, v2) соседствует с гранью B(v2, v3) по ребру (apex, v2)
        # Нужно найти пары (v_end == v_start)
        for k in range(new_facet_count):
            idx_k = created_faces_indices[k]
            # Ищем соседа для слота 1 (напротив v1) -> ребро (apex, v2)
            # Ищем соседа для слота 2 (напротив v2) -> ребро (apex, v1)
            
            v1_k = new_facet_data[k, 0]
            v2_k = new_facet_data[k, 1]
            
            # Ищем грань, у которой v_start == v2_k (чтобы соединить по ребру apex-v2_k)
            if face_n[idx_k, 1] == -1: # Слот напротив v1 (ребро apex-v2)
                for j in range(new_facet_count):
                    if k == j: continue
                    # Вершина start у j равна v2_k?
                    if new_facet_data[j, 0] == v2_k:
                         # Нашли соседа!
                         face_n[idx_k, 1] = created_faces_indices[j]
                         break
                    if new_facet_data[j, 1] == v1_k:
                         # Обратная ситуация (зависит от порядка обхода)
                         pass

            if face_n[idx_k, 2] == -1: # Слот напротив v2 (ребро apex-v1)
                 for j in range(new_facet_count):
                    if k == j: continue
                    if new_facet_data[j, 0] == v2_k: # j начинается с v2_k -> это сосед для слота 2? Нет.
                        # Грань k: (apex, v1, v2). Сосед напротив v2 (ребро apex-v1) должен начинаться с v1? Нет.
                        # Ребро apex-v1. Грань с вершинами (apex, v1, ...). v1 должен быть вторым (end)?
                        pass
                    # Логика проще:
                    # Сосед напротив v1 (это ребро apex-v2) -> ищем грань с вершинами (apex, v2, ...)
                    # Сосед напротив v2 (это ребро apex-v1) -> ищем грань с вершинами (apex, ..., v1) -> (apex, v1, ...)
                    if new_facet_data[j, 0] == v2_k: # j: (apex, v2, ...)
                        face_n[idx_k, 1] = created_faces_indices[j] # k напротив v1 соединяем с j
                    
                    if new_facet_data[j, 1] == v1_k: # j: (apex, ..., v1) -> (apex, X, v1) -> wrong order?
                         # Наша грань k: (apex, v1, v2).
                         # Ребро (apex, v1). Напротив v2.
                         # Ищем грань (apex, v1, X).
                         # В данных: (v_start, v_end). (v1, X).
                         if new_facet_data[j, 0] == v1_k:
                             face_n[idx_k, 2] = created_faces_indices[j]

        # Перераспределение точек
        # Точки, у которых point_face == -1 (были сброшены) или принадлежали удаленным
        # Пытаемся добавить в новые грани
        for i in range(N):
            if point_face[i] == -1:
                for k in range(new_facet_count):
                    nf = created_faces_indices[k]
                    if get_signed_distance_to_plane(points[i], face_normal[nf], face_D[nf]) > 1e-6:
                        point_face[i] = nf
                        break
        
        # Добавляем новые грани в стек обработки
        for k in range(new_facet_count):
            nf = created_faces_indices[k]
            # Проверка: есть ли точки?
            has_points = False
            for i in range(N):
                if point_face[i] == nf:
                    has_points = True
                    break
            if has_points:
                process_stack[process_count] = nf
                process_count += 1

    # -----------------------------------------------------------
    # 4. Формирование результата
    # -----------------------------------------------------------
    # Собираем активные грани
    out_f_count = 0
    for i in range(face_count):
        if face_active[i]:
            out_f_count += 1
            
    # Выводим сразу индексы точек оригинального облака
    # Формат: (M, 3) индексы
    hull_faces = np.zeros((out_f_count, 3), dtype=np.int64)
    curr = 0
    for i in range(face_count):
        if face_active[i]:
            hull_faces[curr, 0] = face_v[i, 0]
            hull_faces[curr, 1] = face_v[i, 1]
            hull_faces[curr, 2] = face_v[i, 2]
            curr += 1
            
    # Возвращаем только уникальные вершины
    used_v = np.zeros(N, dtype=boolean)
    for i in range(out_f_count):
        used_v[hull_faces[i, 0]] = True
        used_v[hull_faces[i, 1]] = True
        used_v[hull_faces[i, 2]] = True
        
    v_count = 0
    for i in range(N):
        if used_v[i]: v_count += 1
        
    hull_vertices = np.zeros((v_count, 3), dtype=np.float32)
    old_to_new = np.full(N, -1, dtype=np.int64)
    c = 0
    for i in range(N):
        if used_v[i]:
            hull_vertices[c] = points[i]
            old_to_new[i] = c
            c += 1
            
    # Перенумеруем грани
    for i in range(out_f_count):
        hull_faces[i, 0] = old_to_new[hull_faces[i, 0]]
        hull_faces[i, 1] = old_to_new[hull_faces[i, 1]]
        hull_faces[i, 2] = old_to_new[hull_faces[i, 2]]
            
    return hull_vertices, hull_faces