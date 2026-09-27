"""GPU PandaSet viewer (PyQt5 + vispy, 30+ fps, non-blocking).

Run inside the viz container:  python -m pandaset_pipe.visualize --roots R1 R2 [--seq 033]

Layout: 3D point cloud + boxes + class labels | 6 camera views | OSM map.
Controls: Start/Stop button (or Space), frame slider, speedometer (km/h),
checkboxes for lidar/box projection, point-color mode combo, N/B sweeps,
n/b frames. Every view zooms to cursor with the mouse wheel.

All drawing is GPU (vispy visuals: Markers/Line/Image/Text); data loading and
map tiles run in background threads so navigation never stalls.
"""

import argparse
import math
import os
import threading
import time
from collections import OrderedDict, deque

import cv2
import numpy as np

from .reader import PandaDataset
from . import fastops
from .reader import _quat_to_mat

CAM_ORDER = ["front_left_camera", "front_camera", "front_right_camera",
             "left_camera", "back_camera", "right_camera"]

SEMSEG_PALETTE = {
    0: (128, 128, 128), 1: (150, 60, 60), 5: (90, 90, 220), 6: (60, 60, 200),
    7: (80, 80, 180), 11: (220, 220, 60), 13: (160, 100, 40), 15: (200, 160, 60),
    19: (60, 160, 220), 20: (0, 220, 220), 23: (30, 30, 220), 24: (40, 80, 200),
    25: (60, 60, 160), 29: (120, 120, 120), 30: (200, 120, 120),
    34: (140, 140, 60), 35: (90, 180, 90), 41: (100, 100, 180),
}

MAX_LABELS_PER_VIEW = 96
CAM_IMG_W, CAM_IMG_H = 960, 540


def z_colors(z, z_min=-3.0, z_max=3.0):
    """(N,) z -> (N,3) uint8 RGB height colormap."""
    return fastops.height_colors(np.ascontiguousarray(z.astype(np.float64)), z_min, z_max)


class AsyncMap:
    """OSM tile fetching in a background thread; track drawing on CPU (small)."""

    def __init__(self, zoom=17, span=0.004):
        self.zoom = zoom
        self.span = span
        self.track = deque(maxlen=2000)
        self._lock = threading.Lock()
        self._img = None        # base tile (RGB)
        self._map_obj = None
        self._center = None
        self._loading = False
        self._failed = False
        self._gen = 0
        self._ok = True
        try:
            import smopy  # noqa: F401
        except ImportError:
            self._ok = False

    def clear(self):
        self.track.clear()

    def update(self, lat, lon):
        if not self._ok or (lat == 0.0 and lon == 0.0):
            return
        self.track.append((lat, lon))
        need = self._img is None and not self._loading and not self._failed
        if self._center is not None:
            dlat = abs(lat - self._center[0])
            dlon = abs(lon - self._center[1])
            if (dlat > self.span / 2 or dlon > self.span / 2) and not self._loading:
                need = True
        if need:
            self._start_fetch(lat, lon)

    def _start_fetch(self, lat, lon):
        import smopy
        self._loading = True
        self._gen += 1
        gen = self._gen
        span = self.span
        zoom = self.zoom

        def work():
            try:
                m = smopy.Map(lat - span, lon - span, lat + span, lon + span,
                              z=zoom, tilesize=256)
                img = np.array(m.img_pil)  # RGB
                with self._lock:
                    if gen == self._gen:
                        self._img = img
                        self._map_obj = m
                        self._center = (lat, lon)
                        self._failed = False
            except Exception:  # noqa: BLE001
                with self._lock:
                    self._failed = True
            finally:
                with self._lock:
                    self._loading = False

        threading.Thread(target=work, daemon=True).start()

    def render(self, lat, lon, yaw=None):
        """Return RGB image with track+position overlay, or a status tile."""
        with self._lock:
            base = None if self._img is None else self._img.copy()
            m = self._map_obj
            loading, failed = self._loading, self._failed
        if base is None:
            img = np.zeros((512, 512, 3), np.uint8)
            msg = "map loading..." if loading else ("map offline" if (failed or not self._ok) else "no gps")
            cv2.putText(img, msg, (120, 256), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (200, 200, 200), 2)
            return img
        if m is not None:
            for tlat, tlon in list(self.track)[-600:]:
                try:
                    x, y = m.to_pixels(tlat, tlon)
                except Exception:  # noqa: BLE001
                    continue
                cv2.circle(img, (int(x), int(y)), 2, (255, 60, 60), -1)
            try:
                x, y = m.to_pixels(lat, lon)
                cv2.circle(img, (int(x), int(y)), 7, (0, 80, 255), -1)
                if yaw is not None:
                    cv2.arrowedLine(img, (int(x), int(y)),
                                    (int(x + 24 * math.cos(yaw)), int(y - 24 * math.sin(yaw))),
                                    (0, 80, 255), 3, tipLength=0.35)
            except Exception:  # noqa: BLE001
                pass
        return img


def load_payload(sweep, frame, cam_names):
    """Heavy IO (runs in prefetch thread): one frame -> dict of numpy payloads."""
    snap = sweep[frame]
    pts = snap.points                      # float32 (N,3), ego X fwd / Y left
    inten = snap.intensity
    sem = snap.semseg
    boxes, labels = snap.boxes_nms()
    gps = snap.gps
    cams = {}
    for cam in cam_names:
        entry = snap.cameras.get(cam)
        if entry is None or entry.get("K") is None:
            continue
        img = entry["image"]
        h0, w0 = img.shape[:2]
        scale = CAM_IMG_W / w0
        img_s = cv2.resize(img, (CAM_IMG_W, int(h0 * scale)))
        K = entry["K"].astype(np.float64).copy()
        K[0, :] *= scale
        K[1, :] *= scale
        R_s2e = _quat_to_mat(entry["sensor2ego_rotation"])
        t_s2e = entry["sensor2ego_translation"].astype(np.float64)
        R_e2c = R_s2e.T
        t_e2c = -R_s2e.T @ t_s2e
        cams[cam] = {"img": img_s[:, :, ::-1], "K": K, "R": R_e2c, "t": t_e2c}
    return {"pts": pts, "intensity": inten, "semseg": sem,
            "boxes": boxes, "labels": labels, "gps": gps, "cams": cams,
            "ts": snap.ts}


class Viewer:
    def __init__(self, roots, seq=None, data_fps=10.0, no_map=False):
        import vispy
        vispy.use(app="pyqt5")
        from vispy import scene
        from vispy.app import Timer
        from PyQt5 import QtWidgets, QtCore, QtGui

        self.scene_mod = scene
        self.ds = PandaDataset(roots)
        if len(self.ds) == 0:
            raise SystemExit("no sweeps found")
        self.sweep_idx = 0
        if seq:
            names = self.ds.sequence_names()
            if seq in names:
                self.sweep_idx = names.index(seq)
        self.sweep = self.ds[self.sweep_idx]
        self.frame = 0
        self.playing = False
        self.data_fps = data_fps
        self.color_mode = "height"
        self.show_lidar_proj = True
        self.show_box_proj = True

        self._cache = OrderedDict()
        self._pool = None
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending = set()

        # ---------------- Qt main window ----------------
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("PandaSet GPU viewer")
        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        root_l = QtWidgets.QHBoxLayout(central)
        root_l.setContentsMargins(2, 2, 2, 2)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_l.addWidget(split)

        # --- 3D canvas ---
        self.canvas3d = scene.SceneCanvas(keys="interactive", size=(1280, 900),
                                          show=False, title="3D")
        self.view3d = self.canvas3d.central_widget.add_view()
        self.view3d.camera = "turntable"
        self.view3d.camera.fov = 45
        self.view3d.camera.distance = 50
        scene.visuals.XYZAxis(parent=self.view3d.scene)
        self.v3d_points = scene.visuals.Markers(parent=self.view3d.scene)
        self.v3d_boxes = scene.visuals.Line(parent=self.view3d.scene, width=2,
                                            connect="segments", color="green")
        self.v3d_arrows = scene.visuals.Line(parent=self.view3d.scene, width=3,
                                             connect="segments", color="red")
        self.labels3d = [scene.visuals.Text("", parent=self.view3d.scene, color=(1, 1, 0.4, 1),
                                            font_size=10, anchor_x="center", anchor_y="bottom")
                         for _ in range(MAX_LABELS_PER_VIEW)]
        split.addWidget(self.canvas3d.native)

        # --- right: camera grid + map + controls ---
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        split.addWidget(right)
        split.setSizes([1400, 1100])

        self.cam_canvas = scene.SceneCanvas(keys="interactive", size=(1100, 640),
                                            show=False, title="cameras")
        grid = self.cam_canvas.central_widget.add_grid(spacing=2, margin=2)
        self.cam_views = {}
        for i, cam in enumerate(CAM_ORDER):
            vb = grid.add_view(row=i // 3, col=i % 3, border_color=(0.3, 0.3, 0.3, 1))
            vb.camera = scene.PanZoomCamera(aspect=1)
            img_v = scene.visuals.Image(np.zeros((2, 2, 3), np.uint8), parent=vb.scene,
                                        interpolation="bilinear")
            pts_v = scene.visuals.Markers(parent=vb.scene)
            box_v = scene.visuals.Line(parent=vb.scene, width=2, connect="segments",
                                       color="lime")
            title = scene.visuals.Text(cam.replace("_camera", ""), parent=vb.scene,
                                       color=(1, 1, 1, 0.9), font_size=9,
                                       anchor_x="left", anchor_y="top")
            labels_v = [scene.visuals.Text("", parent=vb.scene, color=(0, 1, 1, 1),
                                           font_size=9, anchor_x="center", anchor_y="bottom")
                        for _ in range(MAX_LABELS_PER_VIEW)]
            self.cam_views[cam] = {"vb": vb, "img": img_v, "pts": pts_v, "box": box_v,
                                   "labels": labels_v, "title": title, "K_scale": 1.0}
        rl.addWidget(self.cam_canvas.native, stretch=3)

        self.map_canvas = scene.SceneCanvas(keys="interactive", size=(1100, 300),
                                            show=False, title="map")
        mv = self.map_canvas.central_widget.add_view()
        mv.camera = scene.PanZoomCamera(aspect=1)
        self.map_img = scene.visuals.Image(np.zeros((2, 2, 3), np.uint8), parent=mv.scene,
                                           interpolation="bilinear")
        self.map_view = mv
        self.osm = None if no_map else AsyncMap()
        rl.addWidget(self.map_canvas.native, stretch=1)

        # --- control bar ---
        bar = QtWidgets.QHBoxLayout()
        rl.addLayout(bar)
        self.btn = QtWidgets.QPushButton("Start")
        self.btn.setFixedWidth(90)
        self.btn.clicked.connect(self.toggle_play)
        bar.addWidget(self.btn)

        self.speedo = QtWidgets.QLabel("  --.- km/h ")
        f = QtGui.QFont("monospace", 22, QtGui.QFont.Bold)
        self.speedo.setFont(f)
        self.speedo.setStyleSheet("color: #0f0; background: #111; padding: 2px 10px;")
        bar.addWidget(self.speedo)

        self.chk_lidar = QtWidgets.QCheckBox("LiDAR proj")
        self.chk_lidar.setChecked(True)
        self.chk_lidar.stateChanged.connect(self._flags_changed)
        bar.addWidget(self.chk_lidar)
        self.chk_boxes = QtWidgets.QCheckBox("Boxes proj")
        self.chk_boxes.setChecked(True)
        self.chk_boxes.stateChanged.connect(self._flags_changed)
        bar.addWidget(self.chk_boxes)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["height", "intensity", "semseg"])
        self.combo.currentTextChanged.connect(self._color_changed)
        bar.addWidget(self.combo)

        self.frame_lbl = QtWidgets.QLabel("f 0/0")
        bar.addWidget(self.frame_lbl)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self._slider_seek)
        bar.addWidget(self.slider, stretch=1)

        self.status = QtWidgets.QStatusBar()
        self.win.setStatusBar(self.status)

        # keys on all canvases
        for c in (self.canvas3d, self.cam_canvas, self.map_canvas):
            c.events.key_press.connect(self._on_key)

        self._zoom_init = {"3d": None}
        self._last_advance = 0.0
        self._dirty = True
        self._map_last = 0.0

        self._update_sweep_meta()
        self.timer = Timer(interval=1.0 / 30.0, connect=self._tick, start=True,
                           app=self.app)
        self.win.resize(2500, 1300)
        self.win.show()
        self.toggle_play()  # autostart

    # ---------------- data ----------------
    def _update_sweep_meta(self):
        n = len(self.sweep)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n - 1, 0))
        self.slider.setValue(self.frame)
        self.slider.blockSignals(False)
        self.status.showMessage(f"sweep {self.sweep.name}  ({n} frames)   "
                                f"[space]=start/stop [n/b]=frame [N/B]=sweep [c]=color")
        if self.osm:
            self.osm.clear()
        self._cache.clear()
        self._dirty = True

    def _want(self, sweep, frame):
        key = (id(sweep), frame)
        if key not in self._cache and key not in self._pending:
            self._pending.add(key)

            def job():
                try:
                    pl = load_payload(sweep, frame, self.sweep.camera_names)
                except Exception as e:  # noqa: BLE001
                    pl = {"error": str(e)}
                with self._cache_lock:
                    self._cache[key] = pl
                    self._cache.move_to_end(key)
                    while len(self._cache) > 12:
                        self._cache.popitem(last=False)
                    self._pending.discard(key)
                self._dirty = True

            self._pending.add(key)
            self._executor.submit(job)

    _cache_lock = threading.Lock()

    def _get(self, sweep, frame):
        with self._cache_lock:
            pl = self._cache.get((id(sweep), frame))
            if pl is not None:
                self._cache.move_to_end((id(sweep), frame))
        return pl

    def _prefetch(self):
        n = len(self.sweep)
        for f in range(self.frame, min(self.frame + 6, n)):
            self._want(self.sweep, f)

    # ---------------- UI events ----------------
    def toggle_play(self):
        self.playing = not self.playing
        self.btn.setText("Stop" if self.playing else "Start")

    def _flags_changed(self, *_):
        self.show_lidar_proj = self.chk_lidar.isChecked()
        self.show_box_proj = self.chk_boxes.isChecked()
        self._dirty = True

    def _color_changed(self, text):
        self.color_mode = text
        self._dirty = True

    def _slider_seek(self, v):
        if v != self.frame:
            self.frame = v
            self._dirty = True
            self._prefetch()

    def _on_key(self, event):
        name = getattr(event.key, "name", event.key)
        if name in ("Escape", "q"):
            self.win.close()
        elif name == " ":
            self.toggle_play()
        elif name == "n":
            self.playing = False
            self.btn.setText("Start")
            self._step(1)
        elif name == "b":
            self.playing = False
            self.btn.setText("Start")
            self._step(-1)
        elif name == "N":
            self._step_sweep(1)
        elif name == "B":
            self._step_sweep(-1)
        elif name == "c":
            modes = ["height", "intensity", "semseg"]
            self.combo.setCurrentText(modes[(modes.index(self.color_mode) + 1) % 3])

    def _step(self, d):
        n = len(self.sweep)
        if n:
            self.frame = (self.frame + d) % n
            self._dirty = True
            self._prefetch()

    def _step_sweep(self, d):
        self.sweep_idx = (self.sweep_idx + d) % len(self.ds)
        self.sweep = self.ds[self.sweep_idx]
        self.frame = 0
        self._update_sweep_meta()
        self._prefetch()

    # ---------------- frame loop ----------------
    def _tick(self, event):
        now = time.time()
        if self.playing and len(self.sweep) and now - self._last_advance >= 1.0 / self.data_fps:
            self._last_advance = now
            self.frame = (self.frame + 1) % len(self.sweep)
            self._dirty = True
        self._prefetch()
        if self._dirty:
            self._render()

    # ---------------- rendering (GPU visuals only) ----------------
    def _render(self):
        self._dirty = False
        pl = self._get(self.sweep, self.frame)
        if pl is None:
            return
        if "error" in pl:
            self.status.showMessage(f"frame load error: {pl['error']}")
            return
        pts = pl["pts"]
        # --- 3D scene ---
        if self.color_mode == "intensity" and len(pl["intensity"]):
            v = pl["intensity"].astype(np.float32) / 255.0
            colors = (np.stack([v, v, v], 1) * 255).astype(np.uint8)
        elif self.color_mode == "semseg" and pl["semseg"] is not None:
            palette = np.zeros((256, 3), np.float32)
            for cid in range(256):
                bgr = SEMSEG_PALETTE.get(cid, ((cid * 47) % 255, (cid * 91) % 255, (cid * 137) % 255))
                palette[cid] = np.array(bgr, np.float32) / 255.0
            colors = fastops.semseg_colors(pl["semseg"].astype(np.uint8), palette)[:, ::-1]
        else:
            colors = z_colors(pts[:, 2]) if len(pts) else np.zeros((0, 3), np.uint8)
        if len(pts):
            self.v3d_points.set_data(pts, face_color=colors.astype(np.float32) / 255.0,
                                     edge_color=None, size=2, edge_width=0)
        else:
            self.v3d_points.set_data(np.zeros((0, 3), np.float32))
        boxes = pl["boxes"]
        labels = pl["labels"]
        if len(boxes):
            self.v3d_boxes.set_data(fastops.box_lines(boxes), color=(0, 1, 0, 1))
            self.v3d_arrows.set_data(self._arrows(boxes), color=(1, 0, 0, 1))
        else:
            self.v3d_boxes.set_data(np.zeros((0, 3), np.float32))
            self.v3d_arrows.set_data(np.zeros((0, 3), np.float32))
        self._render_labels3d(boxes, labels)

        # --- speedometer ---
        spd = pl["gps"].get("speed", float("nan")) * 3.6
        self.speedo.setText(f" {spd:5.1f} km/h ")

        # --- cameras ---
        zc = z_colors(pts[:, 2]) if len(pts) else None  # projection colored by Z
        for cam, view in self.cam_views.items():
            cd = pl["cams"].get(cam)
            if cd is None:
                view["img"].set_data(np.zeros((2, 2, 3), np.uint8))
                view["pts"].set_data(np.zeros((0, 2)), size=1)
                view["box"].set_data(np.zeros((0, 2)))
                self._render_labels2d(view, None, None, 0, 0)
                continue
            img = cd["img"]
            h, w = img.shape[:2]
            view["img"].set_data(np.ascontiguousarray(img[::-1]))
            if view.get("w") != w:
                view["w"], view["h"] = w, h
                view["vb"].camera.set_range(x=(0, w), y=(0, h), margin=0)
                view["title"].pos = (4, 4)
            R, t, K = cd["R"], cd["t"], cd["K"]
            pc = (R @ pts.T).T + t if len(pts) else np.zeros((0, 3))
            m = pc[:, 2] > 0.5
            pc = pc[m]
            if self.show_lidar_proj and len(pc):
                cc = zc[m] if zc is not None else None
                uv = pc[:, :2] / pc[:, 2:3]
                inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                uv = uv[inb]
                # image row v maps to scene y = h - v (image is y-flipped)
                pos = np.stack([uv[:, 0], h - uv[:, 1]], 1).astype(np.float32)
                view["pts"].set_data(pos, face_color=(cc[inb].astype(np.float32) / 255.0
                                                      if cc is not None else (1, 1, 1, 1)),
                                     edge_color=None, size=1.6, edge_width=0)
            else:
                view["pts"].set_data(np.zeros((0, 2)), size=1)
            if self.show_box_proj and len(boxes):
                self._render_boxes2d(view, boxes, labels, R, t, K, w, h)
            else:
                view["box"].set_data(np.zeros((0, 2)))
                self._render_labels2d(view, None, None, w, h)

        # --- map ---
        if self.osm is not None:
            gps = pl["gps"]
            self.osm.update(gps.get("lat", 0.0), gps.get("long", 0.0))
            if time.time() - self._map_last > 0.5:
                self._map_last = time.time()
                mimg = self.osm.render(gps.get("lat", 0.0), gps.get("long", 0.0))
                mh, mw = mimg.shape[:2]
                self.map_img.set_data(np.ascontiguousarray(mimg[::-1]))
                self.map_view.camera.set_range(x=(0, mw), y=(0, mh), margin=0)

        # --- frame label / slider ---
        n = len(self.sweep)
        self.frame_lbl.setText(f"f {self.frame + 1}/{n}")
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame)
        self.slider.blockSignals(False)

    def _arrows(self, boxes):
        n = boxes.shape[0]
        out = np.zeros((n * 2, 3), np.float32)
        hl = boxes[:, 3] / 2.0
        c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
        out[0::2] = boxes[:, :3]
        out[1::2, 0] = boxes[:, 0] + hl * c
        out[1::2, 1] = boxes[:, 1] + hl * s
        out[1::2, 2] = boxes[:, 2]
        return out

    def _render_labels3d(self, boxes, labels):
        # zoom-adaptive font size (turntable scale_factor)
        cam = self.view3d.camera
        if self._zoom_init["3d"] is None and getattr(cam, "scale_factor", None):
            self._zoom_init["3d"] = cam.scale_factor
        z0 = self._zoom_init["3d"] or 1.0
        zoom = (cam.scale_factor / z0) if getattr(cam, "scale_factor", None) else 1.0
        fs = float(np.clip(9 * zoom, 6, 48))
        n = min(len(boxes), MAX_LABELS_PER_VIEW)
        for i, tv in enumerate(self.labels3d):
            if i < n:
                b = boxes[i]
                tv.text = str(labels[i])
                tv.pos = (b[0], b[1], b[2] + b[5] / 2 + 0.4)
                tv.font_size = fs
                tv.visible = True
            else:
                tv.visible = False

    def _render_boxes2d(self, view, boxes, labels, R, t, K, w, h):
        corners = fastops.box_corners_batch(boxes.astype(np.float32)).astype(np.float64)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        segs = []
        label_pos = {}
        for k in range(corners.shape[0]):
            cc = (R @ corners[k].T).T + t
            if cc[:, 2].min() < 0.2:
                continue
            uv = (K @ cc.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            for a_, b_ in edges:
                segs.append((uv[a_, 0], h - uv[a_, 1]))
                segs.append((uv[b_, 0], h - uv[b_, 1]))
            top_i = int(np.argmax(uv[:, 1]))
            label_pos[k] = (uv[top_i, 0], h - uv[top_i, 1])
        if segs:
            view["box"].set_data(np.array(segs, np.float32), color=(0, 1, 0, 1))
        else:
            view["box"].set_data(np.zeros((0, 2)))
        self._render_labels2d(view, boxes, labels, w, h, label_pos)

    def _render_labels2d(self, view, boxes, labels, w, h, label_pos=None):
        zoom = 1.0
        try:
            rect = view["vb"].camera.rect
            zoom = (w or 1) / max(rect.width, 1e-6)
        except Exception:  # noqa: BLE001
            pass
        fs = float(np.clip(9 * zoom, 6, 60))
        items = list(label_pos.items())[:MAX_LABELS_PER_VIEW] if label_pos else []
        for i, tv in enumerate(view["labels"]):
            if i < len(items):
                k, (u, v) = items[i]
                tv.text = str(labels[k])
                tv.pos = (u, v)
                tv.font_size = fs
                tv.visible = True
            else:
                tv.visible = False

    def run(self):
        from vispy import app as vispy_app
        vispy_app.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--seq", default=None)
    ap.add_argument("--fps", type=float, default=10.0, help="dataset frame rate")
    ap.add_argument("--no-map", action="store_true")
    args = ap.parse_args()

    v = Viewer(args.roots, seq=args.seq, data_fps=args.fps, no_map=args.no_map)
    v.run()


if __name__ == "__main__":
    main()
