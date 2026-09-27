"""Reader for the converted PandaSet npz sweep layout.

Supports merging several roots (e.g. NFS mounts of hdd1 and hdd2):

    ds = PandaDataset(["/mnt/server/hdd1/datasets/pandaset_npz",
                       "/mnt/server/hdd2/datasets/pandaset_npz"])
    sweep = ds[0]                 # Sweep
    snap = sweep[0]               # Snapshot (lazy)
    snap.points                   # (N,3) float32, ego frame X fwd / Y left
    snap.intensity                # (N,) uint8
    snap.semseg                   # (N,) uint8 or None
    snap.boxes                    # (M,7) float32  x y z dx dy dz yaw
    snap.box_labels               # (M,) str
    snap.cameras["front_camera"]  # {"image": BGR, "K": 3x3, "sensor2ego_*": ...}
    snap.gps                      # dict
    snap.ego2global               # (4,4) float32
"""

import bisect
import glob
import json
import os

import numpy as np

MAX_CAM_TIME_DIFF_MS = 60


def _quat_to_mat(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


class Snapshot:
    def __init__(self, ts, sweep_path, cam_index):
        self.ts = ts
        self.sweep_path = sweep_path
        self._cam_index = cam_index  # {cam: {"timestamps": [...], "jpg": [...], "npz": [...]}}
        self._lidar = None
        self._boxes = None
        self._semseg = None
        self._gps = None
        self._cameras = None

    def _npz(self, stem):
        path = os.path.join(self.sweep_path, f"{stem}_{self.ts}.npz")
        return path if os.path.exists(path) else None

    # ---- lidar ---------------------------------------------------------
    @property
    def lidar(self):
        if self._lidar is None:
            p = self._npz("lidar")
            if p is None:
                self._lidar = {}
            else:
                d = np.load(p)
                self._lidar = {
                    "points": d["points"].astype(np.float32),
                    "intensity": d["intensity"],
                    "sensor_id": d["sensor_id"],
                    "rel_time": d["rel_time"],
                    "ego2global_translation": d["ego2global_translation"],
                    "ego2global_rotation": d["ego2global_rotation"],
                }
        return self._lidar

    @property
    def points(self):
        return self.lidar.get("points", np.zeros((0, 3), np.float32))

    @property
    def intensity(self):
        return self.lidar.get("intensity", np.zeros(0, np.uint8))

    @property
    def sensor_id(self):
        return self.lidar.get("sensor_id", np.zeros(0, np.uint8))

    @property
    def ego2global(self):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = _quat_to_mat(self.lidar["ego2global_rotation"])
        T[:3, 3] = self.lidar["ego2global_translation"]
        return T

    # ---- boxes -----------------------------------------------------------
    @property
    def boxes_data(self):
        if self._boxes is None:
            p = self._npz("boxes")
            if p is None:
                self._boxes = {}
            else:
                d = np.load(p, allow_pickle=True)
                self._boxes = {k: d[k] for k in d.files}
        return self._boxes

    @property
    def boxes(self):
        return self.boxes_data.get("boxes", np.zeros((0, 7), np.float32))

    @property
    def box_labels(self):
        return self.boxes_data.get("class_names", np.array([], dtype=str))

    @property
    def box_uuids(self):
        return self.boxes_data.get("uuids", np.array([], dtype=str))

    @property
    def box_sensor_id(self):
        return self.boxes_data.get("sensor_id", np.zeros(0, np.int8))

    @property
    def nms_keep(self):
        k = self.boxes_data.get("nms_keep")
        if k is None:
            return np.ones(len(self.boxes), bool)
        return k.astype(bool)

    def boxes_nms(self):
        """(boxes, labels) filtered by the BEV NMS keep-mask."""
        m = self.nms_keep
        return self.boxes[m], self.box_labels[m]

    # ---- semseg ---------------------------------------------------------
    @property
    def semseg(self):
        if self._semseg is None:
            p = self._npz("semseg")
            self._semseg = np.load(p)["classes"] if p else None
        return self._semseg

    # ---- gps --------------------------------------------------------------
    @property
    def gps(self):
        if self._gps is None:
            p = self._npz("gps")
            if p is None:
                self._gps = {}
            else:
                d = np.load(p)
                self._gps = {k: float(d[k]) for k in d.files}
        return self._gps

    # ---- cameras ------------------------------------------------------------
    def _match_cam(self, cam):
        idx = self._cam_index.get(cam)
        if not idx or not idx["timestamps"]:
            return None
        ts_list = idx["timestamps"]
        i = bisect.bisect_left(ts_list, self.ts)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts_list):
                diff = abs(ts_list[j] - self.ts)
                if best is None or diff < best[0]:
                    best = (diff, j)
        if best and best[0] <= MAX_CAM_TIME_DIFF_MS:
            return idx["jpg"][best[1]], idx["npz"][best[1]]
        return None

    @property
    def cameras(self):
        if self._cameras is None:
            self._cameras = {}
            import cv2
            for cam in self._cam_index:
                m = self._match_cam(cam)
                if m is None:
                    continue
                jpg, npz_path = m
                img = cv2.imread(jpg)
                if img is None:
                    continue
                entry = {"image": img, "K": None, "jpg": jpg}
                if os.path.exists(npz_path):
                    d = np.load(npz_path)
                    entry["K"] = d["intrinsic"]
                    entry["sensor2ego_translation"] = d["sensor2ego_translation"]
                    entry["sensor2ego_rotation"] = d["sensor2ego_rotation"]
                self._cameras[cam] = entry
        return self._cameras

    def cam_projection(self, cam):
        """Return (K, R_cam_from_ego, t_cam_from_ego) for projecting ego points."""
        c = self.cameras.get(cam)
        if c is None or c.get("K") is None:
            return None
        R_s2e = _quat_to_mat(c["sensor2ego_rotation"])
        t_s2e = c["sensor2ego_translation"]
        # sensor2ego: p_ego = R p_cam + t  ->  cam_from_ego: p_cam = R.T (p_ego - t)
        return c["K"], R_s2e.T, (-R_s2e.T @ t_s2e)


class Sweep:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path.rstrip("/"))
        meta_path = os.path.join(path, "sweep_meta.json")
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
        self.semseg_classes = {int(k): v for k, v in
                               self.meta.get("semseg_classes", {}).items()}
        self._cam_index = self._build_cam_index()
        self._timestamps = self._build_lidar_index()

    def _build_cam_index(self):
        idx = {}
        for jpg in glob.glob(os.path.join(self.path, "*.jpg")):
            stem = os.path.basename(jpg)[:-4]
            cam, _, ts_str = stem.rpartition("_")
            try:
                ts = int(ts_str)
            except ValueError:
                continue
            e = idx.setdefault(cam, {"timestamps": [], "jpg": [], "npz": []})
            e["timestamps"].append(ts)
            e["jpg"].append(jpg)
            e["npz"].append(jpg[:-4] + ".npz")
        for e in idx.values():
            order = np.argsort(e["timestamps"])
            for k in ("timestamps", "jpg", "npz"):
                e[k] = [e[k][i] for i in order]
        return idx

    def _build_lidar_index(self):
        ts = []
        for f in glob.glob(os.path.join(self.path, "lidar_*.npz")):
            stem = os.path.basename(f)[len("lidar_"):-4]
            try:
                ts.append(int(stem))
            except ValueError:
                continue
        return sorted(ts)

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def camera_names(self):
        return sorted(self._cam_index.keys())

    def __len__(self):
        return len(self._timestamps)

    def __getitem__(self, i):
        return Snapshot(self._timestamps[i], self.path, self._cam_index)


class PandaDataset:
    """Dataset over one or more npz roots (merges sweep_* across roots)."""

    def __init__(self, roots):
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        self.roots = [str(r) for r in roots]
        self.sweep_paths = []
        for r in self.roots:
            for d in sorted(glob.glob(os.path.join(r, "sweep_*"))):
                if os.path.isdir(d):
                    self.sweep_paths.append(d)
        self.sweep_paths.sort(key=lambda p: os.path.basename(p))
        self._sweeps = {}

    def __len__(self):
        return len(self.sweep_paths)

    def __getitem__(self, i):
        if i not in self._sweeps:
            self._sweeps[i] = Sweep(self.sweep_paths[i])
        return self._sweeps[i]

    def sequence_names(self):
        return [os.path.basename(p).replace("sweep_", "") for p in self.sweep_paths]
