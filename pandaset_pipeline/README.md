# pandaset_pipeline

Загрузка **полного** PandaSet (103 секвенции, ~41.5 GiB zip с HuggingFace
`georghess/pandaset`) на сервер с одновременной распаковкой, балансировкой по
двум дискам и конвертацией в npz (fp16 облака точек) на лету. Плюс ридер и
визуализатор. Все скрипты запускаются только в docker-контейнерах.

## Поток данных

```
HF zip (41.5 GiB)
  │  HTTP Range, per-sequence spans (секвенции в архиве идут подряд)
  ▼
download worker (N streams) ──► SpanExtractor (потоковая распаковка по мере скачивания)
  │                              raw: /mnt/hdd{1,2}/datasets/pandaset_raw/pandaset/<SEQ>/
  ▼
converter ──► npz sweep: /mnt/hdd{1,2}/datasets/pandaset_npz/sweep_<SEQ>/
```

- Секвенции распределяются по дискам жадным LPT по сжатому размеру → баланс байт.
- Состояние в `/work/state.json` → прерывания продолжаются с места обрыва
  (файлы-члены архива с корректным размером пропускаются).
- Kaggle-версия не используется — там неполный датасет.

## Система координат (converted)

Ego: **X вперёд, Y влево, Z вверх**. Исходный vehicle-фрейм pandaset — Y вперёд
(проверяется эмпирически `pandaset_pipe.validate` по GPS-скорости и дельтам поз).
Боксы: `x y z dx(длина) dy(ширина) dz(высота) yaw` — yaw CCW от ego +X,
dx вдоль направления (pandaset yaw=0 → +Y мира, length axis; см. docstring devkit).
Облака точек: fp16 (N,3) + intensity u8 + sensor_id u8 + rel_time f32.

Файлы кадра: `lidar_<ts>.npz`, `boxes_<ts>.npz` (+`nms_keep`), `semseg_<ts>.npz`,
`gps_<ts>.npz`, `<cam>_<ts>.jpg/.npz` (K, sensor2ego, ego2global),
`sweep_meta.json` (карта классов семсега).

## Сервер: запуск скачивания

```bash
# с клиентской машины (креды сервера только в env, не в репо):
SERVER_PASS=... bash deploy/server_setup.sh
# логи: ssh user@192.168.0.1 'docker logs -f pandaset-dl'
```

Образ: `docker/pipeline/Dockerfile` (python:3.12-slim + numpy/pandas/numba).
Валидация осей на сырых данных:

```bash
docker run --rm -v /mnt/hdd1/datasets:/mnt/hdd1/datasets pandaset-pipeline:latest \
  python -m pandaset_pipe.validate /mnt/hdd1/datasets/pandaset_raw/pandaset/001
```

## LAN-доступ к дискам (NFS)

```bash
# на сервере (один раз):
ssh user@192.168.0.1 'sudo bash -s' < deploy/nfs_server_setup.sh
# на клиенте:
sudo bash deploy/client_mount.sh     # /mnt/server/hdd{1,2}
```

## Просмотр (клиент, X11)

```bash
bash docker/viz/build.sh
bash docker/viz/run.sh --seq 001
# пробел=пауза n/b=кадр N/B=свип c=цвет(высота/интенсив/семсег) q=выход
```

Чтение без GUI:

```python
from pandaset_pipe.reader import PandaDataset
ds = PandaDataset(["/mnt/server/hdd1/datasets/pandaset_npz",
                   "/mnt/server/hdd2/datasets/pandaset_npz"])
snap = ds[0][0]
snap.points, snap.boxes_nms(), snap.semseg, snap.cameras["front_camera"], snap.gps
```
