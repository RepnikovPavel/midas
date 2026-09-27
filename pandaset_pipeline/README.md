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

Контроль завершённости скачивания (после остановки контейнера):

```bash
ssh user@192.168.0.1 'docker run --rm \
  -v /home/user/pandaset_pipeline/src/pandaset_pipe:/app/pandaset_pipe:ro \
  --mount type=bind,src=/mnt/hdd1/datasets,target=/mnt/hdd1/datasets \
  --mount type=bind,src=/mnt/hdd2/datasets,target=/mnt/hdd2/datasets \
  pandaset-pipeline:latest python -m pandaset_pipe.verify \
  --zip-index /mnt/hdd1/datasets/pandaset_work/zip_index.json'
# raw: 74445 files, missing=0, badsize=0; npz: 103 seq, 8240 lidar frames → VERDICT: OK
```

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

## OSM-тайлы: предзакачка

```bash
# на сервере (один раз; ~1200 тайлов на весь датасет, idempotent):
ssh user@192.168.0.1 'docker run --rm \
  --mount type=bind,src=/mnt/hdd1/datasets,target=/mnt/hdd1/datasets \
  --mount type=bind,src=/mnt/hdd2/datasets,target=/mnt/hdd2/datasets \
  -v /home/user/pandaset_pipeline/src/pandaset_pipe:/app/pandaset_pipe:ro \
  -w /app -e PYTHONPATH=/app pandaset-pipeline:latest \
  python -m pandaset_pipe.osmtiles \
  --roots /mnt/hdd1/datasets/pandaset_npz /mnt/hdd2/datasets/pandaset_npz \
  --out /mnt/hdd1/datasets/pandaset_osm'
# -> <out>/<z>/<x>/<y>.png + plans/<seq>.json (план тайлов секвенции)
```

Проверка локализации карты по всем секвенциям (RMS подгонки GPS↔world +
«эго на дорожном пикселе» по палитре OSM): `pandaset_pipe.geocheck` —
**102/103 PASS** (004 — машина стояла, подгонка вырождена, карта не показывается).
Корневой баг локализации был в переводе метров ENU в градусы (радианы
складывались с градусами) — карта «отставала» от машины в 57 раз.

## Просмотр

### Web-визуализатор (рекомендуется)

```bash
bash docker/viz/build.sh
bash docker/viz/run_web.sh                 # контейнер сам монтирует NFS сервера
# открыть http://localhost:8777  (deep-link: /?seq=001&frame=40)
```

Тайлы OSM раздаются из предзакачки (`/api/tile`, ~3 мс с диска; нет сети в
рантайме), карта секвенции — один канвас на всю поездку с запасом (план из
`plans/<seq>.json`), дальше только перемещается по траектории — без пустых мест.

Возможности: 3D-облако (классическая jet-колорация по высоте / интенсивность /
дальность turbo / семсег), класс-цветные 3D-боксы с подписями и фильтрами классов
(кнопка «classes»), BEV-радар с зумом (колесо), паном (drag) и сбросом (dblclick),
подложка OSM в 3D-плоскости земли и в BEV (чекбоксы, выравнивание GPS↔world
подгонкой по треку), проекция точек и боксов на 6 камер (ячейки 2×3 под 16:9,
зум колесом к курсору), Leaflet-карта со всей траекторией поездки сразу
(fitBounds; клик по треку → переход к кадру), таймлайн профиля скорости,
скорость воспроизведения 0.5–10x, NMS on/off, range-clip и размер точек.
Горячие клавиши: `space` `n/b` (кадр) `N/B` (секвенция) `c` (режим цвета) `h` (справка).
Тайлы карты — OpenStreetMap (предзагружены на сервер, раздаются локально).
Транспорт кадра: точки fp16 (~1.3 МБ/кадр вместо 2.3), чтение в трэдпуле —
холодный кадр ~70-90 мс, кэш prefetch на 24 кадра.

Бэкенд: `pandaset_pipe.webserver` (aiohttp) — бинарный протокол кадра
(~2 МБ/кадр), `/api/frame /api/meta /api/gps_track /api/camimg`.

### Десктоп (PyQt5/vispy, X11)

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
