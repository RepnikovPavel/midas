#!/bin/bash
# Run the WEB viewer locally: container mounts server NFS exports itself
# (no host mounts needed), serves the frontend + frame data on port 8777.
# Open: http://localhost:8777/?seq=001
# Env: NFS_SERVER (default 192.168.0.1), PORT (default 8777),
#      OSM_DIR (default /mnt/server/hdd1/pandaset_osm — predownloaded tiles)
set -e
IMG=pandaset-viz:latest
NAME=pandaset-web

docker rm -f $NAME 2>/dev/null || true
docker run -d --name $NAME \
  --privileged --net host \
  -e NFS_SERVER="${NFS_SERVER:-192.168.0.1}" \
  $IMG python -m pandaset_pipe.webserver \
    --roots /mnt/server/hdd1/pandaset_npz /mnt/server/hdd2/pandaset_npz \
    --osm-dir "${OSM_DIR:-/mnt/server/hdd1/pandaset_osm}" \
    --port "${PORT:-8777}"
echo "serving on http://localhost:${PORT:-8777} ; logs: docker logs -f $NAME"
