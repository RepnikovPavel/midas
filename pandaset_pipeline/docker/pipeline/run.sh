#!/bin/bash
# Run the downloader+converter container ON the server.
# Both dataset disks + a workdir (state, zip index cache) are mounted.
# Direct download with CDN IP-pool rotation (PANDASET_PROXY optional fallback).
set -e
IMG=pandaset-pipeline:latest
NAME=pandaset-dl
docker rm -f $NAME 2>/dev/null || true
docker run -d --name $NAME \
  --restart unless-stopped \
  --net host \
  --cap-add SYS_PTRACE \
  ${PANDASET_PROXY:+-e PANDASET_PROXY="$PANDASET_PROXY"} \
  --mount type=bind,src=/mnt/hdd1/datasets,target=/mnt/hdd1/datasets \
  --mount type=bind,src=/mnt/hdd2/datasets,target=/mnt/hdd2/datasets \
  --mount type=bind,src=/mnt/hdd1/datasets/pandaset_work,target=/work \
  $IMG python -m pandaset_pipe.download "$@"
echo "started container $NAME; logs: docker logs -f $NAME"
