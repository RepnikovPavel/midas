#!/bin/bash
# Run the viewer container locally with X11 forwarding and in-container NFS mounts.
# Usage: run.sh [args for visualize.py, e.g. --seq 001 --fps 10]
# Env: NFS_SERVER (default 192.168.0.1)
set -e
IMG=pandaset-viz:latest
NAME=pandaset-viz

xhost +local:docker >/dev/null 2>&1 || true
docker rm -f $NAME 2>/dev/null || true
docker run --rm -it \
  --name $NAME \
  --net host \
  --privileged \
  -e DISPLAY="$DISPLAY" \
  -e NFS_SERVER="${NFS_SERVER:-192.168.0.1}" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
  --mount type=bind,src=/tmp/.X11-unix,target=/tmp/.X11-unix \
  $IMG python -m pandaset_pipe.visualize \
    --roots /mnt/server/hdd1/pandaset_npz /mnt/server/hdd2/pandaset_npz "$@"
