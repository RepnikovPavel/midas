#!/bin/bash
# bash Algorithms/TransactFormer/docker/run_cu124.sh ./
# bash Algorithms/TransactFormer/docker/attach_cu124.sh
# cd /app/Algorithms/TransactFormer/ && python3 -m docker.system

if [ $# -eq 0 ]; then
    echo "Ошибка: необходимо указать путь для монтирования как аргумент скрипта"
    echo "Пример использования: $0 /путь/к/директории"
    exit 1
fi

MOUNT_PATH=$1
 
IMG_NAME=torchcu124:latest
CONTAINER_NAME=torchcu124container

xhost +local:docker

docker run --rm -it \
  -e USERNAME=$USERNAME \
  -e password=$password \
  --mount type=bind,src="$MOUNT_PATH",target=/app \
  --mount type=bind,src=/tmp/.X11-unix,target=/tmp/.X11-unix,consistency=cached \
  --mount type=bind,src=/mnt/nvme,target=/mnt/nvme \
  --net host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
  -e DISPLAY="$(echo $DISPLAY)" \
  --runtime=nvidia \
  --gpus all \
  --name $CONTAINER_NAME \
  --privileged \
  -p 64012:64012 \
  --detach \
  $IMG_NAME