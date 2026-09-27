#!/bin/bash
# Container entrypoint: mount server NFS exports (LAN access to hdd1/hdd2),
# then run the requested command (default: the viewer).
# Requires --privileged (kernel nfs module load + mount).
set -e
SERVER=${NFS_SERVER:-192.168.0.1}

if [ "${SKIP_NFS_MOUNT:-0}" != "1" ]; then
  modprobe nfs 2>/dev/null || true
  modprobe nfsv4 2>/dev/null || true
  mkdir -p /mnt/server/hdd1 /mnt/server/hdd2
  mountpoint -q /mnt/server/hdd1 || \
    mount -t nfs -o ro,soft,timeo=50,retrans=3,nfsvers=4 \
      "$SERVER:/mnt/hdd1/datasets" /mnt/server/hdd1
  mountpoint -q /mnt/server/hdd2 || \
    mount -t nfs -o ro,soft,timeo=50,retrans=3,nfsvers=4 \
      "$SERVER:/mnt/hdd2/datasets" /mnt/server/hdd2
  echo "NFS mounted: $(df -h /mnt/server/hdd1 | tail -1)"
fi

if [ "$#" -eq 0 ]; then
  set -- python -m pandaset_pipe.visualize \
    --roots /mnt/server/hdd1/pandaset_npz /mnt/server/hdd2/pandaset_npz
fi
exec "$@"
