#!/bin/bash
# Pipeline entrypoint: optionally mount server NFS exports, then run the command.
# Requires --privileged when NFS_SERVER is set.
set -e
if [ -n "${NFS_SERVER:-}" ]; then
  modprobe nfs 2>/dev/null || true
  modprobe nfsv4 2>/dev/null || true
  mkdir -p /mnt/hdd1/datasets /mnt/hdd2/datasets
  mountpoint -q /mnt/hdd1/datasets || \
    mount -t nfs -o rw,soft,timeo=50,retrans=3,nfsvers=4 \
      "$NFS_SERVER:/mnt/hdd1/datasets" /mnt/hdd1/datasets
  mountpoint -q /mnt/hdd2/datasets || \
    mount -t nfs -o rw,soft,timeo=50,retrans=3,nfsvers=4 \
      "$NFS_SERVER:/mnt/hdd2/datasets" /mnt/hdd2/datasets
  echo "NFS mounted rw: $(df -h /mnt/hdd1/datasets | tail -1)"
fi
if [ "$#" -eq 0 ]; then
  set -- python -m pandaset_pipe.download
fi
exec "$@"
