#!/bin/bash
# Mount the server's dataset disks over NFS (run on the client, needs sudo).
set -e
SERVER=${SERVER:-192.168.0.1}
for d in hdd1 hdd2; do
  mkdir -p /mnt/server/$d
  if mountpoint -q /mnt/server/$d; then
    echo "/mnt/server/$d already mounted"
  else
    mount -t nfs -o ro,soft,timeo=50,retrans=3,nfsvers=4 $SERVER:/mnt/$d/datasets /mnt/server/$d
    echo "mounted $SERVER:/mnt/$d/datasets -> /mnt/server/$d"
  fi
done
df -h /mnt/server/hdd1 /mnt/server/hdd2
