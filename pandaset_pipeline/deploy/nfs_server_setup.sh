#!/bin/bash
# Install + configure NFS exports of the two dataset disks (run ON the server via ssh sudo).
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nfs-kernel-server

grep -q '^/mnt/hdd1/datasets' /etc/exports || \
  echo '/mnt/hdd1/datasets 192.168.0.0/24(rw,sync,no_subtree_check,all_squash,anonuid=1000,anongid=1000)' >> /etc/exports
grep -q '^/mnt/hdd2/datasets' /etc/exports || \
  echo '/mnt/hdd2/datasets 192.168.0.0/24(rw,sync,no_subtree_check,all_squash,anonuid=1000,anongid=1000)' >> /etc/exports

mkdir -p /mnt/hdd1/datasets /mnt/hdd2/datasets
exportfs -ra
systemctl enable --now nfs-kernel-server

# open firewall for LAN clients if ufw is active
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  for net in 192.168.0.0/24 192.168.1.0/24; do
    ufw allow from $net to any port 2049 comment nfs || true
    ufw allow from $net to any port 111 comment rpcbind || true
  done
fi
exportfs -v
