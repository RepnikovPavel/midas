#!/bin/bash
# Deploy the pipeline to the server and start the download.
# Run from the client workstation. Requires: sshpass, rsync.
set -e
SERVER=${SERVER:-user@192.168.0.1}
SSH_OPTS="-o StrictHostKeyChecking=no -o BindInterface=eno1"
PASS=${SERVER_PASS:?set SERVER_PASS env}
SSHP="sshpass -p $PASS"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR=/home/user/pandaset_pipeline

echo "== rsync project to $SERVER =="
$SSHP rsync -az --delete -e "ssh $SSH_OPTS" \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"

echo "== build image on server =="
$SSHP ssh $SSH_OPTS $SERVER "cd $REMOTE_DIR && bash docker/pipeline/build.sh"

echo "== start downloader container =="
$SSHP ssh $SSH_OPTS $SERVER "mkdir -p /mnt/hdd1/datasets/pandaset_work && cd $REMOTE_DIR && bash docker/pipeline/run.sh"
