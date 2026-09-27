#!/bin/bash
# Build the pandaset pipeline image. Run ON the server (see deploy/server_setup.sh).
set -e
cd "$(dirname "$0")/../.."
docker build -t pandaset-pipeline:latest -f docker/pipeline/Dockerfile .
docker image ls pandaset-pipeline:latest
