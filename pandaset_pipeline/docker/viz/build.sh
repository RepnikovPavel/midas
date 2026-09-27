#!/bin/bash
# Build the pandaset viewer image (context = midas repo root, includes visualizer pkg).
set -e
cd "$(dirname "$0")/../../.."
docker build -t pandaset-viz:latest -f pandaset_pipeline/docker/viz/Dockerfile .
docker image ls pandaset-viz:latest
