#!/bin/bash
# docker/build.sh — РАБОТАЕТ БЕЗ SSL ошибок

check_nvidia_container_toolkit() {
    if command -v nvidia-ctk &> /dev/null; then
        echo "✅ nvidia-container-toolkit установлен."
    else
        echo "❌ Установите nvidia-container-toolkit"
        exit 1
    fi
}

check_nvidia_container_toolkit

echo "✅ modelscu124:latest найден локально"
echo "🚀 Строим ragcu124:latest с pip fallback..."

DOCKER_BUILDKIT=1 docker buildx build \
    --pull=false \
    -t torchcu124:latest \
    -f docker/DockerFile \
    --progress=plain \
    . \
&& echo "✅ Сборка завершена!" \
&& docker image ls torchcu124
