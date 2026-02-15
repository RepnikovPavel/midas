# Функция для проверки установки nvidia-container-toolkit
check_nvidia_container_toolkit() {
    if command -v nvidia-ctk &> /dev/null; then
        echo "nvidia-container-toolkit установлен."
    else
        echo "Ошибка: nvidia-container-toolkit не установлен."
        echo "Установите его, выполнив следующие команды:"
        echo ""
        echo "1. Добавьте репозиторий NVIDIA:"
        echo "   distribution=\$(. /etc/os-release; echo \$ID\$VERSION_ID) \\"
        echo "   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \\"
        echo "   && curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
        echo ""
        echo "2. Установите nvidia-container-toolkit:"
        echo "   sudo apt-get update \\"
        echo "   && sudo apt-get install -y nvidia-container-toolkit"
        echo ""
        echo "3. Перезапустите Docker:"
        echo "   sudo systemctl restart docker"
        echo ""
        exit 1
    fi
}

check_nvidia_container_toolkit

DOCKER_BUILDKIT=1 docker buildx build -t modelscu124:latest -f docker/DockerFileCU124 --progress=plain . \
&& docker image ls | grep -E 'models|TAG|SIZE'
