FROM python:3.12-slim

# GL/X11 runtime libs for vispy (PyQt5 backend) + OpenCV GUI + NFS client
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglib2.0-0 libxkbcommon0 libdbus-1-3 \
        libfontconfig1 libx11-6 libxext6 libxrender1 libsm6 libice6 \
        libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-shape0 libxcb-sync1 libxcb-xfixes0 \
        libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 \
        libglfw3 \
        nfs-common kmod \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} \
        numpy pandas numba tqdm \
        vispy PyQt5 glfw \
        opencv-python-headless \
        smopy Pillow

WORKDIR /app
# build context = midas repo root
COPY pandaset_pipeline/src/pandaset_pipe /app/pandaset_pipe
COPY visualizer /opt/visualizer_pkg
RUN pip install --no-cache-dir /opt/visualizer_pkg
COPY pandaset_pipeline/docker/viz/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV PYTHONPATH=/app

ENTRYPOINT ["/entrypoint.sh"]
CMD []
