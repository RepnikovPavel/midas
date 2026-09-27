FROM python:3.12-slim

# pypi.org is unreachable from some server networks; use a reachable mirror.
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# numpy/pandas for conversion, numba for njit hot paths, tqdm for logs
# nfs-common+kmod: optional in-container NFS mounts (see entrypoint.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
        nfs-common kmod \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -i ${PIP_INDEX_URL} \
        numpy pandas numba tqdm "requests[socks]"

WORKDIR /app
COPY src/pandaset_pipe /app/pandaset_pipe
COPY docker/pipeline/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV PYTHONPATH=/app

ENTRYPOINT ["/entrypoint.sh"]
# default: full download+unpack+convert; override with docker run args
CMD ["python", "-m", "pandaset_pipe.download", "--streams", "24", "--chunk-mb", "16"]
