pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==26.2.*" "dask-cudf-cu12==26.2.*" "cuml-cu12==26.2.*" \
    "cugraph-cu12==26.2.*" "nx-cugraph-cu12==26.2.*" "cuxfilter-cu12==26.2.*" \
    "cucim-cu12==26.2.*" "pylibraft-cu12==26.2.*" "raft-dask-cu12==26.2.*" \
    "cuvs-cu12==26.2.*" "nx-cugraph-cu12==26.2.*"

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129