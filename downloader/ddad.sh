#!/bin/bash
# mkdir -p /mnt/nvme/datasets/ddad
# curl -L -o /mnt/nvme/datasets/ddad/ddad.zip https://www.kaggle.com/api/v1/datasets/download/artemmmtry/ddad-dense-depth-for-autonomous-driving

mkdir -p /mnt/nvme/datasets/ddad
wget -L -O /mnt/nvme/datasets/ddad/ddad.zip "https://www.kaggle.com/api/v1/datasets/download/artemmmtry/ddad-dense-depth-for-autonomous-driving"
