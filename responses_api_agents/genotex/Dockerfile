# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.11.14-slim@sha256:c8271b1f627d0068857dce5b53e14a9558603b527e46f1f901722f935b786a39
RUN apt-get update && apt-get install -y --no-install-recommends bash coreutils build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 scikit-learn==1.5.2 statsmodels==0.14.4 sparse-lmm==1.4 biopython==1.85 tqdm==4.67.1
ENV PYTHONPATH=/data OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
RUN mkdir /work && chmod 1777 /work && ln -s /data/metadata /work/metadata
WORKDIR /work
