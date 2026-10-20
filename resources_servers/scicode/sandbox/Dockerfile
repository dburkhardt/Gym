# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
COPY requirements.txt /opt/scicode-requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/scicode-requirements.txt
WORKDIR /tmp
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib HOME=/tmp
USER 65534:65534
