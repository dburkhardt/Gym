# SPDX-License-Identifier: Apache-2.0
# Build with --pull=false and --build-arg BASE_IMAGE=<verified immutable image>.
# This recreates the historical Python judge subset, not every optional R/CLI handler.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git libcairo2 poppler-utils && rm -rf /var/lib/apt/lists/*
COPY responses_api_agents/promptbio_bench/scorer-requirements.txt /opt/promptbio/scorer-requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/promptbio/scorer-requirements.txt && python -m pip freeze > /opt/promptbio/installed-freeze.txt && dpkg-query -W > /opt/promptbio/system-packages.txt
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
