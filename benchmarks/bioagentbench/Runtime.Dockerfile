# SPDX-License-Identifier: Apache-2.0
# Required: an already qualified immutable Linux Python 3.13 base image digest.
# Build context must contain the exact upstream OpenCode 1.15.11 release archive.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG OPENCODE_SHA256
COPY opencode.tar.gz /opt/bioagent/opencode.tar.gz
RUN test -n "$OPENCODE_SHA256" && echo "$OPENCODE_SHA256  /opt/bioagent/opencode.tar.gz" | sha256sum -c - && mkdir -p /tmp/vf-opencode/bin && tar -xzf /opt/bioagent/opencode.tar.gz -C /tmp/vf-opencode/bin && test "$(/tmp/vf-opencode/bin/opencode --version)" = 1.15.11 && command -v python && command -v curl && command -v flock
# No apt/pip resolution here. Scientific downloads remain available at task runtime,
# as in the creator protocol; this image does not claim all scientific tools exist.
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
