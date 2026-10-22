#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu
test ! -e /tests/answers.json
test ! -e /workspace/answers.json
test ! -S /var/run/docker.sock
test -z "$(find -L /results -mindepth 1 -maxdepth 1 -print -quit)"
cat /proc/mounts > /workspace/scientific-mounts.txt
cat /proc/net/route > /workspace/scientific-routes.txt
cd /workspace/capsule-0238624
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-before.log
cd /code
# Run all 150 epochs on the original data and three frozen submodels.
# Preserve the creator's unseeded split and original training parameters.
bash run > /workspace/scientific-workflow.log 2>&1
cd /workspace/capsule-0238624
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-after.log
# This control establishes workflow execution, not model performance.
printf '{}\n' > /workspace/report.json
