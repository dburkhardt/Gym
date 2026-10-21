#!/bin/bash
set -eu
test ! -e /tests/answers.json
test ! -e /workspace/answers.json
test ! -S /var/run/docker.sock
test -z "$(find -L /results -mindepth 1 -maxdepth 1 -print -quit)"
cat /proc/mounts > /workspace/scientific-mounts.txt
cat /proc/net/route > /workspace/scientific-routes.txt
cd /workspace/capsule-6746514
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-before.log
cd /code
# Preserve the creator's complete training and parallel figure schedule.
bash run > /workspace/scientific-workflow.log 2>&1
cd /workspace/capsule-6746514
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-after.log
/opt/corebench-harness/bin/python3 /solution/rendered_kfold.py /workspace/scientific-workflow.log /results /workspace/workflow-measurements.json
# This is a model-free operational control, not an answer to visual questions.
printf '{}\n' > /workspace/report.json
