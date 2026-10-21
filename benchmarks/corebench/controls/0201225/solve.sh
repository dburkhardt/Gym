#!/bin/bash
set -eu
test ! -e /tests/answers.json
test ! -e /workspace/answers.json
test ! -S /var/run/docker.sock
# Refuse stale output; never delete historical evidence to make a run look fresh.
test ! -e /results/output.txt
test ! -L /results/output.txt
cat /proc/mounts > /workspace/scientific-mounts.txt
cat /proc/net/route > /workspace/scientific-routes.txt
cd /workspace/capsule-0201225
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-before.log
# /results is retained by task artifacts even if the original workflow fails.
cd /code
bash run > /workspace/scientific-workflow.log 2>&1
cd /workspace/capsule-0201225
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-after.log
/opt/corebench-harness/bin/python3 /solution/rendered_ecg.py /results/output.txt /workspace/report.json
