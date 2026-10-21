#!/bin/bash
set -eu
test ! -e /tests/answers.json
test ! -e /workspace/answers.json
test ! -S /var/run/docker.sock
cat /proc/mounts > /workspace/scientific-mounts.txt
cat /proc/net/route > /workspace/scientific-routes.txt
cd /workspace/capsule-1906954
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-before.log
# /results is included in task artifacts, retaining partial HTML on failure.
cd /code
bash run.sh > /workspace/scientific-workflow.log 2>&1
cd /workspace/capsule-1906954
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-after.log
/opt/corebench-harness/bin/python3 /solution/rendered_subgroups.py /results/vegan-meta.html /workspace/report.json
test -s /results/vegan-meta-pap.html
