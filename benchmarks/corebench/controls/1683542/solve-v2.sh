#!/bin/bash
set -eu
test ! -e /tests/answers.json
test ! -e /workspace/answers.json
test ! -S /var/run/docker.sock
# Refuse stale outputs, never delete historical evidence.
test -z "$(find /results -mindepth 1 -maxdepth 1 -print -quit)"
cat /proc/mounts > /workspace/scientific-mounts.txt
cat /proc/net/route > /workspace/scientific-routes.txt
cd /workspace/capsule-1683542
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-before.log
cd /code
bash run > /workspace/scientific-workflow.log 2>&1
cd /workspace/capsule-1683542
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-after.log
test -s /results/Figure.2.png
test -s /results/Figure.12.png
# /results is a symlink in the qualified image. Follow it and refuse an empty list.
find -L /results -type f -name '*.png' -print0 | sort -z > /workspace/scientific-figures.paths0
test -s /workspace/scientific-figures.paths0
xargs -0 sha256sum < /workspace/scientific-figures.paths0 > /workspace/scientific-figures.sha256
# No fabricated visual answer: retain ordinary evaluator behavior for an empty answer.
printf '{}\n' > /workspace/report.json
