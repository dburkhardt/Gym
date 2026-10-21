#!/bin/bash
set -eu
test ! -e /tests/answers.json
test ! -e /workspace/answers.json
test ! -S /var/run/docker.sock
cat /proc/mounts > /workspace/scientific-mounts.txt
cat /proc/net/route > /workspace/scientific-routes.txt
cd /workspace/capsule-3894632
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-before.log
# Task artifacts include /results, preserving partial outputs on failure.
cd /code
bash run > /workspace/scientific-workflow.log 2>&1
cd /workspace/capsule-3894632
sha256sum -c /solution/source.sha256 > /workspace/scientific-source-after.log
test -s /results/correlation_results.csv
test -s /results/reduced-feature-set.csv
/opt/corebench-harness/bin/python3 /solution/rendered_survival.py /results/dp_survival.nb.html /workspace/report.json
# Cross-check from original inputs, without loading creator/evaluator answers.
Rscript --vanilla /solution/recompute.R > /workspace/scientific-recompute.log 2>&1
/opt/corebench-harness/bin/python3 - <<'PY'
import json
from pathlib import Path
observed = json.loads(Path('/workspace/report.json').read_text())['Report the p value for Lesions.Contoured.']
recomputed = float(Path('/workspace/recomputed-pfs-wald.txt').read_text())
if observed != recomputed:
    raise ValueError('Rendered PFS Wald p value differs from independent recomputation')
Path('/workspace/recomputation-review.json').write_text(json.dumps({'rendered': observed, 'recomputed': recomputed, 'equal': True, 'scope': 'source-defined standardized PFS Cox Wald p value, two significant digits'}))
PY
