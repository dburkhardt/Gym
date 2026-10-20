#!/bin/bash
set -eu
trap '/opt/corebench-harness/bin/python3 /tests/capture_artifacts.py || true' EXIT
/opt/corebench-harness/bin/python3 /tests/check.py
