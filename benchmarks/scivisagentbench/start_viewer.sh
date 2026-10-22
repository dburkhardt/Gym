#!/bin/bash
# Runtime startup only; the creator viewer, data and callbacks remain unchanged.
set -euo pipefail
Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp > /tmp/xvfb.log 2>&1 &
xvfb_pid=$!
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
ready=0
for i in $(seq 1 100); do
  # Consume xdpyinfo completely: grep -q can cause SIGPIPE under pipefail.
  if xdpyinfo -display "$DISPLAY" 2>/dev/null | grep GLX >/dev/null; then ready=1; break; fi
  sleep 0.1
done
test "$ready" = 1
python /control/viewer.py
