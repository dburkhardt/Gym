#!/usr/bin/env bash
set -euo pipefail
root=/home/ubuntu/projects/broad-eval-campaign/scivisagentbench
cd "$root"
# No image build, no full host project mount, no credentials, no network.
mkdir -p runtime-ram/isolation-output
chmod 777 runtime-ram/isolation-output
command=("$root/runtime-ram/env/bin/python" /control/mcp_control.py /source /workspace/isolation-control.json)
if (( $# )); then command=("$@"); fi
full_args=()
if [[ ${SCIVIS_FULL_CONTROL:-0} == 1 ]]; then full_args=(--mount "type=bind,src=$root/runtime-ram/assets,dst=/workspace/data/dataset_002,readonly" -e SCIVIS_FULL_CONTROL=1); fi
exec docker run --ulimit core=0 --rm --name scivis-isolation-control --network none --read-only --cap-drop ALL --security-opt no-new-privileges --user 1000:1000 --cpus 2 --memory "${SCIVIS_MEMORY:-2g}" "${full_args[@]}" --pids-limit 128 \
 --tmpfs /tmp:rw,nosuid,nodev,size=128m --tmpfs /home/control:rw,nosuid,nodev,size=32m,uid=1000,gid=1000 \
 --mount "type=bind,src=$root/runtime-ram/env,dst=$root/runtime-ram/env,readonly" \
 --mount "type=bind,src=$root/runtime-ram/sysroot,dst=$root/runtime-ram/sysroot,readonly" \
 --mount "type=bind,src=$root/runtime-ram/mcp-deps,dst=$root/runtime-ram/mcp-deps,readonly" \
 --mount "type=bind,src=$root/native-source,dst=/source,readonly" \
 --mount "type=bind,src=$root/mcp_control.py,dst=/control/mcp_control.py,readonly" \
 --mount "type=bind,src=$root/mcp_bootstrap.py,dst=/control/mcp_bootstrap.py,readonly" \
 --mount "type=bind,src=$root/runtime-ram/isolation-output,dst=/workspace" \
 --mount "type=bind,src=$root/runtime-ram/container-ld.so.cache,dst=/etc/ld.so.cache,readonly" \
 --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly \
 -e HOME=/home/control -e DISPLAY=:97 -e QT_API=pyside6 -e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe -e QT_XCB_GL_INTEGRATION=xcb_glx \
 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e LP_NUM_THREADS=2 -e SCIVIS_ISOLATION_CONTROL=1 \
 -e "LD_LIBRARY_PATH=$root/runtime-ram/sysroot/usr/lib/x86_64-linux-gnu:$root/runtime-ram/env/lib" \
 -e "ES2_LIBRARY=$root/runtime-ram/sysroot/usr/lib/x86_64-linux-gnu/libGLESv2.so.2" \
 -e "LIBGL_DRIVERS_PATH=$root/runtime-ram/sysroot/usr/lib/x86_64-linux-gnu/dri" \
 -e "PYTHONPATH=$root/runtime-ram/mcp-deps" -w /workspace \
 researchclaw-tools:terminal-v2 "${command[@]}"
