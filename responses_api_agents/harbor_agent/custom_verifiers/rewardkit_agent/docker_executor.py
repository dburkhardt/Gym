# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-side executor; only an explicitly labeled isolated judge sandbox."""

import json
import subprocess


def execute(
    container: str,
    command: str,
    timeout: float,
    expected_image: str,
    expected_role: str = "gym-rewardkit-judge-surrogate",
) -> dict:
    info = json.loads(subprocess.check_output(["docker", "inspect", container], text=True, timeout=15))[0]
    assert info["Config"]["Labels"].get("campaign.role") == expected_role
    assert expected_role in ("gym-rewardkit-judge-surrogate", "gym-rewardkit-judge-scientific")
    assert info["Image"] == expected_image
    assert info["HostConfig"]["NetworkMode"] == "none" and info["HostConfig"]["ReadonlyRootfs"]
    assert info["Config"]["User"] == "65534:65534"
    allowed = {"/app"} if expected_role == "gym-rewardkit-judge-surrogate" else {"/app", "/logs/agent/trajectory.json"}
    assert {m["Destination"] for m in info["Mounts"]} == allowed
    assert all(not m["RW"] for m in info["Mounts"])
    assert not any(
        x.split("=", 1)[0]
        in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "POLICY_API_KEY", "CODEX_ACCESS_TOKEN"}
        for x in info["Config"]["Env"]
    )
    # Timeout cleanup occurs INSIDE the sandbox so killing docker exec cannot orphan children.
    code = """import json,sys,subprocess,os,signal
req=json.loads(sys.argv[1]);p=subprocess.Popen(['/bin/sh','-lc',req['command']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
try:
 out,err=p.communicate(timeout=req['timeout']);r={'stdout':out,'stderr':err,'returncode':p.returncode}
except subprocess.TimeoutExpired:
 try:os.killpg(p.pid,signal.SIGKILL)
 except ProcessLookupError:pass
 out,err=p.communicate();r={'stdout':out,'stderr':err,'returncode':p.returncode,'timed_out':True,'reaped':True}
print(json.dumps(r))
"""
    raw = subprocess.check_output(
        ["docker", "exec", container, "python3", "-c", code, json.dumps({"command": command, "timeout": timeout})],
        text=True,
        timeout=timeout + 30,
    )
    return json.loads(raw)
