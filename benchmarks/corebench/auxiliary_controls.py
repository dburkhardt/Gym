# SPDX-License-Identifier: Apache-2.0
"""Opt-in auxiliary executable observations; never scientific qualification."""

import hashlib
import json
import re
import shlex
from pathlib import Path


def probe_contract(row, selections):
    """Bind explicit operator selections to frozen source evidence, without guessing tools.

    Names use the creator PATH; absolute paths diagnose creator-installed tools
    outside PATH. Only version queries are admitted. Missing/broken tools remain
    recorded observations, never proof that a scientific workflow is ready.
    """
    if not selections:
        raise ValueError("Explicit auxiliary selections required")
    manifest = {item["path"]: item["sha256"] for item in row["medium_file_manifest"]}
    normalized = []
    seen = set()
    commands = ["set -u", "printf 'creator-PATH=%s\\n' \"$PATH\""]
    for item in selections:
        if set(item) != {"executable", "version_arg", "source_files"}:
            raise ValueError("Unknown auxiliary selection fields")
        executable = item["executable"]
        if not isinstance(executable, str) or not re.fullmatch(
            r"(?:/[A-Za-z0-9_.+-]+)+|[A-Za-z0-9_][A-Za-z0-9_+.-]*", executable
        ):
            raise ValueError("Invalid executable")
        if ".." in Path(executable).parts or executable in seen:
            raise ValueError("Duplicate or traversing executable")
        seen.add(executable)
        if item["version_arg"] not in {"--version", "-V"}:
            raise ValueError("Only bounded version observations supported")
        sources = item["source_files"]
        if (
            not isinstance(sources, dict)
            or not sources
            or any(
                k not in manifest or not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) or manifest[k] != v
                for k, v in sources.items()
            )
        ):
            raise ValueError("Auxiliary source binding differs from frozen manifest")
        normalized.append({"executable": executable, "version_arg": item["version_arg"], "source_files": sources})
        quoted = shlex.quote(executable)
        commands.extend(
            [
                "printf '%s\\n' " + shlex.quote("auxiliary=" + executable),
                "resolved=$(command -v " + quoted + " 2>/dev/null) || resolved=''",
                'if [ -z "$resolved" ] || [ ! -f "$resolved" ] || [ ! -x "$resolved" ]; then',
                "  printf 'availability=missing-or-not-executable\\n'",
                "else",
                '  printf "resolved=%s\\n" "$resolved"',
                '  sha256sum "$resolved" || exit 1',
                '  "$resolved" ' + shlex.quote(item["version_arg"]) + " 2>&1; status=$?",
                '  printf "version_exit=%s\\n" "$status"',
                '  if command -v ldd >/dev/null 2>&1; then ldd "$resolved" 2>&1; printf "ldd_exit=%s\\n" "$?"; else printf "ldd=unavailable\\n"; fi',
                "fi",
            ]
        )
    command = "\n".join(commands) + "\n"
    binding = {
        "schema": "corebench-auxiliary-observation-v1",
        "capsule": row["id"],
        "creator_image": row["published_image"],
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(row["medium_file_manifest"], sort_keys=True).encode()
        ).hexdigest(),
        "sandbox_probe_implementation_sha256": hashlib.sha256(
            Path(__file__).with_name("image_controls.py").read_bytes()
        ).hexdigest(),
        "selections": normalized,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
    }
    binding["contract_sha256"] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return command, binding


async def observe_auxiliaries(row, selections, destination, *, cpus=1, memory_mb=1024):
    """Use existing no-network, 120s execution/180s TTL owned sandbox machinery.

    No final-image qualification is emitted. A caller must review missing tools,
    nonzero version/ldd statuses and declared environment variants explicitly.
    """
    from benchmarks.corebench.image_controls import probe_image
    from nemo_gym.sandbox.lifecycle import _publish

    if type(cpus) is not int or type(memory_mb) is not int or cpus <= 0 or memory_mb <= 0:
        raise ValueError("Positive explicit resources required")
    command, binding = probe_contract(row, selections)
    result = await probe_image(row["published_image"], command, Path(destination), cpus=cpus, memory_mb=memory_mb)
    receipt = dict(
        binding,
        image=result["image"],
        stdout_sha256=hashlib.sha256(result["stdout"].encode()).hexdigest(),
        scope="Auxiliary runtime observations only; no scientific workflow or readiness acceptance",
    )
    _publish(Path(destination) / "auxiliary-observation.json", receipt)
    return receipt
