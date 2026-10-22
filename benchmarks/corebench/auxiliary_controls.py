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


def required_tool_contract(row, selections):
    """Opt-in strict tool closure; ordinary diagnostic observations remain unchanged."""
    command, binding = probe_contract(row, selections)
    command += "\nset -eu\n"
    for item in selections:
        executable = shlex.quote(item["executable"])
        command += (
            "resolved=$(command -v " + executable + ")\n"
            'test -f "$resolved" && test -x "$resolved"\n'
            'libraries=$(ldd "$resolved")\n'
            'printf "%s\n" "$libraries" | grep -q "not found" && exit 1\n'
            'printf "%s\n" "$libraries" | awk ' + shlex.quote("{for(i=1;i<=NF;i++) if($i ~ /^\\//) print $i}") + " "
            '| sort -u | while IFS= read -r library; do sha256sum "$library"; done\n'
        )
    binding = dict(
        binding, schema="corebench-required-tools-v1", command_sha256=hashlib.sha256(command.encode()).hexdigest()
    )
    binding.pop("contract_sha256")
    binding["contract_sha256"] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return command, binding


def validate_required_tools(stdout, selections):
    """Reject missing, failed, or unresolved dynamic tools; retain exact version/path text."""
    sections = stdout.split("auxiliary=")[1:]
    if len(sections) != len(selections):
        raise ValueError("Required tool observation count differs")
    for section, selection in zip(sections, selections, strict=True):
        if section.splitlines()[0] != selection["executable"]:
            raise ValueError("Required tool identity differs")
        if "availability=" in section or "not found" in section:
            raise ValueError("Required tool or library unavailable")
        if re.findall(r"^version_exit=(.*)$", section, re.M) != ["0"]:
            raise ValueError("Required tool version query failed")
        if re.findall(r"^ldd_exit=(.*)$", section, re.M) != ["0"]:
            raise ValueError("Required dynamic library query failed")
        if not re.search(r"^resolved=/[^\n]+$", section, re.M) or not re.search(r"^[0-9a-f]{64}  /", section, re.M):
            raise ValueError("Required executable identity missing")
    # Only loader ASLR addresses are nondeterministic; paths, hashes and versions remain exact.
    return re.sub(r"\(0x[0-9a-fA-F]+\)", "(ASLR)", stdout)


async def qualify_required_tools(row, selections, final_image, destination, *, cpus=1, memory_mb=1024):
    """Additional before/after evidence, never a replacement for qualify_images."""
    from benchmarks.corebench.image_controls import probe_image
    from nemo_gym.sandbox.lifecycle import _publish

    destination = Path(destination)
    command, binding = required_tool_contract(row, selections)
    before = await probe_image(row["published_image"], command, destination / "before", cpus=cpus, memory_mb=memory_mb)
    after = await probe_image(final_image, command, destination / "after", cpus=cpus, memory_mb=memory_mb)
    left = validate_required_tools(before["stdout"], selections)
    right = validate_required_tools(after["stdout"], selections)
    if left != right:
        raise ValueError("Required tool path/version/library state changed")
    receipt = dict(
        binding,
        before_image=before["image"],
        after_image=after["image"],
        normalized_stdout_sha256=hashlib.sha256(left.encode()).hexdigest(),
        required_tools_equal=True,
        scope="Selected tools only; separate image/source qualification required",
    )
    _publish(destination / "required-tools.json", receipt)
    return receipt


CREATOR_R_STARTUP_PROBE = r"""set -eu
command -v Rscript
sha256sum "$(command -v Rscript)"
Rscript -e 'cat("creator-r-startup-v1\n"); cat(R.version.string, "\n"); writeLines(.libPaths()); print(Sys.getenv(c("R_HOME", "R_LIBS", "R_LIBS_SITE", "R_LIBS_USER", "R_ENVIRON", "R_ENVIRON_USER", "R_PROFILE", "R_PROFILE_USER", "HOME"))); p <- installed.packages(); p <- p[order(p[, "Package"], p[, "LibPath"]), c("Package", "Version", "LibPath", "Built"), drop=FALSE]; write.table(p, stdout(), sep="\t", quote=TRUE, row.names=FALSE)'
"""


async def qualify_creator_r_startup(row, final_image, destination, *, cpus=1, memory_mb=1024):
    """Compare normal image R startup; vanilla inventories can omit site libraries.

    This supplements image/source qualification. It does not exercise project-local
    startup files, workflow inputs, or scientific execution. No packages are installed.
    """
    from benchmarks.corebench.image_controls import probe_image
    from nemo_gym.sandbox.lifecycle import _publish

    destination = Path(destination)
    before = await probe_image(
        row["published_image"], CREATOR_R_STARTUP_PROBE, destination / "before", cpus=cpus, memory_mb=memory_mb
    )
    after = await probe_image(
        final_image, CREATOR_R_STARTUP_PROBE, destination / "after", cpus=cpus, memory_mb=memory_mb
    )
    if "creator-r-startup-v1\n" not in before["stdout"] or before["stdout"] != after["stdout"]:
        raise ValueError("Creator R startup interpreter/library/package state changed or absent")
    receipt = {
        "schema": "corebench-creator-r-startup-v1",
        "capsule": row["id"],
        "before_image": before["image"],
        "after_image": after["image"],
        "source_manifest_sha256": hashlib.sha256(json.dumps(row["medium_file_manifest"], sort_keys=True).encode()).hexdigest(),
        "command_sha256": hashlib.sha256(CREATOR_R_STARTUP_PROBE.encode()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stdout_sha256": hashlib.sha256(before["stdout"].encode()).hexdigest(),
        "creator_r_startup_equal": True,
        "scope": "Normal image startup only; project startup, source isolation and scientific workflow remain separate",
    }
    _publish(destination / "creator-r-startup.json", receipt)
    return receipt
