# SPDX-License-Identifier: Apache-2.0
"""Commands for bounded image qualification; never dispatch an LLM or install at run time."""

import hashlib
import json


# This command runs identically in creator and prepared images. Scientific
# interpreters keep their original PATH, executable and installed distributions.
SCIENTIFIC_PROBE = r"""set -eu
for command in python python2 python3 R Rscript; do
  if command -v "$command" >/dev/null 2>&1; then
    executable=$(command -v "$command")
    printf '%s\n' "$command:$executable"
    sha256sum "$executable"
    case "$command" in
      python*) "$executable" -c 'import sys; print(sys.version); print(sys.executable); print(sys.path)'; "$executable" -m pip freeze 2>/dev/null || true ;;
      R) "$executable" --version 2>&1 ;;
      Rscript) "$executable" --version 2>&1
        "$executable" --vanilla -e 'cat("R-library-paths\n"); writeLines(.libPaths()); p <- installed.packages(); p <- p[order(p[, "Package"], p[, "LibPath"]), c("Package", "Version", "LibPath", "Built"), drop=FALSE]; write.table(p, stdout(), sep="\t", quote=TRUE, row.names=FALSE)' ;;
    esac
  fi
done
"""

HARNESS_PROBE = r"""set -eu
/opt/corebench-harness/bin/python3 -c 'import numpy, scipy, openai, backoff, tqdm; print("harness-python-ok")'
tmux -V
asciinema --version
test ! -e /tests/answers.json
test ! -e /solution
test ! -e /workspace/answers.json
echo reference-isolation-ok
"""


def dockerfile(row):
    """Source-preserving overlay; image-specific controls remain mandatory.

    The provided python directory is the separately hash-verified recovered
    runtime. No scientific Python executable or global PATH is replaced.
    """
    env = "\n".join("ENV " + k + "=" + json.dumps(v) for k, v in row["runtime_environment_variant"].items())
    return f"""FROM {row["published_image"]}
COPY python /opt/corebench-harness
RUN /opt/corebench-harness/bin/python3 -m pip install numpy==1.26.4 scipy==1.14.1 openai==1.109.1 backoff==2.2.1 tqdm==4.67.1 asciinema==2.4.0
RUN command -v tmux >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-upgrade --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*)
RUN if ! command -v asciinema >/dev/null 2>&1; then printf '#!/bin/sh\\nexec /opt/corebench-harness/bin/python3 -m asciinema "$@"\\n' > /usr/local/bin/asciinema && chmod 755 /usr/local/bin/asciinema; fi
COPY capsule /workspace/{row["id"]}
RUN mkdir -p /workspace/{row["id"]}/results
RUN for name in code data results; do if [ -e "/$name" ] || [ -L "/$name" ]; then rmdir "/$name"; fi; ln -s "/workspace/{row["id"]}/$name" "/$name"; done
{env}
WORKDIR /workspace
"""


def recipe_digest(row):
    return hashlib.sha256(dockerfile(row).encode()).hexdigest()


def source_digest(row):
    return hashlib.sha256(json.dumps(row["medium_file_manifest"], sort_keys=True).encode()).hexdigest()


def qualification_digest(row):
    """Bind the checked task, recovered runtime and exact qualification probes.

    Historical receipts without this binding are not upgraded implicitly.
    """
    from pathlib import Path

    from benchmarks.corebench.prepare import RUNTIME_SHA256

    payload = {
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "schema": "corebench-image-qualification-v2",
        "recipe": recipe_digest(row),
        "source": source_digest(row),
        "instruction": row["instruction_sha256"],
        "reference": row["reference_sha256"],
        "scorer": row["scorer_sha256"],
        "harness_archive": RUNTIME_SHA256,
        "scientific_probe": SCIENTIFIC_PROBE,
        "harness_probe": HARNESS_PROBE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def stage_context(row, medium_root, reference_root, runtime_archive, destination):
    """Build context from preserved, checked sources; no Docker/network action."""
    import os
    import tarfile
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from benchmarks.corebench.prepare import RUNTIME_SHA256
    from benchmarks.corebench.prepare_frozen20 import digest, verify_sources
    from nemo_gym.dataset_staging import stage_file

    source, _ = verify_sources(row, medium_root, reference_root)
    if digest(runtime_archive) != RUNTIME_SHA256:
        raise ValueError("Recovered harness runtime hash differs")
    if destination.exists():
        raise ValueError("Refusing existing image context")
    with TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        with tarfile.open(runtime_archive) as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.parts[0] != "python"
                    or not (member.isfile() or member.isdir() or member.issym())
                ):
                    raise ValueError("Unsafe runtime member")
            archive.extractall(temporary, filter="data")
        for f in row["medium_file_manifest"]:
            src = source / f["path"]
            dst = destination / "capsule" / f["path"]
            stage_file(src, dst, expected_sha256=f["sha256"])
            dst.chmod(src.stat().st_mode & 0o777)
        for src in (temporary / "python").rglob("*"):
            dst = destination / src.relative_to(temporary)
            if src.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(os.readlink(src))
            elif src.is_file():
                stage_file(src, dst, expected_sha256=digest(src))
                dst.chmod(src.stat().st_mode & 0o777)
        recipe = temporary / "Dockerfile"
        recipe.write_text(dockerfile(row))
        stage_file(recipe, destination / "Dockerfile")
    return destination


def inspect_local_image(image):
    """Inspect only an already installed immutable image; never pull."""
    import re
    import subprocess

    if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image):
        raise ValueError("Immutable image required")
    result = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True, timeout=30, check=True
    )
    value = json.loads(result.stdout)[0]
    if value.get("Os") != "linux" or value.get("Architecture") != "amd64":
        raise ValueError("Qualified runtime requires Linux amd64")
    return {k: value[k] for k in ("Id", "RepoDigests", "Os", "Architecture")}


async def probe_image(image, command, destination, *, cpus=2, memory_mb=4096):
    """Run a bounded no-network probe through the existing owned sandbox path."""
    import shlex
    import uuid

    from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
    from nemo_gym.sandbox.lifecycle import _owned_sandbox, _publish

    identity = inspect_local_image(image)
    destination.mkdir(parents=True, exist_ok=False)
    sandbox = AsyncSandbox(
        {"docker": {"create": {"network": "none", "cap_drop": ["ALL"], "security_opt": ["no-new-privileges"]}}},
        SandboxSpec(
            image=identity["Id"],
            ttl_s=180,
            workdir="/tmp",
            resources=SandboxResources(cpu=cpus, memory_mib=memory_mb),
            metadata={"workload": "corebench-image-control", "run": uuid.uuid4().hex},
        ),
    )
    async with _owned_sandbox(sandbox, destination / "lifecycle.json"):
        result = await sandbox.exec("/bin/sh -c " + shlex.quote(command), timeout_s=120)
        (destination / "stdout.txt").write_text(result.stdout or "")
        (destination / "stderr.txt").write_text(result.stderr or "")
        _publish(
            destination / "probe.json",
            {
                "image": identity,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "return_code": result.return_code,
                "error_type": result.error_type,
            },
        )
        if result.return_code != 0 or result.error_type:
            raise ValueError("Image probe failed; retained probe/lifecycle evidence")
    if json.loads((destination / "lifecycle.json").read_text()).get("cleanup_confirmed") is not True:
        raise ValueError("Image probe cleanup unconfirmed")
    return {"image": identity, "stdout": result.stdout or ""}


async def qualify_images(row, medium_root, reference_root, runtime_archive, final_image, destination, *, resources):
    """Produce v2 observed image controls, not scientific or build reproducibility acceptance.

    The supplied final image is inspected and tested, not built here. The recipe
    digest identifies the expected staging recipe, not proof of its build history.
    """
    from benchmarks.corebench.prepare import RUNTIME_SHA256
    from benchmarks.corebench.prepare_frozen20 import digest, validate_qualification, verify_sources
    from nemo_gym.sandbox.lifecycle import _publish

    source, _ = verify_sources(row, medium_root, reference_root)
    if digest(runtime_archive) != RUNTIME_SHA256:
        raise ValueError("Recovered harness runtime hash differs")
    if any(
        type(resources.get(k)) is not int or resources[k] <= 0
        for k in ("cpus", "memory_mb", "storage_mb", "agent_timeout_sec")
    ):
        raise ValueError("Explicit positive resource limits required")
    if row["status"] in {"requires_cuda_no_cpu_substitution", "measured_oom_requires_resource_review"}:
        raise ValueError("This CPU qualification producer does not admit CUDA or observed OOM tasks")
    destination.mkdir(parents=True, exist_ok=False)
    kwargs = {"cpus": resources["cpus"], "memory_mb": resources["memory_mb"]}
    before = await probe_image(row["published_image"], SCIENTIFIC_PROBE, destination / "creator", **kwargs)
    after = await probe_image(final_image, SCIENTIFIC_PROBE, destination / "final", **kwargs)
    if not before["stdout"] or before["stdout"] != after["stdout"]:
        raise ValueError("Scientific interpreter/package state changed")
    harness = await probe_image(final_image, HARNESS_PROBE, destination / "harness", **kwargs)
    if "harness-python-ok" not in harness["stdout"] or "reference-isolation-ok" not in harness["stdout"]:
        raise ValueError("Harness probe markers absent")
    import shlex

    modes = {f["path"]: (source / f["path"]).stat().st_mode & 0o777 for f in row["medium_file_manifest"]}
    source_script = image_source_probe(row, modes, "/workspace/" + row["id"])
    observed_source = await probe_image(
        final_image,
        "/opt/corebench-harness/bin/python3 -c " + shlex.quote(source_script),
        destination / "image-source",
        **kwargs,
    )
    if "image-source-ok" not in observed_source["stdout"]:
        raise ValueError("Image source probe marker absent")
    if not (after["image"]["Id"] == harness["image"]["Id"] == observed_source["image"]["Id"]):
        raise ValueError("Final image identity changed between probes")
    receipt = {
        "capsule": row["id"],
        "creator_image": row["published_image"],
        "image_source_modes": modes,
        "image_source_verified": True,
        "creator_identity": before["image"],
        "final_image": after["image"]["Id"],
        "recipe_sha256": recipe_digest(row),
        "source_sha256": source_digest(row),
        "qualification_sha256": qualification_digest(row),
        "scientific_before": before["stdout"],
        "scientific_after": after["stdout"],
        "harness_python_ok": True,
        "tmux_ok": True,
        "asciinema_ok": True,
        "reference_isolation_ok": True,
        "cleanup_ok": True,
        "resources": resources,
        "scope": "Observed image/tool probes only; no scientific workflow, solver, or build-history attestation",
    }
    validate_qualification(row, receipt)
    _publish(destination / "qualification.json", receipt)
    return receipt


def image_source_probe(row, modes, root):
    """Exact read-only image content check; explicit errors survive optimization."""
    return "\n".join(
        [
            "import pathlib, hashlib, json",
            "root = pathlib.Path(" + repr(str(root)) + ")",
            "expected = " + repr({f["path"]: f["sha256"] for f in row["medium_file_manifest"]}),
            "modes = " + repr(modes),
            "actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}",
            "if actual != set(expected): raise ValueError('image source inventory differs')",
            "for name, checksum in expected.items():",
            " p = root / name",
            " if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != checksum: raise ValueError(name)",
            " if p.stat().st_mode & 0o777 != modes[name]: raise ValueError('mode: ' + name)",
            "print('image-source-ok')",
        ]
    )
