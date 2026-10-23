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
    installer = "RUN command -v tmux >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-upgrade --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*)"
    variant = row.get("harness_install_variant")
    if variant is not None:
        admitted_capsule = row["id"] == "capsule-6133093" or (
            row["id"] == "capsule-1906954"
            and row["published_image"]
            == (
                "registry.codeocean.com/published/fd033d1a-d09f-43fc-a1fe-2b8e348df846"
                "@sha256:86040eccc32964f6b99b2e1c4aac8d4c49bd55e4da16c68c7e14884042afda4a"
            )
        )
        admitted_capsule = admitted_capsule or (
            row["id"] == "capsule-1683542"
            and row["published_image"]
            == (
                "registry.codeocean.com/published/5515bfd9-5cbb-47c7-a913-c186272085bd"
                "@sha256:b162464289b35dd5db3253f7e2d66e4c31fc3e99163369041b02cf6ad1d9ee1b"
            )
        )
        admitted_capsule = admitted_capsule or (
            row["id"] == "capsule-3137115"
            and row["published_image"]
            == (
                "registry.codeocean.com/published/34ac6903-ae50-422b-8255-a2162d267de7"
                "@sha256:6cd58626442882ecd11c347dc77de4267499891837841b548b57ad7ac1e7caa5"
            )
        )
        admitted_capsule = admitted_capsule or (
            row["id"] == "capsule-4600160"
            and row["published_image"]
            == (
                "registry.codeocean.com/published/7e09306f-e019-480c-a09c-db23c6047085"
                "@sha256:1e861ca208612175f7571297690bf77eaea33f9ad5e72b1d2a7c3caa40741903"
            )
        )
        if variant != "ubuntu-bionic-isolated-v1" or not admitted_capsule:
            raise ValueError("Unsupported capsule-specific harness variant")
        from pathlib import Path

        helper_sha = hashlib.sha256(Path(__file__).with_name("harness_install.py").read_bytes()).hexdigest()
        installer = (
            "# harness_install_sha256=" + helper_sha + "\n"
            "COPY harness_install.py /opt/corebench-harness-install.py\n"
            "RUN /opt/corebench-harness/bin/python3 /opt/corebench-harness-install.py"
        )
    layout = ""
    if row.get("filesystem_layout_variant") is not None:
        if (
            row["filesystem_layout_variant"] != "8912293-relative-data-v1"
            or row["id"] != "capsule-8912293"
            or row["published_image"]
            != (
                "registry.codeocean.com/published/2a0bef51-a61e-4c00-82c5-c79023c606d3"
                "@sha256:e72a3d557a31640b2340714199641b1674925aa5039ceb3e36ef717126654fca"
            )
        ):
            raise ValueError("Unsupported capsule filesystem layout variant")
        # R resolves /code physically; retain the creator's relative paths.
        layout = (
            "RUN ln -s /workspace/capsule-8912293/data /workspace/data"
            " && ln -s /workspace/capsule-8912293/results /workspace/results\n"
        )
    from benchmarks.corebench.external_inputs import recipe as external_recipe

    external = external_recipe(row)
    env = "\n".join("ENV " + k + "=" + json.dumps(v) for k, v in row["runtime_environment_variant"].items())
    return f"""FROM {row["published_image"]}
COPY python /opt/corebench-harness
RUN /opt/corebench-harness/bin/python3 -m pip install numpy==1.26.4 scipy==1.14.1 openai==1.109.1 backoff==2.2.1 tqdm==4.67.1 asciinema==2.4.0
{installer}
RUN if ! command -v asciinema >/dev/null 2>&1; then printf '#!/bin/sh\\nexec /opt/corebench-harness/bin/python3 -m asciinema "$@"\\n' > /usr/local/bin/asciinema && chmod 755 /usr/local/bin/asciinema; fi
COPY capsule /workspace/{row["id"]}
RUN mkdir -p /workspace/{row["id"]}/results
RUN for name in code data results; do if [ -e "/$name" ] || [ -L "/$name" ]; then rmdir "/$name"; fi; ln -s "/workspace/{row["id"]}/$name" "/$name"; done
{env}
{layout}{external}WORKDIR /workspace
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


def stage_context(row, medium_root, reference_root, runtime_archive, destination, *, external_input_path=None):
    """Publish a clean checked context; staging locks stay outside COPY trees.

    Uses the shared trusted-parent/cooperating-writer lock convention. Original
    creator filenames, including legitimate .stage.lock files, are preserved.
    """
    import fcntl
    import os
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from nemo_gym.dataset_staging import _regular_fd, _sync_parent

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name("." + destination.name + ".stage.lock")
    with os.fdopen(_regular_fd(lock, os.O_CREAT | os.O_RDWR), "rb+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise ValueError("Refusing existing image context")
        with TemporaryDirectory(dir=destination.parent) as temporary:
            clean = Path(temporary) / "context"
            _stage_context_tree(row, medium_root, reference_root, runtime_archive, clean)
            from benchmarks.corebench.external_inputs import stage_files

            stage_files(row, external_input_path, clean)
            for path in clean.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
            directories = [clean, *(p for p in clean.rglob("*") if p.is_dir() and not p.is_symlink())]
            for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
                _sync_parent(directory / ".directory-sync")
            os.replace(clean, destination)
            _sync_parent(destination)
    return destination


def _stage_context_tree(row, medium_root, reference_root, runtime_archive, destination):
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
    with TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        counter = 0

        def publish_clean(src, dst, *, expected_sha256):
            nonlocal counter
            counter += 1
            staged = temporary / "cache" / str(counter)
            stage_file(src, staged, expected_sha256=expected_sha256)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, dst)

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
            publish_clean(src, dst, expected_sha256=f["sha256"])
            dst.chmod(src.stat().st_mode & 0o777)
        for src in (temporary / "python").rglob("*"):
            dst = destination / src.relative_to(temporary)
            if src.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(os.readlink(src))
            elif src.is_file():
                publish_clean(src, dst, expected_sha256=digest(src))
                dst.chmod(src.stat().st_mode & 0o777)
        if row.get("harness_install_variant"):
            helper = Path(__file__).with_name("harness_install.py")
            publish_clean(helper, destination / "harness_install.py", expected_sha256=digest(helper))
        recipe = temporary / "Dockerfile"
        recipe.write_text(dockerfile(row))
        publish_clean(recipe, destination / "Dockerfile", expected_sha256=digest(recipe))
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


async def probe_image(image, command, destination, *, cpus=2, memory_mb=4096, script=None):
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
        if script is not None:
            script_path = destination / "probe.py"
            script_path.write_text(script)
            await sandbox.upload(script_path, "/tmp/corebench-image-probe.py")
        result = await sandbox.exec("/bin/sh -c " + shlex.quote(command), timeout_s=120)
        (destination / "stdout.txt").write_text(result.stdout or "")
        (destination / "stderr.txt").write_text(result.stderr or "")
        _publish(
            destination / "probe.json",
            {
                "image": identity,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "script_sha256": hashlib.sha256(script.encode()).hexdigest() if script is not None else None,
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
    modes = {f["path"]: (source / f["path"]).stat().st_mode & 0o777 for f in row["medium_file_manifest"]}
    source_script = image_source_probe(row, modes, "/workspace/" + row["id"])
    observed_source = await probe_image(
        final_image,
        "/opt/corebench-harness/bin/python3 -I /tmp/corebench-image-probe.py",
        destination / "image-source",
        script=source_script,
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
