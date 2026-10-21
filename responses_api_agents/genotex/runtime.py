# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container execution and public-input staging, independent of the model transport."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shlex
import shutil
import uuid
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.scoring_workspace import scoring_workspace


SHELL_TOOL = {
    "type": "function",
    "name": "shell",
    "strict": True,
    "description": "Execute bash in the isolated analysis container. Outputs are truncated to 20000 characters; use files for large results.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
}


async def command(*args: str, timeout: int = 300) -> tuple[int, str]:
    """Execute trusted control/scorer code with shared process-group settlement."""
    provider = LocalProvider(concurrency=1)
    async with scoring_workspace(provider) as handle:
        result = await provider.exec(handle, shlex.join(args) + " 2>&1", timeout_s=timeout)
    if result.error_type == "timeout":
        raise TimeoutError(f"Native command exceeded {timeout}s; process group settled")
    return result.return_code, (result.stdout or "") + (result.stderr or "")


async def execution_identity(*, image: dict, grader_python: str, data_root: Path) -> dict:
    """Bind resumes to the actual container image and selected implementation files."""
    root = Path(__file__).resolve().parent
    files = [root / name for name in ("app.py", "runtime.py", "grade.py", "checkpoint.py")]
    files += [
        Path(inspect.getfile(inspect.unwrap(obj))).resolve()
        for obj in (AsyncSandbox, LocalProvider, _owned_sandbox, scoring_workspace)
    ]
    # Preserve the invocation path: resolving a venv symlink selects the base environment.
    interpreter = Path(shutil.which(grader_python) or grader_python).absolute()
    resolved_interpreter = interpreter.resolve()
    try:
        code, freeze = await command(
            str(interpreter),
            "-c",
            "import importlib.metadata as m,json,sys;print(json.dumps({'python_version':sys.version,'distributions':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))",
            timeout=60,
        )
    except TimeoutError as exc:
        raise RuntimeError("Grader runtime inspection timed out before inference") from exc
    if code:
        raise RuntimeError("Could not inspect selected grader dependency environment")
    scientific = data_root / "upstream"
    files += [p for p in scientific.rglob("*.py") if ".git" not in p.parts]

    return {
        "image_id": image["image_id"],
        "grader_environment": json.loads(freeze),
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "grader_interpreter": {
            "path": str(interpreter),
            "resolved_path": str(resolved_interpreter),
            "sha256": hashlib.sha256(resolved_interpreter.read_bytes()).hexdigest(),
        },
    }


def stage(data_root: Path, run_root: Path, task: dict) -> dict[str, Path]:
    """Only raw data, public helper tools, and domain knowledge enter the sandbox."""
    trait, condition = task["trait"], task.get("condition")
    for value in [trait, *(task["traits"]), *([condition] if condition else [])]:
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError("Unsafe trait identifier")
    source = data_root / "upstream"
    selected_roots = [source / "tools", data_root / "assets/metadata"]
    selected_roots += [data_root / "assets/input/GEO" / t for t in task["traits"]]
    selected_roots += [data_root / "assets/output/preprocess" / t for t in task["traits"]]
    selected_roots += [data_root / "assets/output/regress" / trait]
    for root in selected_roots:
        if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
            raise ValueError("Staged GenoTEX assets contain a symlink")
    paths = {name: run_root / name for name in ["public", "prediction", "reference", "grading", "workspace"]}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    public = paths["public"]
    shutil.copytree(source / "tools", public / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(data_root / "assets/metadata", public / "metadata")
    shutil.copytree(data_root / "assets/metadata", paths["grading"] / "metadata")
    for selected in task["traits"]:
        shutil.copytree(data_root / "assets/input/GEO" / selected, public / "input/GEO" / selected)
        shutil.copytree(
            data_root / "assets/output/preprocess" / selected, paths["reference"] / "preprocess" / selected
        )
    reference_name = f"significant_genes_condition_{condition}.json"
    target = paths["reference"] / "regress" / trait / reference_name
    target.parent.mkdir(parents=True)
    shutil.copy2(data_root / "assets/output/regress" / trait / reference_name, target)
    return paths


def task_prompt(task: dict) -> str:
    """Describe all three upstream stages and required artifact formats without answers."""
    return f"""Complete a GenoTEX gene-trait association analysis for trait {task["trait"]}, condition {task.get("condition")}.
Raw GEO inputs for all required traits {task["traits"]} are in /data/input/GEO.
Public domain knowledge and gene synonym mappings are in /data/metadata. Public helper code is in /data/tools and importable as tools. Reference notebooks and answers are deliberately unavailable.
Work in /work; save all graded artifacts in /output. Network is disabled. Use the shell tool to inspect data, write and execute Python, and debug your outputs.
Perform all stages: determine which cohorts are usable (gene expression and target trait available, no severe trait imbalance); preprocess available cohorts; select the largest usable cohort with the required Age/Gender covariate or, for a two-trait problem, the usable cohort pair with largest sample-size product and common gene regressors; perform statistical analysis to identify genes associated with the target trait accounting for the condition. Follow GenoTEX's methodology: normalize gene symbols, integrate samples, and use the published helper functions for sparse regression/Lasso and cohort selection. You may inspect the helper implementations and their signatures.
For EACH required trait write:
/output/preprocess/TRAIT/cohort_info.json: object keyed by cohort ID containing is_usable, is_gene_available, is_trait_available, is_available, is_biased, has_age, has_gender, sample_size.
/output/preprocess/TRAIT/gene_data/COHORT.csv: gene-symbol rows, sample-ID columns.
/output/preprocess/TRAIT/clinical_data/COHORT.csv: clinical-feature rows, sample-ID columns (target trait, Age, Gender as available).
/output/preprocess/TRAIT/COHORT.csv: linked table with sample-ID rows and gene/clinical columns, numeric values.
Unavailable or unusable cohorts must be recorded honestly; do not fabricate data.
For the requested analysis write /output/regress/{task["trait"]}/significant_genes_condition_{task.get("condition")}.json with significant_genes containing Variable (gene symbols ranked by absolute effect), Coefficient, and Absolute Coefficient arrays. Include cv_performance if computed. The final prose response is not graded; files are graded.
This is an end-to-end pilot with raw inputs, not an analysis-only test with supplied preprocessed data. Complete as much of the pipeline as possible within the execution budget."""


class Container:
    """Compatibility surface backed by shared attributed Docker lifecycle."""

    def __init__(self, *, image: str, paths: dict[str, Path], cpus: int, memory_gb: int, ttl_s: float = 7200):
        self.paths = paths
        self.lineage = uuid.uuid4().hex
        self.sandbox = AsyncSandbox(
            {
                "docker": {
                    "create": {
                        "network": "none",
                        "cap_drop": ["ALL"],
                        "security_opt": ["no-new-privileges"],
                        "pids_limit": 256,
                    }
                }
            },
            SandboxSpec(
                image=image,
                ttl_s=ttl_s + 180,
                workdir="/work",
                resources=SandboxResources(cpu=cpus, memory_mib=memory_gb * 1024),
                metadata={"workload": "gym-genotex", "run": self.lineage},
                provider_options={
                    "volumes": [
                        f"{paths['public']}:/data:ro",
                        f"{paths['prediction']}:/output",
                        f"{paths['workspace']}:/work",
                    ],
                    "run_args": ["--user", f"{os.getuid()}:{os.getgid()}"],
                },
            ),
        )
        self._ownership = None
        self._descriptor = None

    async def start(self) -> None:
        self._ownership = _owned_sandbox(self.sandbox, self.paths["workspace"].parent / "sandbox-lifecycle.json")
        await self._ownership.__aenter__()
        try:
            self._descriptor = await self.sandbox.serialize()
            check = await self.shell(
                "touch /output/.write-check /work/.write-check && rm /output/.write-check /work/.write-check"
            )
            if check["exit_code"]:
                raise RuntimeError("Container output directory is not writable")
        except BaseException as original:
            try:
                await self.stop()
            except BaseException as cleanup:
                original.add_note(f"Container cleanup failed: {cleanup}")
            raise

    async def inspect(self) -> dict:
        if not self._descriptor:
            raise RuntimeError("Container has no owned handle")
        code, output = await command("docker", "inspect", self._descriptor["sandbox_id"])
        if code:
            raise RuntimeError("Cannot inspect owned task container")
        item = json.loads(output)[0]
        return {
            "image_id": item["Image"],
            "network_mode": item["HostConfig"]["NetworkMode"],
            "mounts": item["Mounts"],
            "memory_bytes": item["HostConfig"]["Memory"],
            "nano_cpus": item["HostConfig"]["NanoCpus"],
            "ownership": self._descriptor,
        }

    async def shell(self, text: str, timeout: int = 300) -> dict:
        result = await self.sandbox.exec("bash -lc " + shlex.quote(text) + " 2>&1", timeout_s=timeout)
        if result.error_type == "timeout":
            raise TimeoutError(f"Shell command exceeded {timeout}s; scoped execution settled")
        if result.error_type:
            raise RuntimeError(f"Shared sandbox execution failed: {result.error_type}")
        return {"exit_code": result.return_code, "output": ((result.stdout or "") + (result.stderr or ""))[-20000:]}

    async def stop(self) -> None:
        if self._ownership is not None:
            ownership, self._ownership = self._ownership, None
            await ownership.__aexit__(None, None, None)


async def score(*, data_root: Path, run_root: Path, paths: dict[str, Path], task: dict, python: str) -> dict:
    """Run upstream grader as a separate host process; no scorer is in the model container."""
    result = run_root / "grading.json"
    try:
        code, output = await command(
            python,
            str(Path(__file__).with_name("grade.py")),
            "--upstream",
            str(data_root / "upstream"),
            "--staging",
            str(paths["grading"]),
            "--prediction",
            str(paths["prediction"]),
            "--reference",
            str(paths["reference"]),
            "--trait",
            task["trait"],
            "--condition",
            str(task.get("condition")),
            "--result",
            str(result),
            "--identity",
            str(run_root / "scorer-imports.json"),
            timeout=900,
        )
    except TimeoutError as exc:
        raise RuntimeError("Upstream grader timed out; no automatic regrade or scientific zero") from exc
    (run_root / "grading.log").write_text(output)
    if code or not result.exists():
        raise RuntimeError(f"Upstream grader failed (exit {code}); see {run_root / 'grading.log'}")
    return json.loads(result.read_text())
