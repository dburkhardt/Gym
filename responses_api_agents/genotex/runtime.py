# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container execution and public-input staging, independent of the model transport."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path


SHELL_TOOL = {
    "type": "function",
    "name": "shell",
    "description": "Execute bash in the isolated analysis container. Outputs are truncated to 20000 characters; use files for large results.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
}


async def command(*args: str, timeout: int = 300) -> tuple[int, str]:
    """Execute an argument vector, terminating the child on timeout."""
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout)
    except (TimeoutError, asyncio.CancelledError):
        process.kill()
        await process.wait()
        raise
    return process.returncode, output.decode(errors="replace")


def stage(data_root: Path, run_root: Path, task: dict) -> dict[str, Path]:
    """Only raw data, public helper tools, and domain knowledge enter the sandbox."""
    trait, condition = task["trait"], task.get("condition")
    for value in [trait, *(task["traits"]), *([condition] if condition else [])]:
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError("Unsafe trait identifier")
    source = data_root / "upstream"
    paths = {name: run_root / name for name in ["public", "prediction", "reference", "grading"]}
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
    """A per-task container with no credentials, references, network, or Docker socket."""

    def __init__(self, *, image: str, paths: dict[str, Path], cpus: int, memory_gb: int):
        self.name = "genotex-" + uuid.uuid4().hex
        self.image, self.paths, self.cpus, self.memory_gb = image, paths, cpus, memory_gb

    async def start(self) -> None:
        code, output = await command(
            "docker",
            "run",
            "--detach",
            "--name",
            self.name,
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--cpus",
            str(self.cpus),
            "--memory",
            f"{self.memory_gb}g",
            "--mount",
            f"type=bind,source={self.paths['public']},target=/data,readonly",
            "--mount",
            f"type=bind,source={self.paths['prediction']},target=/output",
            "--workdir",
            "/work",
            self.image,
            "sleep",
            "infinity",
        )
        if code:
            raise RuntimeError(f"Container startup failed: {output}")

    async def inspect(self) -> dict:
        """Record the resolved image and isolation controls without environment secrets."""
        code, output = await command("docker", "inspect", self.name)
        if code:
            raise RuntimeError("Cannot inspect task container")
        item = json.loads(output)[0]
        return {
            "image_id": item["Image"],
            "network_mode": item["HostConfig"]["NetworkMode"],
            "mounts": item["Mounts"],
            "memory_bytes": item["HostConfig"]["Memory"],
            "nano_cpus": item["HostConfig"]["NanoCpus"],
        }

    async def shell(self, text: str, timeout: int = 300) -> dict:
        code, output = await command(
            "docker",
            "exec",
            self.name,
            "timeout",
            "--kill-after=5",
            str(timeout),
            "bash",
            "-lc",
            text,
            timeout=timeout + 15,
        )
        return {"exit_code": code, "output": output[-20000:]}

    async def stop(self) -> None:
        await command("docker", "rm", "--force", self.name, timeout=60)


async def score(*, data_root: Path, run_root: Path, paths: dict[str, Path], task: dict, python: str) -> dict:
    """Run upstream grader as a separate host process; no scorer is in the model container."""
    result = run_root / "grading.json"
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
        timeout=900,
    )
    (run_root / "grading.log").write_text(output)
    if code or not result.exists():
        raise RuntimeError(f"Upstream grader failed (exit {code}); see {run_root / 'grading.log'}")
    return json.loads(result.read_text())
