# SPDX-License-Identifier: Apache-2.0
"""Creator FlyAOC execution with separate solver and reference-bearing scorer mounts."""

import hashlib
import json
import re
import secrets
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.provider_dispatch import configuration_digest
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox, _publish
from nemo_gym.sandbox.model_bridge import bridge


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"
CONTROLLER = Path(__file__).parent.resolve()


def validate_assets(root: Path) -> dict:
    runtime = json.loads((CONTROLLER / "runtime_sources.json").read_text())
    frozen = json.loads((CONTROLLER / "frozen_sources.json").read_text())
    expected = {
        **{"src/" + k: v for k, v in runtime["source_files"].items()},
        **{"cache/corpus/" + k: v for k, v in runtime["corpus_files"].items()},
        **{"ontologies/" + k: v for k, v in frozen["ontology_sha256"].items()},
        **{
            "cache/hf/datasets--anonymous-042--flyaoc/snapshots/" + runtime["dataset_revision"] + "/ontologies/" + k: v
            for k, v in frozen["ontology_sha256"].items()
        },
    }
    for name, checksum in expected.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError("Pinned solver asset mismatch: " + name)
    for path in root.rglob("*"):
        if path.is_symlink() or (
            path.is_file() and not path.name.endswith(".stage.lock") and str(path.relative_to(root)) not in expected
        ):
            raise ValueError("Unexpected solver asset; references must remain outside solver view")
    return runtime


def spec(
    *,
    image: str,
    root: Path,
    output: Path,
    phase: str,
    lineage: str,
    request: Path | None = None,
    socket_dir: str | None = None,
    token: str | None = None,
    references: Path | None = None,
    predictions: Path | None = None,
) -> SandboxSpec:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("A qualified immutable runtime image ID is required")
    if phase not in ("solver", "scorer"):
        raise ValueError("Unknown phase")
    volumes = [f"{root.resolve()}:/assets:ro", f"{output.resolve()}:/output", f"{CONTROLLER}:/controller:ro"]
    env = {
        "PYTHONPATH": "/assets/src:/controller:/shared",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "FLYAOC_CACHE_DIR": "/assets/cache",
        "HF_HUB_CACHE": "/assets/cache/hf",
        "OPENAI_AGENTS_DISABLE_TRACING": "1",
    }
    if phase == "solver":
        if request is None or socket_dir is None or token is None:
            raise ValueError("Solver requires task-scoped request/bridge")
        transport = CONTROLLER.parents[1] / "nemo_gym/guarded_chat_transport.py"
        volumes += [
            f"{request.resolve()}:/request.json:ro",
            f"{socket_dir}:/model-socket:ro",
            f"{transport}:/shared/guarded_chat_transport.py:ro",
        ]
        env.update(
            OPENAI_API_KEY="task-local-placeholder", OPENAI_BASE_URL="http://gym-internal/v1", FLYAOC_TOKEN=token
        )
    else:
        if references is None or predictions is None:
            raise ValueError("Scorer requires explicit references and saved predictions")
        frozen = json.loads((CONTROLLER / "frozen_sources.json").read_text())
        if (
            references.is_symlink()
            or hashlib.sha256(references.read_bytes()).hexdigest() != frozen["reference_records_sha256"]
        ):
            raise ValueError("Reference checksum mismatch")
        volumes += [f"{references.resolve()}:/references.jsonl:ro", f"{predictions.resolve()}:/predictions.jsonl:ro"]
    return SandboxSpec(
        image=image,
        ttl_s=3720 if phase == "solver" else 720,
        resources=SandboxResources(cpu=2, memory_mib=4096),
        metadata={"workload": "gym-flyaoc-" + phase, "run": lineage},
        env=env,
        provider_options={"volumes": volumes},
    )


async def run_solver(*, image: str, root: Path, output: Path, payload: dict, endpoint: str, model: str = MODEL):
    runtime = validate_assets(root)
    if model != MODEL:
        raise ValueError("Only the campaign-authorized model is permitted")
    frozen = json.loads((CONTROLLER / "frozen_sources.json").read_text())
    if payload.get("gene_id") not in frozen["frozen_task_ids"] or set(payload) != {
        "gene_id",
        "gene_symbol",
        "summary",
        "pmcids",
    }:
        raise ValueError("Invalid frozen public gene input")
    if (
        hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        != runtime["public_input_sha256"][payload["gene_id"]]
    ):
        raise ValueError("Frozen gene input changed")
    output.mkdir(parents=True, exist_ok=False)
    lineage = uuid.uuid4().hex
    digest = configuration_digest(
        {
            "image": image,
            "guard_transport_sha256": hashlib.sha256(
                (CONTROLLER.parents[1] / "nemo_gym/guarded_chat_transport.py").read_bytes()
            ).hexdigest(),
            "runtime": runtime,
            "model": model,
            "paper_budget": 10,
            "max_turns": 50,
            "controller": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CONTROLLER.glob("*.py")},
        }
    )
    request = output / "request.json"
    _publish(
        request,
        {
            "input": payload,
            "lineage": lineage,
            "config_sha256": digest,
            "model": model,
            "restart": "existing attempt fails closed; no implicit tool-state recovery",
        },
    )
    with TemporaryDirectory(prefix="flyaoc-uds-", dir="/tmp") as sockets:
        token = secrets.token_urlsafe(32)
        async with bridge(
            Path(sockets) / "model.sock",
            endpoint=endpoint,
            token=token,
            lineage=lineage,
            digest=digest,
            namespace="flyaoc",
            expected_model=model,
        ):
            sandbox = AsyncSandbox(
                {"docker": {"create": {"network": "none"}}},
                spec(
                    image=image,
                    root=root,
                    output=output,
                    phase="solver",
                    lineage=lineage,
                    request=request,
                    socket_dir=sockets,
                    token=token,
                ),
            )
            async with _owned_sandbox(sandbox, output / "lifecycle.json"):
                result = await sandbox.exec("python /controller/native_phase.py", timeout_s=3600)
                (output / "stdout.log").write_text(result.stdout or "")
                (output / "stderr.log").write_text(result.stderr or "")
                if result.return_code != 0:
                    raise RuntimeError("Creator solver failed; preserve raw traces, no scientific zero")
    if not (output / "predictions.jsonl").is_file():
        raise RuntimeError("Creator prediction artifact missing")
    return output / "predictions.jsonl"


async def run_scorer(*, image: str, root: Path, output: Path, predictions: Path, references: Path):
    validate_assets(root)
    output.mkdir(parents=True, exist_ok=False)
    sandbox = AsyncSandbox(
        {"docker": {"create": {"network": "none"}}},
        spec(
            image=image,
            root=root,
            output=output,
            phase="scorer",
            lineage=uuid.uuid4().hex,
            predictions=predictions,
            references=references,
        ),
    )
    async with _owned_sandbox(sandbox, output / "lifecycle.json"):
        result = await sandbox.exec(
            "python /controller/regrade.py --source /assets/src --references /references.jsonl "
            "--ontologies /assets/ontologies --predictions /predictions.jsonl --output /output/evaluation.json",
            timeout_s=600,
        )
        (output / "stdout.log").write_text(result.stdout or "")
        (output / "stderr.log").write_text(result.stderr or "")
        if result.return_code != 0:
            raise RuntimeError("Creator scorer failed; no scientific score")
    return json.loads((output / "evaluation.json").read_text())
