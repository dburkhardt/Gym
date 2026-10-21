# SPDX-License-Identifier: Apache-2.0
import json
import shutil
from types import SimpleNamespace

import pytest


pytest.importorskip("harbor")
from benchmarks.biabench import verifier as module


@pytest.mark.asyncio
@pytest.mark.parametrize("process_score", [None, 0.4])
async def test_native_hook_preserves_saved_evidence_and_separate_metrics(tmp_path, monkeypatch, process_score):
    workspace = tmp_path / "original"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output/result.txt").write_text("prediction")
    (workspace / "analysis.py").write_text("creator analysis")
    paths = SimpleNamespace(
        verifier_dir=tmp_path / "verifier", artifacts_dir=tmp_path / "artifacts", agent_dir=tmp_path / "agent"
    )
    paths.verifier_dir.mkdir()
    paths.artifacts_dir.mkdir()
    paths.agent_dir.mkdir()
    (paths.artifacts_dir / "manifest.json").write_text(json.dumps([{"source": "/workspace", "status": "ok"}]))
    (paths.agent_dir / "trajectory.json").write_text('{"steps":[]}')
    taskdir = tmp_path / "wound-healing-speed-kymograph-gigadb100118"
    taskdir.mkdir()
    (taskdir / "instruction.md").write_text("exact instruction")

    class Env:
        async def download_dir(self, source, dest):
            assert source == "/workspace"
            shutil.copytree(workspace, dest)

    seen = []

    async def score(**kwargs):
        seen.append(kwargs)
        assert (kwargs["submission"] / "artifacts/result.txt").read_text() == "prediction"
        assert (kwargs["submission"] / "logs/analysis.py").read_text() == "creator analysis"
        assert (kwargs["submission"] / "logs/trajectory.json").is_file()
        return {"native_scores": {"result_score": 0.79, "checklist_score": process_score}}

    monkeypatch.setattr(module, "score_managed", score)
    verifier = module.BIAVerifier(
        source=str(tmp_path),
        task_source=str(tmp_path),
        verifier_image="pinned",
        gym_model_api_base="http://gym/ng-rollout/task/v1",
        gym_model_name="model",
        task=SimpleNamespace(
            paths=SimpleNamespace(task_dir=taskdir),
            config=SimpleNamespace(verifier=SimpleNamespace(environment_mode="separate")),
        ),
        trial_paths=paths,
        environment=Env(),
    )
    result = await verifier.verify()
    assert result.rewards == (
        {"reward": 0.79, "outcome": 0.79} | ({"process": process_score} if process_score is not None else {})
    )
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_failed_capture_never_reaches_scoring(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps([{"source": "/workspace", "status": "failed"}]))
    verifier = module.BIAVerifier(
        source=str(tmp_path),
        task_source=str(tmp_path),
        verifier_image="pinned",
        gym_model_api_base="unused",
        gym_model_name="model",
        task=SimpleNamespace(config=SimpleNamespace(verifier=SimpleNamespace(environment_mode="separate"))),
        trial_paths=SimpleNamespace(artifacts_dir=tmp_path),
        environment=None,
    )
    with pytest.raises(RuntimeError, match="capture"):
        await verifier.verify()
