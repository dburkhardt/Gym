import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


pytest.importorskip("harbor")
from benchmarks.airsbench import verifier as module


@pytest.fixture
def setup_verifier(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    output = tmp_path / "verifier"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "manifest.json").write_text(json.dumps([{"source": "/workspace", "status": "ok"}]))
    environment = SimpleNamespace(
        download_dir_with_exclusions=AsyncMock(), upload_dir=AsyncMock(), exec=AsyncMock(), download_file=AsyncMock()
    )
    instance = module.AIRSVerifier(
        source_roots_json=json.dumps({"task": str(source)}),
        task=SimpleNamespace(
            config=SimpleNamespace(
                verifier=SimpleNamespace(environment_mode="separate"),
                environment=SimpleNamespace(docker_image="sha256:fixed"),
            ),
            paths=SimpleNamespace(task_dir=Path("task")),
        ),
        trial_paths=SimpleNamespace(verifier_dir=output, artifacts_dir=artifacts),
        environment=environment,
    )
    monkeypatch.setattr(module, "validate_inputs", lambda *args: {"image": "sha256:fixed", "test_predictions": 10000})
    monkeypatch.setattr(module, "scorer_identity", lambda *args: {"creator_revision": "fixed"})

    async def download(*, source_dir, target_dir, exclude):
        destination = target_dir
        assert exclude == ["data", "./data"]
        destination.mkdir(parents=True)
        (destination / "submission.csv").write_text("prediction\n1\n")

    environment.download_dir_with_exclusions.side_effect = download
    return instance, environment, output


async def test_native_verifier_uses_existing_environment_and_records_identity(setup_verifier):
    instance, environment, output = setup_verifier
    environment.exec.return_value = SimpleNamespace(return_code=0, stdout="", stderr="")

    async def metrics(_, destination):
        destination.write_text(json.dumps({"MAE": 0.25, "rows": 10000}))

    environment.download_file.side_effect = metrics
    result = await instance.verify()
    assert result.rewards == {"reward": 0.25, "raw_mae": 0.25}
    environment.exec.assert_awaited_once_with("python /tests/check.py", timeout_sec=300)
    assert (
        json.loads((output / "native-score/identity.json").read_text())["execution_boundary"]
        == "Harbor separate verifier environment"
    )
    with pytest.raises(FileExistsError):
        await instance.verify()


async def test_download_failure_is_not_missing_model_submission(setup_verifier):
    instance, environment, output = setup_verifier
    environment.download_dir_with_exclusions.side_effect = RuntimeError("transfer failed")
    with pytest.raises(RuntimeError, match="transfer failed"):
        await instance.verify()
    assert not (output / "native-score/failure.json").exists()
    environment.exec.assert_not_awaited()


async def test_complete_missing_submission_is_model_failure(setup_verifier):
    instance, environment, output = setup_verifier

    async def empty(*, source_dir, target_dir, exclude):
        destination = target_dir
        destination.mkdir(parents=True)

    environment.download_dir_with_exclusions.side_effect = empty
    with pytest.raises(module.ModelMissingSubmissionError, match="omitted submission"):
        await instance.verify()
    assert json.loads((output / "native-score/failure.json").read_text())["kind"] == "model_missing_submission"
    environment.exec.assert_not_awaited()


async def test_creator_failure_never_becomes_zero_score(setup_verifier):
    instance, environment, output = setup_verifier
    environment.exec.return_value = SimpleNamespace(return_code=1, stdout="", stderr="invalid predictions")
    with pytest.raises(RuntimeError, match="Creator scorer rejected"):
        await instance.verify()
    assert not (output / "native-score/score.json").exists()


async def test_failed_native_artifact_capture_is_not_model_failure(setup_verifier):
    instance, environment, output = setup_verifier
    (instance.trial_paths.artifacts_dir / "manifest.json").write_text(
        json.dumps([{"source": "/workspace", "status": "failed"}])
    )
    with pytest.raises(RuntimeError, match="Incomplete native workspace"):
        await instance.verify()
    environment.download_dir_with_exclusions.assert_not_awaited()
    assert not (output / "native-score/failure.json").exists()
