from pathlib import Path
import pytest
from benchmarks.corebench.saved_regrade import prepare_task


def make_task(tmp_path):
    task = tmp_path / "original"
    (task / "tests").mkdir(parents=True)
    (task / "tests/check.py").write_text('open("/workspace/report.json").read()')
    (task / "task.toml").write_text('artifacts = ["/workspace"]\n[verifier]\ntimeout_sec = 120\n')
    return task


def test_preserves_scorer_and_original_while_selecting_replay(tmp_path):
    source = make_task(tmp_path)
    before = (source / "task.toml").read_bytes()
    out = tmp_path / "replay"
    receipt = prepare_task(source, out)
    assert (source / "task.toml").read_bytes() == before
    assert (out / "tests/check.py").read_bytes() == (source / "tests/check.py").read_bytes()
    assert 'environment_mode = "separate"' in (out / "task.toml").read_text()
    assert receipt["new_solver_calls"] == 0


def test_rejects_oracle_solution(tmp_path):
    source = make_task(tmp_path)
    (source / "solution").mkdir()
    with pytest.raises(ValueError, match="solution"):
        prepare_task(source, tmp_path / "replay")


def test_requires_original_artifact_declaration(tmp_path):
    source = make_task(tmp_path)
    (source / "task.toml").write_text('[verifier]\ntimeout_sec=120\n')
    with pytest.raises(ValueError, match="workspace"):
        prepare_task(source, tmp_path / "replay")
