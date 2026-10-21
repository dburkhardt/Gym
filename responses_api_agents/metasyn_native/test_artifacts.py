# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from nemo_gym._checkpoint.control import ControlError
from nemo_gym.base_responses_api_model import CaptureStore
from responses_api_agents.metasyn_native import app
from responses_api_agents.metasyn_native.artifacts import assert_prior_attempts_settled, attempt_root


@pytest.mark.asyncio
async def test_real_child_isolates_repeats_and_refuses_unknown(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "rag/203").mkdir(parents=True)
    (data / "rag/203/prompt.txt").write_text("frozen prompt")
    refs = tmp_path / "refs"
    refs.write_text("reference")
    capture = tmp_path / "capture"
    capture.mkdir()
    runner = tmp_path / "child.py"
    runner.write_text("""import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--review-id');p.add_argument('--output-root');p.add_argument('--reference-file')
a=p.parse_args();out=Path(a.output_root)/a.review_id
identity=json.loads((out/'identity.json').read_text())
assert identity['review_id']==int(a.review_id)
assert not (out/'child-effect').exists()
(out/'child-effect').write_text(identity['lineage'])
(out/'evaluation.json').write_text(json.dumps({'status':'succeeded','metrics':{'inclusion_f1':0.8}}))
""")
    config = SimpleNamespace(
        output_root=tmp_path / "output",
        materialized_root=data,
        reference_file=refs,
        runner_script=runner,
        native_python=sys.executable,
        creator_source_root=data,
    )
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"model_call_capture_dir": str(capture)})
    monkeypatch.setattr(app, "scientific_assets", lambda root, review_id: {"review_id": review_id})
    monkeypatch.setattr(app, "creator_identity", lambda root: {"root": str(root), "files": {}})

    @asynccontextmanager
    async def no_model_bridge(*args, **kwargs):
        yield

    monkeypatch.setattr(app, "bridge", no_model_bridge)
    agent = SimpleNamespace(config=config)
    first, result = await app.MetaSynAgent._evaluate(agent, 203, "http://unused", "7-0")
    assert result["metrics"]["inclusion_f1"] == 0.8
    before = {p.name: p.read_bytes() for p in first.iterdir() if p.is_file()}
    assert_prior_attempts_settled(config.output_root, 203, "7-0-a1", capture)
    second, _ = await app.MetaSynAgent._evaluate(agent, 203, "http://unused", "7-1")
    assert first != second
    assert (first / "child-effect").read_text() == "7-0"
    assert (second / "child-effect").read_text() == "7-1"
    with pytest.raises(FileExistsError):
        await app.MetaSynAgent._evaluate(agent, 203, "http://unused", "7-0")
    assert before == {p.name: p.read_bytes() for p in first.iterdir() if p.is_file()}
    store = CaptureStore(capture, provider_dispatch_guard=True)
    store.write_dispatch("7-0", "unknown", {"provider_attempts": [{"outcome": "unknown"}]})
    with pytest.raises(ControlError, match="unresolved"):
        assert_prior_attempts_settled(config.output_root, 203, "7-0-a1", capture)
    assert not attempt_root(config.output_root, 203, "7-0-a1").exists()
    # Independent scientific repeat remains possible; prior unknown is preserved.
    assert_prior_attempts_settled(config.output_root, 203, "7-2", capture)
    third, _ = await app.MetaSynAgent._evaluate(agent, 203, "http://unused", "7-2")
    assert (third / "child-effect").read_text() == "7-2"
    with pytest.raises(ControlError, match="unresolved"):
        store.assert_dispatch_reconciled(["7-0"])
    with pytest.raises(ValueError, match="missing or changed"):
        assert_prior_attempts_settled(config.output_root, 203, "7-0-a1", tmp_path / "different-capture")


def test_incomplete_and_legacy_attempts_refuse(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    root = tmp_path / "outputs"
    legacy = root / "203"
    legacy.mkdir(parents=True)
    with pytest.raises(ValueError, match="Legacy"):
        assert_prior_attempts_settled(root, 203, "7-0-a1", capture)
    legacy.rmdir()
    attempt_root(root, 203, "7-0").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        assert_prior_attempts_settled(root, 203, "7-0-a1", capture)
