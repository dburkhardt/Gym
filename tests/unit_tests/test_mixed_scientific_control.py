# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "mixed_control", Path(__file__).parents[1] / "e2e/coordinator/mixed_scientific_restart_control.py"
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def test_exact_transcript_rejection_then_saved_attempt():
    mol = {"input": "molecular task", "model": control.MODEL}
    captures = [{"request": {"input": f"scientific step {i}"}, "response": {"output": i}} for i in range(2)]
    bodies = [
        mol,
        dict(captures[0]["request"], model=control.MODEL),
        dict(captures[0]["request"], model=control.MODEL),
        dict(captures[1]["request"], model=control.MODEL),
    ]
    replies = [control.fixture_reply(i, body, mol, {"output": "saved"}, captures) for i, body in enumerate(bodies, 1)]
    assert [status for status, _ in replies] == [200, 429, 200, 200]
    assert replies[2][1] is captures[0]["response"]
    assert replies[3][1] is captures[1]["response"]
    with pytest.raises(ValueError, match="extra"):
        control.fixture_reply(5, bodies[-1], mol, {}, captures)
    with pytest.raises(ValueError, match="mismatch"):
        control.fixture_reply(3, bodies[-1], mol, {}, captures)
    with pytest.raises(ValueError, match="mismatch"):
        control.fixture_reply(1, dict(mol, temperature=0), mol, {}, captures)


def test_all_invocations_share_absolute_deadline(monkeypatch):
    monkeypatch.setattr(control.time, "monotonic", lambda: 10)
    assert control.remaining(20) == 10
    monkeypatch.setattr(control.time, "monotonic", lambda: 19)
    assert control.remaining(20) == 1
    monkeypatch.setattr(control.time, "monotonic", lambda: 20)
    with pytest.raises(TimeoutError):
        control.remaining(20)


@pytest.mark.parametrize("message", ["Error: No such object: exact-owned", "error: no such object: exact-owned"])
def test_container_absence_case_variants(message):
    from types import SimpleNamespace

    control.require_container_absence("exact-owned", SimpleNamespace(returncode=1, stderr=message))


@pytest.mark.parametrize("code,message", [(0, ""), (1, "permission denied"), (1, "No such object: unrelated")])
def test_container_absence_refuses_unproven(code, message):
    from types import SimpleNamespace

    with pytest.raises(RuntimeError):
        control.require_container_absence("exact-owned", SimpleNamespace(returncode=code, stderr=message))
