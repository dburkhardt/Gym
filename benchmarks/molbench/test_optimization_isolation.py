# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys

from benchmarks.molbench import adapter


def test_optimization_child_ignores_parent_pythonpath(tmp_path, monkeypatch):
    poison = tmp_path / "parent-python-libraries"
    poison.mkdir()
    (poison / "json.py").write_text("raise RuntimeError('parent interpreter contamination')\n")
    (tmp_path / "optimization_worker.py").write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({'isolated': sys.flags.isolated, 'request': request}))\n"
    )
    monkeypatch.setattr(adapter, "__file__", str(tmp_path / "adapter.py"))
    monkeypatch.setattr(adapter, "modules", lambda repo: (None, None))
    monkeypatch.setenv("MOLBENCH_OPT_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(poison))
    meta = {"stratum": "mo-opt", "subtask": "qed"}
    result = adapter.score(tmp_path, meta, "CCO")
    assert result == {"isolated": 1, "request": {"meta": meta, "answer": "CCO"}}
