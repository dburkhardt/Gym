# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from benchmarks.blade.prepare import stage


SOURCE = Path("../../outputs/broad-eval-campaign/harbor/agent-expansion/sources/121/behavioral-data-BLADE-6118fa8")
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason="Pinned public BLADE source fixture unavailable")


def test_native_source_staging_and_reference_isolation(tmp_path):
    destination = tmp_path / "stage"
    output = stage(SOURCE, destination)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_id"] for row in rows] == ["blade::crofoot", "blade::panda_nuts", "blade::amtl"]
    for row in rows:
        dataset = row["verifier_metadata"]["dataset"]
        public = destination / "solver/blade_bench/datasets" / dataset
        assert {p.name for p in public.iterdir() if not p.name.startswith(".")} == {"data.csv", "info.json"}
        assert (public / "data.csv").read_bytes() == (
            SOURCE / "blade_bench/datasets" / dataset / "data.csv"
        ).read_bytes()
        assert row["verifier_metadata"]["native_info"] == json.loads((public / "info.json").read_text())
        assert (destination / "references/blade_bench/datasets" / dataset / "annotations.csv").is_file()
    example = destination / "solver/blade_bench/datasets/fertility"
    assert {p.name for p in example.iterdir() if not p.name.startswith(".")} == {"info.json"}
    assert (example / "info.json").read_bytes() == (SOURCE / "blade_bench/datasets/fertility/info.json").read_bytes()
    before = output.read_bytes()
    assert stage(SOURCE, destination).read_bytes() == before


def test_refuse_contaminated_solver_context(tmp_path):
    destination = tmp_path / "stage"
    stage(SOURCE, destination)
    leak = destination / "solver/blade_bench/datasets/crofoot/annotations.csv"
    leak.write_text("reference contamination")
    with pytest.raises(ValueError, match="Unexpected file"):
        stage(SOURCE, destination)
