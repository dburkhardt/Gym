# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real interpreter selection regression; no scientific execution or network."""

import json
import subprocess
import venv

import pytest

from responses_api_agents.genotex.runtime import execution_identity


@pytest.mark.asyncio
async def test_identity_probes_selected_venv_not_resolved_base(tmp_path):
    environment = tmp_path / "grader"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / "bin/python"
    purelib = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip()
    from pathlib import Path

    metadata = Path(purelib) / "genotex_identity_sentinel-0.0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: genotex-identity-sentinel\nVersion: 0.0.1\n")
    base = json.loads(
        subprocess.check_output(
            [
                str(python.resolve()),
                "-c",
                "import importlib.metadata as m,json; print(json.dumps([d.metadata['Name'] for d in m.distributions()]))",
            ],
            text=True,
        )
    )
    assert "genotex-identity-sentinel" not in base
    identity = await execution_identity(
        image={"image_id": "fixture-image"}, grader_python=str(python), data_root=tmp_path
    )
    assert ["genotex-identity-sentinel", "0.0.1"] in identity["grader_environment"]["distributions"]
    assert identity["grader_interpreter"]["path"] == str(python)
    assert identity["grader_interpreter"]["resolved_path"] == str(python.resolve())
