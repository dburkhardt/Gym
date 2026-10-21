# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest
from csv_state_server.app import SOURCE, CSVConfig, CSVResources


@pytest.mark.parametrize("fault", ["missing", "corrupt"])
async def test_snapshot_refusal_before_workspace_publication(tmp_path, fault):
    server = CSVResources.model_construct(config=CSVConfig.model_construct(workspace_root=tmp_path / "fresh"))
    files = {"source.csv": SOURCE, "selected.json": "[]"}
    state = {
        "files": files,
        "sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()},
    }
    if fault == "missing":
        files.pop("selected.json")
    else:
        files["selected.json"] = '[{"sample":"a","value":"999"}]'
    with pytest.raises(ValueError, match="snapshot"):
        await server.restore_session_states({"session": state})
    assert not (tmp_path / "fresh").exists()
    assert not server.sessions
