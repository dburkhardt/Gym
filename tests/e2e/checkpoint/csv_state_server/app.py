# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixture only: quiescent CSV-analysis files exported through native session hooks."""

import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

from fastapi import Request
from pydantic import Field

from nemo_gym.base_resources_server import BaseSeedSessionResponse, BaseVerifyResponse
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.example_session_state_mgmt.app import (
    IncrementCounterRequest,
    IncrementCounterResponse,
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
    StatefulCounterSeedSessionRequest,
    StatefulCounterVerifyRequest,
)


SOURCE = "sample,value\na,2\nb,4\nc,9\n"


class CSVConfig(StatefulCounterResourcesServerConfig):
    workspace_root: Path


class CSVResources(StatefulCounterResourcesServer):
    config: CSVConfig
    sessions: dict[str, str] = Field(default_factory=dict)

    def workspace(self, session):
        return self.config.workspace_root / hashlib.sha256(session.encode()).hexdigest()

    async def seed_session(self, request: Request, body: StatefulCounterSeedSessionRequest):
        session = request.session[SESSION_ID_KEY]
        if session not in self.sessions:
            path = self.workspace(session)
            path.mkdir(parents=True, exist_ok=False)
            (path / "source.csv").write_text(SOURCE)
            (path / "selected.json").write_text("[]")
            self.sessions[session] = str(path)
        return BaseSeedSessionResponse()

    async def increment_counter(self, request: Request, body: IncrementCounterRequest):
        path = self.workspace(request.session[SESSION_ID_KEY])
        rows = list(csv.DictReader(io.StringIO((path / "source.csv").read_text())))
        selected = json.loads((path / "selected.json").read_text())
        selected.extend(rows[len(selected) : len(selected) + body.count])
        (path / "selected.json").write_text(json.dumps(selected, sort_keys=True))
        return IncrementCounterResponse(success=True)

    async def export_session_states(self, session_ids):
        result = {}
        for session in session_ids:
            if session not in self.sessions:
                continue
            files = {name: (self.workspace(session) / name).read_text() for name in ("source.csv", "selected.json")}
            result[session] = {
                "files": files,
                "sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()},
            }
        return result

    async def restore_session_states(self, states):
        # Validate the complete batch before publishing any restored workspace.
        for state in states.values():
            files = state.get("files", {})
            if set(files) != {"source.csv", "selected.json"} or files["source.csv"] != SOURCE:
                raise ValueError("missing or incompatible CSV snapshot")
            if state.get("sha256") != {
                name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()
            }:
                raise ValueError("corrupt CSV snapshot")
            selected = json.loads(files["selected.json"])
            rows = list(csv.DictReader(io.StringIO(SOURCE)))
            if selected != rows[: len(selected)]:
                raise ValueError("invalid selected rows")
        for session, state in states.items():
            path = self.workspace(session)
            path.mkdir(parents=True, exist_ok=False)
            for name, value in state["files"].items():
                (path / name).write_text(value)
            self.sessions[session] = str(path)

    async def retire_session_state(self, session):
        self.sessions.pop(session, None)
        shutil.rmtree(self.workspace(session), ignore_errors=True)

    async def verify(self, request: Request, body: StatefulCounterVerifyRequest):
        path = self.workspace(request.session[SESSION_ID_KEY])
        selected = json.loads((path / "selected.json").read_text())
        mean = sum(float(row["value"]) for row in selected) / len(selected)
        (path / "metric.json").write_text(json.dumps({"mean": mean, "n": len(selected)}))
        retained = self.config.workspace_root.parent / "csv-result-artifacts" / path.name
        shutil.copytree(path, retained)
        return BaseVerifyResponse(**body.model_dump(), reward=float(mean == 5 and len(selected) == 3))


if __name__ == "__main__":
    CSVResources.run_webserver()
