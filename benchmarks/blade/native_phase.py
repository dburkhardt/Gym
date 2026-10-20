# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python3.11 native BLADE phase; HTTP-only bridge to the guarded Gym model service."""

import asyncio
import hashlib
import http.client
import inspect
import json
import os
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace


MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def publish(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def post_json(socket_path: str, payload: dict, headers: dict[str, str]) -> dict:
    class UnixConnection(http.client.HTTPConnection):
        def connect(self) -> None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(socket_path)

    connection = UnixConnection("gym-internal", timeout=600)
    try:
        connection.request("POST", "/chat/completions", body=json.dumps(payload).encode(), headers=headers)
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"Guarded model service returned HTTP {response.status}")
        return json.loads(body)
    finally:
        connection.close()


def response_outcome(result) -> dict:
    """Classify captured responses without inventing answers or scientific scores."""
    choices = result.choices
    reasons = [choice.finish_reason for choice in choices]
    if result.model != MODEL or not choices:
        kind = "response_contract_failure"
    elif "length" in reasons:
        kind = "model_output_budget_exhausted"
    elif any(reason != "stop" for reason in reasons):
        kind = "unsupported_response_termination"
    elif any(choice.message.content is None or not choice.message.content.strip() for choice in choices):
        kind = "model_no_visible_answer"
    else:
        kind = "visible_answer"
    return {
        "kind": kind,
        "response_captured": True,
        "finish_reasons": reasons,
        "usage": result.usage.model_dump() if result.usage else None,
        "creator_score": None,
        "scope": "call outcome only; task completion requires native analysis and verification",
    }


class Transport:
    """One physical request per native call; an uncertain failure poisons this phase."""

    def __init__(
        self, endpoint: str, lineage: str, digest: str, output: Path, *, socket_path: str, token: str
    ) -> None:
        if "/ng-rollout/" not in endpoint or not endpoint.endswith("/v1"):
            raise ValueError("Guarded Gym capture endpoint required")
        self.endpoint, self.lineage, self.digest = endpoint, lineage, digest
        self.socket_path, self.token = socket_path, token
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.lock = threading.Lock()
        self.ordinal = 0
        self.failed = False

    def create(self, **payload):
        from openai.types.chat import ChatCompletion

        with self.lock:
            if self.failed:
                raise RuntimeError("BLADE phase has an unusable prior call; no automatic resend")
            if payload.get("model") != MODEL:
                raise ValueError("Only exact Nemotron model is authorized")
            ordinal = self.ordinal
            self.ordinal += 1
            stem = self.output / f"{ordinal:06d}"
            self.failed = True  # Cleared only after durable complete response.
            publish(stem.with_suffix(".request.json"), payload)
            headers = {
                "Content-Type": "application/json",
                "x-nemo-gym-logical-call-id": f"blade:{self.lineage}:{ordinal}",
                "x-nemo-gym-call-config-sha256": self.digest,
            }
            headers["Authorization"] = "Bearer " + self.token
            response_captured = False
            try:
                body = post_json(self.socket_path, payload, headers)
                publish(stem.with_suffix(".response.json"), body)
                response_captured = True
                result = ChatCompletion.model_validate(body)
                publish(stem.with_suffix(".outcome.json"), response_outcome(result))
                if (
                    result.model != MODEL
                    or not result.choices
                    or any(c.finish_reason != "stop" for c in result.choices)
                ):
                    raise ValueError("Incomplete or wrong-model native response")
            except Exception as error:
                publish(
                    stem.with_suffix(".failure.json"),
                    {"type": type(error).__name__, "resend": False, "response_captured": response_captured},
                )
                # Deliberately not an OpenAI retry exception: creator backoff cannot resend.
                raise RuntimeError("BLADE native call unusable; inspect response and outcome receipts") from error
            self.failed = False
            return result


def redirect_native_caches(output: Path, manifest: dict) -> dict:
    """Relocate native eager caches; retain Cache behavior and native size limits."""
    import importlib

    relocated = {}
    for name in ("blade_bench.llms.base", "blade_bench.baselines.agent.react_agent", "blade_bench.eval.convert"):
        module = importlib.import_module(name)
        path = Path(inspect.getfile(module)).resolve()
        relative = str(path.relative_to("/opt/blade"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"].get(relative):
            raise ValueError("Native cache module source mismatch")
        target = output / "cache" / name.replace(".", "-")
        factory = module.Cache

        def owned_cache(directory=None, *args, _factory=factory, _target=target, **kwargs):
            return _factory(str(_target), *args, **kwargs)

        module.Cache = owned_cache
        relocated[name] = str(target)
    return relocated


def main() -> None:
    from blade_bench.baselines.config import EvalConfig, MultiRunConfig
    from blade_bench.baselines.multirun import RunLLMMultiRun
    from blade_bench.data.datamodel.transforms import TransformDataReturn
    from blade_bench.eval.evaluator import run_eval_on_analyses
    from blade_bench.llms.textgen_openai import OpenAITextGenerator

    # Native Jupyter output pickles this symbol under __main__, not function locals.
    globals()["TransformDataReturn"] = TransformDataReturn
    manifest = json.loads(Path("/controller/frozen_sources.json").read_text())
    imported = {}
    for obj in (OpenAITextGenerator, MultiRunConfig, EvalConfig, RunLLMMultiRun, run_eval_on_analyses):
        path = Path(inspect.getsourcefile(obj)).resolve()
        relative = str(path.relative_to("/opt/blade"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest["files"].get(relative) != digest:
            raise ValueError("Imported BLADE scientific source mismatch")
        imported[relative] = digest
    phase, dataset, output = sys.argv[1:]
    output = Path(output)
    publish(output / "imported-source.json", imported)
    publish(output / "cache-locations.json", redirect_native_caches(output, manifest))
    transport = Transport(
        os.environ["BLADE_ENDPOINT"],
        os.environ["BLADE_LINEAGE"],
        os.environ["BLADE_CONFIG_SHA256"],
        output / "calls",
        socket_path=os.environ["BLADE_SOCKET"],
        token=os.environ["BLADE_TOKEN"],
    )
    original = OpenAITextGenerator.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.client.close()
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=transport.create)))

    OpenAITextGenerator.__init__ = initialize
    llm = {
        "provider": "openai",
        "model": MODEL,
        "use_cache": False,
        "textgen_config": {"temperature": 0.1, "max_tokens": 16384},
    }
    if phase == "solver":
        config = MultiRunConfig(
            llm=llm,
            llm_eval=llm,
            output_dir=str(output / "solver"),
            run_dataset=dataset,
            use_agent=True,
            use_data_desc=True,
            num_runs=1,
            use_code_cache=False,
        )
        asyncio.run(RunLLMMultiRun(config).run())
    elif phase == "scorer":
        config = EvalConfig(
            multirun_load_path="/submitted/multirun_analyses.json",
            llm_eval=llm,
            output_dir=str(output / "scorer"),
            use_code_cache=False,
            diversity_ks=[],
        )
        run_eval_on_analyses(config)
    else:
        raise ValueError("Unknown native phase")
    if transport.failed:
        raise RuntimeError("Native library swallowed an unusable call; inspect outcome receipts")


if __name__ == "__main__":
    main()
