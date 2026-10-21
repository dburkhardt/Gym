# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned child runs pinned creator pipeline with one shared guarded judge transport."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import httpx

from nemo_gym.sandbox.guarded_chat_transport import GuardedSyncChatTransport, publish
from responses_api_agents.promptbio_bench.nemotron_proxy import MODEL, validate_result
from responses_api_agents.promptbio_bench.runtime import SCORER_REVISION


class ValidatedJudgeTransport(httpx.BaseTransport):
    """Keep the existing judge completion qualification over the shared guard."""

    def __init__(self, guarded):
        self.guarded = guarded
        self.failed = False

    def handle_request(self, request):
        if self.failed:
            raise RuntimeError("Prior judge failure; no further evaluator dispatch")
        self.failed = True
        response = self.guarded.handle_request(request)
        validate_result(response.json())
        self.failed = False
        return response

    def close(self):
        self.guarded.close()


def validate_mapping(mapping, *, reference_files, candidate_files):
    """Scorer safety variant: never compare a reference against itself via an invented path."""
    if mapping is None:
        return
    references = {Path(p).resolve() for p in reference_files}
    candidates = {Path(p).resolve() for p in candidate_files}
    seen = set()
    for item in mapping.mappings:
        reference = Path(item.reference_file).resolve()
        if reference not in references or reference in seen:
            raise ValueError("Judge returned unknown or duplicate reference mapping")
        seen.add(reference)
        if item.candidate_file is not None and Path(item.candidate_file).resolve() not in candidates:
            raise ValueError("Judge selected a path outside the supplied candidate files")


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    source = Path(spec["scorer_dir"])
    output = Path(spec["output"])
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != SCORER_REVISION:
        raise ValueError("PromptBio evaluator revision mismatch")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip():
        raise ValueError("PromptBio evaluator checkout contains changed or untracked files")
    tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).split(b"\0")
    source_hashes = {
        name.decode(): hashlib.sha256((source / name.decode()).read_bytes()).hexdigest()
        for name in tracked
        if name and (source / name.decode()).is_file()
    }
    transport_hash = hashlib.sha256(
        Path(sys.modules[GuardedSyncChatTransport.__module__].__file__).read_bytes()
    ).hexdigest()
    publish(
        output / "runtime-source.json",
        {
            "scorer_revision": head,
            "scorer_files": source_hashes,
            "guarded_transport_sha256": transport_hash,
            "guarded_transport_file": str(Path(sys.modules[GuardedSyncChatTransport.__module__].__file__).resolve()),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    guarded = GuardedSyncChatTransport(
        socket_path=spec["socket_path"],
        token=spec["token"],
        lineage=spec["lineage"],
        digest=spec["digest"],
        namespace=spec["namespace"],
        model=MODEL,
        output=output / "calls",
    )
    validated = ValidatedJudgeTransport(guarded)
    with httpx.Client(transport=validated) as client:
        import langchain.chat_models

        original = langchain.chat_models.init_chat_model

        def initialize(model=None, **kwargs):
            print(json.dumps({"campaign_model_override": {"upstream_requested": model, "actual": MODEL}}), flush=True)
            kwargs.update(
                model_provider="openai",
                base_url="http://gym-internal/v1",
                api_key="task-scoped-proxy",
                max_tokens=16384,
                max_retries=0,
                http_client=client,
            )
            return original(MODEL, **kwargs)

        langchain.chat_models.init_chat_model = initialize
        task = Path(spec["task_dir"])
        runs = [Path(x) for x in spec["runs"]]
        sys.path.insert(0, str(source))
        import utils.match

        original_match = utils.match.match_output_files
        mapping_failures = []

        def checked_match(*args, **kwargs):
            mapping = original_match(*args, **kwargs)
            try:
                validate_mapping(
                    mapping, reference_files=kwargs["reference_files"], candidate_files=kwargs["candidate_files"]
                )
            except Exception as error:
                mapping_failures.append(str(error))
                publish(
                    output / "mapping-rejection.json",
                    {"error": str(error), "mapping": mapping.model_dump(mode="json")},
                )
                raise
            return mapping

        utils.match.match_output_files = checked_match
        sys.argv = [
            str(source / "run_eval.py"),
            "--task-dir",
            str(task),
            "--result-dir",
            ",".join(str(r / "workspace/outputs") for r in runs),
            "--label",
            ",".join(r.name for r in runs),
            "--output-dir",
            str(output),
            "--model",
            MODEL,
        ]
        try:
            runpy.run_path(str(source / "run_eval.py"), run_name="__main__")
        finally:
            langchain.chat_models.init_chat_model = original
            utils.match.match_output_files = original_match
        if mapping_failures:
            raise RuntimeError("Creator caught unsafe mapping; no accepted reward")
        if validated.failed:
            raise RuntimeError("Creator caught judge failure; raw scores retained without accepted reward")
    publish(
        output / "worker-complete.json",
        {
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "paired_runs": len(spec["runs"]),
            "scorer_revision": head,
            "guarded_transport_sha256": transport_hash,
        },
    )


if __name__ == "__main__":
    main()
