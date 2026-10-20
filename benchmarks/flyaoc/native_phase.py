# SPDX-License-Identifier: Apache-2.0
"""Run the pinned creator single-agent without any reference-bearing data loader."""

import json
import os
from pathlib import Path


def main():
    import httpx
    import openai
    from guarded_chat_transport import GuardedChatTransport

    output = Path("/output")
    request = json.loads(Path("/request.json").read_text())
    payload = request["input"]
    if set(payload) != {"gene_id", "gene_symbol", "pmcids", "summary"}:
        raise ValueError("Solver input must contain only public frozen gene fields")
    transport = GuardedChatTransport(
        socket_path="/model-socket/model.sock",
        token=os.environ["FLYAOC_TOKEN"],
        lineage=request["lineage"],
        digest=request["config_sha256"],
        namespace="flyaoc",
        model=request["model"],
        output=output / "model-calls",
    )
    client = openai.AsyncOpenAI(
        api_key="task-local-placeholder",
        base_url="http://gym-internal/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport, timeout=600),
    )
    # The pinned creator constructs AsyncOpenAI in its single-agent path. Scope the
    # replacement to this isolated process; no host service/global client is changed.
    openai.AsyncOpenAI = lambda *args, **kwargs: client
    from flyaoc.baselines.normalize import normalize_result
    from flyaoc.baselines.runners import BaselineRunner
    from flyaoc.baselines.types import BaselineRunConfig, GeneInput

    config = BaselineRunConfig(
        baseline="single_agent",
        provider="openai_compatible",
        model=request["model"],
        paper_budget=10,
        output_dir=output,
        gene_id=payload["gene_id"],
        workers=1,
        resume=False,
        max_turns=50,
        max_cost=1.0,
        verbose=True,
    )
    config.validate()
    raw = BaselineRunner(config).run_gene(GeneInput(**payload))
    (output / "raw.json").write_text(json.dumps(raw.to_dict(), indent=2))
    (output / "predictions.jsonl").write_text(json.dumps(normalize_result(raw)) + "\n")
    if raw.error:
        raise RuntimeError("Creator agent failed; full raw trace preserved, not a scientific zero")
    if transport.failed:
        raise RuntimeError("Native provider outcome unresolved; no score")


if __name__ == "__main__":
    main()
