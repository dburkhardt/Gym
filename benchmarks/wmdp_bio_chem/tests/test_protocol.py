# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harmless fixtures execute pinned creator methods; no hazardous task text is emitted."""

import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jinja2
import numpy as np
import pytest
import yaml

from benchmarks.wmdp_bio_chem.prepare import MANIFEST, prepare
from benchmarks.wmdp_bio_chem.protocol import context, requests, score
from nemo_gym.conditional_likelihood import (
    LikelihoodIdentity,
    LikelihoodObservation,
    LikelihoodUnavailable,
    prepare_likelihood,
)


DOC = {"question": "  Which item is a fruit?  ", "choices": ["apple", "stone", "chair", "cloud"], "answer": 0}
IDENTITY = LikelihoodIdentity(
    model="fixture",
    model_revision="fixture",
    tokenizer="byte-fixture",
    tokenizer_sha256="1" * 64,
    tokenizer_config_sha256="2" * 64,
    prefix_token_id=0,
)


def upstream():
    if "WMDP_UPSTREAM_ROOT" not in os.environ:
        pytest.skip("Pinned creator source not supplied; source parity is not exercised")
    root = Path(os.environ["WMDP_UPSTREAM_ROOT"])
    manifest = json.loads(MANIFEST.read_text())
    for relative, evidence in manifest["sources"].items():
        if not relative.startswith("lm_eval/"):
            continue
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == evidence["sha256"]
    namespace = {}
    exec(compile((root / "lm_eval/api/instance.py").read_text(), "pinned-instance", "exec"), namespace)
    namespace.update(np=np, utils=SimpleNamespace(softmax=lambda x: np.exp(x) / sum(np.exp(x))))
    tree = ast.parse((root / "lm_eval/api/task.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ConfigurableTask")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"construct_requests", "process_results"}
    ]
    for node in methods:
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), "pinned-task", "exec"), namespace
    )
    stub = SimpleNamespace(
        OUTPUT_TYPE="multiple_choice",
        config=SimpleNamespace(target_delimiter=" ", process_results=None),
        multiple_input=False,
        multiple_target=False,
        _metric_fn_list={"acc": None},
        doc_to_choice=lambda doc: ["A", "B", "C", "D"],
        doc_to_target=lambda doc: doc["answer"],
    )
    return root, namespace, stub


@pytest.mark.parametrize("subset", ["wmdp-bio", "wmdp-chem"])
def test_exact_creator_requests_and_accuracy(subset):
    root, namespace, stub = upstream()
    template = yaml.safe_load((root / "lm_eval/tasks/wmdp/_default_template_yaml").read_text())
    spec = yaml.safe_load((root / f"lm_eval/tasks/wmdp/{subset.replace('-', '_')}.yaml").read_text())
    native_context = spec["description"] + jinja2.Template(template["doc_to_text"]).render(**DOC)
    assert context(DOC, subset) == native_context
    native = namespace["construct_requests"](stub, DOC, native_context)
    reqs = requests(DOC, subset, IDENTITY, 4096)
    assert [(r.context, r.continuation) for r in reqs] == [r.arguments for r in native]
    plans = [prepare_likelihood(r, encode=lambda s: list(s.encode())) for r in reqs]
    for values in ([-1.0, -2.0, -3.0, -4.0], [-1.0, -1.0, -2.0, -3.0], [-4.0, -1.0, -2.0, -3.0]):
        observations = [
            LikelihoodObservation(
                request_digest=p.request_digest,
                identity=p.identity,
                model_input_token_ids=p.model_input_token_ids,
                continuation_token_ids=p.continuation_token_ids,
                continuation_logit_positions=p.continuation_logit_positions,
                token_logprobs=[v / len(p.continuation_token_ids)] * len(p.continuation_token_ids),
                summed_logprob=v,
            )
            for p, v in zip(plans, values)
        ]
        with pytest.raises(LikelihoodUnavailable):
            score(plans, observations, DOC, subset, IDENTITY, 4096)
        actual = score(plans, observations, DOC, subset, IDENTITY, 4096, backend_qualified=True)
        assert actual["acc"] == namespace["process_results"](stub, DOC, [(v, False) for v in values])["acc"]
        with pytest.raises(LikelihoodUnavailable, match="canonical ordered"):
            score(
                [plans[1], plans[0], *plans[2:]],
                [observations[1], observations[0], *observations[2:]],
                DOC,
                subset,
                IDENTITY,
                4096,
                backend_qualified=True,
            )
        with pytest.raises(LikelihoodUnavailable, match="canonical ordered"):
            score(
                plans,
                observations,
                {**DOC, "question": "A different harmless question?"},
                subset,
                IDENTITY,
                4096,
                backend_qualified=True,
            )
        other = prepare_likelihood(
            requests({**DOC, "question": "Other question?"}, subset, IDENTITY, 4096)[0],
            encode=lambda text: list(text.encode()),
        )
        other_observation = observations[0].model_copy(
            update={
                "request_digest": other.request_digest,
                "model_input_token_ids": other.model_input_token_ids,
                "continuation_logit_positions": other.continuation_logit_positions,
            }
        )
        with pytest.raises(LikelihoodUnavailable, match="canonical ordered"):
            score(
                [other, *plans[1:]],
                [other_observation, *observations[1:]],
                DOC,
                subset,
                IDENTITY,
                4096,
                backend_qualified=True,
            )
        changed = observations[0].model_copy(update={"request_digest": "wrong"})
        with pytest.raises(LikelihoodUnavailable):
            score(plans, [changed, *observations[1:]], DOC, subset, IDENTITY, 4096, backend_qualified=True)


def test_pinned_full_dataset_staging(tmp_path):
    if "WMDP_DATA_ROOT" not in os.environ:
        pytest.skip("Pinned public assets not supplied; full staging is not exercised")
    path = prepare(Path(os.environ["WMDP_DATA_ROOT"]), tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    counts = {
        subset: sum(row["verifier_metadata"]["subset"] == subset for row in rows)
        for subset in ("wmdp-bio", "wmdp-chem")
    }
    assert counts == {"wmdp-bio": 1273, "wmdp-chem": 408}
    assert len({row["task_id"] for row in rows}) == 1681
    assert all(row["verifier_metadata"]["continuations"] == [" A", " B", " C", " D"] for row in rows)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        prepare(tmp_path, tmp_path)
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_native_adapter_never_substitutes_generation():
    from responses_api_agents.wmdp_conditional.app import Agent, RunRequest

    body = RunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "fixture"}]},
            "verifier_metadata": {},
            "_ng_task_index": 4,
            "_ng_rollout_index": 2,
        }
    )
    assert body.model_dump()["_ng_task_index"] == 4
    with pytest.raises(LikelihoodUnavailable, match="planning only"):
        await Agent.run(None, None, body)
