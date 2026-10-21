# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned one-pass creator loop with explicit tokenizer and guarded local transport.

This is a component bridge, not a runnable benchmark. Self-debug is refused until
its dynamic install/execution path has an isolated qualified runtime. The caller
must supply a qualified model-specific trimming function; there is no tokenizer
fallback. Tests may supply a clearly marked fixture trimmer, never production proof.
"""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import OpenAI

from benchmarks.scienceagentbench.prepare import MANIFEST, contained, digest
from nemo_gym.guarded_chat_transport import GuardedSyncChatTransport


def pinned_tree(source, relative):
    files = (
        json.loads(MANIFEST.read_text())["source_files"]
        | json.loads(Path(__file__).with_name("bridge-source.json").read_text())["source_files"]
    )
    expected = files[relative]
    path = contained(source, relative)
    if digest(path) != expected:
        raise ValueError(f"Creator source changed: {relative}")
    return ast.parse(path.read_text())


class CreatorEngine:
    """Execute the actual creator OpenAI request function without its retry decorator.

    The default explicit Nemotron adaptation maps the pinned Bedrock function's
    normal-model sampling to OpenAI. SDK retry count10 and unbounded backoff
    are replaced by Gym's durable local guard. The source OpenAI branch is
    available only through diagnostic-creator-openai-branch, never implicit.
    """

    def __init__(
        self,
        source,
        *,
        model,
        socket_path,
        token,
        lineage,
        configuration_sha256,
        output,
        transport=None,
        routing_policy="nemotron-normal-model-openai-v1",
    ):
        tree = pinned_tree(source, "engine/openai_engine.py")
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "openai_chat_engine")
        function.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned-openai-engine", "exec"), namespace)
        self._request = namespace["openai_chat_engine"]
        self.routing_policy = routing_policy
        if routing_policy == "nemotron-normal-model-openai-v1":
            if model != "nvidia/nvidia/nemotron-3.5-super-vl":
                raise ValueError("Normal-model adaptation requires the declared Nemotron model")
            bedrock = pinned_tree(source, "engine/bedrock_engine.py")
            function = next(
                n for n in bedrock.body if isinstance(n, ast.FunctionDef) and n.name == "bedrock_converse_engine"
            )
            native = {}
            exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned-bedrock-sampling", "exec"), native)

            def normal_request(client, model, messages, temperature, top_p):
                def converse(*, modelId, messages, inferenceConfig):
                    return client.chat.completions.create(
                        model=modelId,
                        messages=messages,
                        max_tokens=inferenceConfig["maxTokens"],
                        temperature=inferenceConfig["temperature"],
                        top_p=inferenceConfig["topP"],
                    )

                return native["bedrock_converse_engine"](
                    SimpleNamespace(converse=converse), model, messages, temperature, top_p
                )

            self._request = normal_request
        elif routing_policy != "diagnostic-creator-openai-branch":
            raise ValueError("Unknown creator routing policy")
        self.llm_engine_name = model
        self.failed = False
        guarded = GuardedSyncChatTransport(
            socket_path=socket_path,
            token=token,
            lineage=lineage,
            digest=configuration_sha256,
            namespace="scienceagentbench",
            model=model,
            output=output,
            transport=transport,
        )
        self.client = OpenAI(
            api_key="local-bridge-only",
            base_url="http://gym-internal",
            max_retries=0,
            http_client=httpx.Client(transport=guarded, trust_env=False),
        )

    def respond(self, user_input, temperature, top_p):
        if self.failed:
            raise RuntimeError("Prior creator call unresolved or malformed; no automatic resend")
        self.failed = True
        response = self._request(self.client, self.llm_engine_name, user_input, temperature, top_p)
        choice = response.choices[0]
        if not isinstance(choice.message.content, str) or response.usage is None:
            raise ValueError("Captured creator response lacks text or token usage")
        # Raw completion and length/content-filter status remain in durable guard output.
        result = (choice.message.content, response.usage.prompt_tokens, response.usage.completion_tokens)
        self.failed = False
        return result

    def close(self):
        self.client.close()


def one_pass_agent(
    source,
    engine,
    *,
    trim_messages,
    tokenizer_identity,
    input_cost_per_token,
    output_cost_per_token,
    use_knowledge=False,
    use_self_debug=False,
    context_cutoff=28000,
):
    """Load unchanged creator class/constant ASTs; no host code execution hooks.

    Tokenizer identity is caller-declared provenance, not runtime qualification.
    An eventual native admission layer must verify it against actual tokenizer
    assets. This function never asserts that an arbitrary declaration is valid.
    """
    if use_self_debug:
        raise ValueError("Self-debug requires qualified isolated install/execution runtime")
    if not tokenizer_identity or not callable(trim_messages):
        raise ValueError("Explicit qualified model-specific tokenizer is required")
    tree = pinned_tree(source, "agent.py")
    nodes = [node for node in tree.body if isinstance(node, (ast.Assign, ast.ClassDef))]

    def refuse(*args, **kwargs):
        raise RuntimeError("Creator host installation/execution is forbidden")

    namespace = {
        "LLMEngine": lambda name: engine,
        "model_cost": {
            engine.llm_engine_name: {
                "input_cost_per_token": input_cost_per_token,
                "output_cost_per_token": output_cost_per_token,
            }
        },
        "trim_messages": trim_messages,
        "Path": Path,
        "re": re,
        "subprocess": SimpleNamespace(run=refuse),
        "os": SimpleNamespace(mkdir=refuse),
        "rmtree": refuse,
        "copyfile": refuse,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-science-agent", "exec"), namespace)
    return namespace["ScienceAgent"](
        engine.llm_engine_name, context_cutoff=context_cutoff, use_self_debug=False, use_knowledge=use_knowledge
    )


def visual_judge(source, engine, predicted_base64, gold_base64):
    """Creator visual prompt/parser and n=3; explicit judge-model substitution.

    Call only in a trusted verifier phase after generated-code execution has
    stopped. This helper does not establish image provenance or isolate a process.
    """
    tree = pinned_tree(source, "gpt4_visual_judge.py")
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PROMPT_ORIGIN" for target in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "score_figure")
    ]

    def create(**kwargs):
        if engine.failed:
            raise RuntimeError("Prior creator call unresolved or malformed; no automatic resend")
        if kwargs.pop("model") != "gpt-4o-2024-05-13":
            raise ValueError("Unexpected creator visual judge model")
        engine.failed = True
        response = engine.client.chat.completions.create(model=engine.llm_engine_name, **kwargs)
        if len(response.choices) != 3 or any(
            not isinstance(choice.message.content, str) for choice in response.choices
        ):
            raise ValueError("Creator visual scoring requires three captured textual choices")
        engine.failed = False
        return response

    namespace = {
        "client": SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        "AzureOpenAI": type("UnusedAzureRoute", (), {}),
        "re": re,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-visual-judge", "exec"), namespace)
    return namespace["score_figure"](predicted_base64, gold_base64)
