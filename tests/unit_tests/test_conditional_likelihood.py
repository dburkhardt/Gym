# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_gym.conditional_likelihood import (
    LikelihoodIdentity,
    LikelihoodObservation,
    LikelihoodRequest,
    LikelihoodUnavailable,
    prepare_likelihood,
    validate_likelihood,
)


def request(**kwargs):
    return LikelihoodRequest(
        identity=LikelihoodIdentity(
            model="benign-fixture",
            model_revision="exact-fixture-v1",
            tokenizer="character-fixture",
            tokenizer_sha256="a" * 64,
            tokenizer_config_sha256="b" * 64,
            prefix_token_id=0,
        ),
        context="cat ",
        continuation="sat",
        max_length=8,
        **kwargs,
    )


def test_exact_boundary_and_sum():
    plan = prepare_likelihood(request(), encode=lambda s: list(s.encode()))
    assert plan.context_token_ids == tuple(b"cat")
    assert plan.continuation_token_ids == tuple(b" sat")
    assert plan.model_input_token_ids == tuple(b"cat sa")
    assert plan.continuation_logit_positions == (2, 3, 4, 5)
    assert plan.trailing_context_characters_moved == 1
    observation = LikelihoodObservation(
        **{
            k: getattr(plan, k)
            for k in (
                "request_digest",
                "identity",
                "model_input_token_ids",
                "continuation_token_ids",
                "continuation_logit_positions",
            )
        },
        token_logprobs=(-0.1, -0.2, -0.3, -0.4),
        summed_logprob=-1.0,
    )
    with pytest.raises(LikelihoodUnavailable, match="not been qualified"):
        validate_likelihood(plan, observation)
    assert validate_likelihood(plan, observation, capability=True) == -1.0
    for updates in (
        {"token_logprobs": (float("nan"),) * 4},
        {"token_logprobs": (-0.1,) * 3},
        {"summed_logprob": -2.0},
        {"continuation_logit_positions": (0, 1, 2, 3)},
        {"request_digest": "different"},
        {"model_input_token_ids": (1, 2)},
        {"identity": plan.identity.model_copy(update={"model_revision": "other"})},
    ):
        with pytest.raises(LikelihoodUnavailable):
            validate_likelihood(plan, observation.model_copy(update=updates), capability=True)


def test_empty_context_prefix_and_context_truncation():
    base = request().model_copy(update={"context": "", "continuation": "a"})
    plan = prepare_likelihood(base, encode=lambda s: list(s.encode()))
    assert plan.context_token_ids == (0,) and plan.model_input_token_ids == (0,)
    assert plan.continuation_logit_positions == (0,)
    base = request().model_copy(update={"context": "abcde", "continuation": "fg", "max_length": 3})
    with pytest.raises(LikelihoodUnavailable, match="context truncation"):
        prepare_likelihood(base, encode=lambda s: list(s.encode()))
    plan = prepare_likelihood(
        base.model_copy(update={"truncation": "left_context_lm_eval_0.4.2"}), encode=lambda s: list(s.encode())
    )
    assert plan.model_input_token_ids == tuple(b"def")
    assert plan.continuation_token_ids == tuple(b"fg")
    assert plan.continuation_logit_positions == (1, 2)
    assert plan.left_context_tokens_dropped == 3
    with pytest.raises(LikelihoodUnavailable, match="Continuation exceeds"):
        prepare_likelihood(base.model_copy(update={"continuation": "long"}), encode=lambda s: list(s.encode()))


def test_cross_boundary_merge_is_not_silently_scored():
    def encode(text):
        return {"a": [1], "ab": [9]}[text]

    with pytest.raises(LikelihoodUnavailable, match="merges"):
        prepare_likelihood(request().model_copy(update={"context": "a", "continuation": "b"}), encode=encode)


def test_tokenizer_change_changes_capture_binding():
    original = request()
    other = original.model_copy(
        update={"identity": original.identity.model_copy(update={"tokenizer_sha256": "c" * 64})}
    )
    assert (
        prepare_likelihood(original, encode=lambda s: list(s.encode())).request_digest
        != prepare_likelihood(other, encode=lambda s: list(s.encode())).request_digest
    )
