# ScienceQA: text-completion recipe

**Accepted engineering integration for a fixed 50-task sample**, reviewed October 23, 2026. Nemotron 3.5 Super VL answered 47 of 50 questions correctly under the strict extraction metric. A fresh saved-output scoring pass reproduced all results. Scientific reference validity has not been systematically reviewed.

This is a general ScienceQA sample: 17 natural-science, 17 language-science, and 16 social-science questions. Five are tagged biology. The creator's text-only recipe intentionally omits images, including images associated with 23 of these tasks. The result applies to this declared protocol and sample.

## Source, sample, and protocol

The adapter uses [ScienceQA](https://github.com/lupantech/ScienceQA) at revision [`2cbf8318e07b9ece895bb2ae605e71e38d623264`](https://github.com/lupantech/ScienceQA/tree/2cbf8318e07b9ece895bb2ae605e71e38d623264). Its public test split contains 4,241 questions. The [sample manifest](scienceqa-sample.json) supplies all 50 IDs in scoring order and the task-packet, output, and scorer hashes.

The named protocol is `scienceqa-creator-qcm-ale-2shot-no-caption-completion-v1`. It calls the creator's prompt builder with two demonstrations, IDs 6493 and 16241, selected with seed 3. Requests use the literal raw-completion prompt, temperature 0, a 512-token allowance, top-p 1, zero frequency/presence penalties, and a newline stop. The engine is explicitly `nvidia/nvidia/nemotron-3.5-super-vl`. The provider does not supply a model-weight hash in this evidence.

[Gym configuration](../../../benchmarks/scienceqa/creator-recipe.yaml) · [Adapter instructions](../../../benchmarks/scienceqa/README.md#separate-native-creator-recipe-configuration) · [Preparation](../../../benchmarks/scienceqa/prepare_recipe.py) · [Creator prompt and scoring bridge](../../../benchmarks/scienceqa/creator_recipe.py) · [Saved-output scorer](../../../benchmarks/scienceqa/regrade_recipe.py)

## Execution and verification

| Check | Result |
|---|---|
| Retained model execution | 50 unique requests, all HTTP 200, all normal stop termination |
| Prompt and engine identity | All 50 captured requests match the literal creator recipe and declared model |
| Strict extraction accuracy | 47/50, or 94% |
| Creator scoring with ordered fallback | 47/50; one response invokes the creator's randomized fallback |
| Masks, missing outcomes, or duplicate task IDs | Zero |
| Fresh saved-output scoring | All 50 results reproduced; zero new model calls |
| Reviewed trajectories | Five complete examples covering biology, correct extraction, an incorrect response, omitted images, and repeated-answer parsing |
| Clean setup check | Fresh Python 3.13.14 environment, frozen `uv.lock`, 84 installed packages, and matching imported module hashes |
| Clean native CLI control | Five deterministic local-provider cases, including a malformed answer; four strict passes and one strict failure |
| Resume and cleanup control | Completed-output resume makes zero additional fixture calls; owned processes and ports are absent |

The clean setup control was performed on macOS using maintained source commit `ec3c3ab6c1e08cf3825e17efc176b0caa19ec2e6`. It qualifies that local CLI path. A supervised Linux worker remains a separate deployment qualification. The five deterministic fixture responses are software controls; the measured 50-task result comes from retained real Nemotron responses. Independent reviews inspected those raw captures and reran scoring.

The frozen source environment can be rebuilt with `uv sync --frozen --no-dev --python 3.13.14`. The adapter instructions describe source checkout, sample staging, raw-completion capability admission, and model binding. The exact executed configurations, commands, captures, and setup receipts are retained in the private first-review evidence directory. Their scientific identities are bound by the public sample manifest; credentials are excluded from this repository.

## Interpretation limits

One response repeats the correct answer letter in its explanation. The creator's parser requires a single matching answer, so this response enters its randomized fallback path and remains a strict zero. Preserve the creator result and the strict extraction result separately. Any alternative parser would define another reported condition.

Omitted images affect what the model can know. For example, a question asking which colony is shown receives no map in this recipe. The resulting score measures the released text-only procedure, including its limitations. It should not be labeled microscopy, visual reasoning, or life-sciences-only performance.

The source software uses MIT licensing, while the dataset is distributed under CC BY-NC-SA 4.0. Source, dataset, and redistribution terms remain distinct. The result is a fixed-sample engineering check; it does not establish population accuracy or reproduce another model's published leaderboard result.
