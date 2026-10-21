# scienceqa campaign

Uses shared creator scorer and explicit preselected sample environment variables. Creator GPT caption-based QCM-A prompt with zero shots, not multimodal image execution. Creator random invalid-answer fallback is retained but made deterministic per task ID and reported alongside strict_score; never interpret fallback correctness as valid answer. All4241test rows; stratified subject sampling.

Native results now expose `creator_seeded_fallback_accuracy` and `strict_accuracy`
separately, retaining `fallback_randomized` and raw scores. A random fallback is
invalid as model answer correctness; it is not silently removed or converted to a
new score. Other items remain scientifically unresolved unless separately reviewed.

The reviewed pinned creator `data/captions.json` contains the baseball caption for
item11434, whose question requires planetary volumes. Exact asset, task and prompt
matching identifies this known caption-only defect and masks its native scalar;
changed inputs are unresolved. The review records subject/topic/category and does
not represent this general school-science/language/social-science sample as a life
sciences-only evaluation.

`caption-qcm-a-zero-shot` and `caption-qcm-a-explicit-format` remain separate named
protocols. Preparation still uses unchanged frozen-sample staging: it adds stable
Gym identity/provenance, not replacement captions or output instructions. Review
metadata is derived at scoring from actual submitted inputs and pinned assets,
not trusted from caller-supplied annotations. No image-based performance is claimed.

## Separate documented creator recipe (offline preparation)

`creator_recipe.prepare(repo, ordered_test_ids)` builds the separately named
`scienceqa-creator-qcm-ale-2shot-no-caption-completion-v1` protocol directly through
the pinned creator's `build_prompt`. It uses training demonstrations6493 and16241
(seed3),QCM-ALE and no captions. Solver requests contain only the raw completion
prompt and creator generation parameters: temperature0,max512,top-p1,zero
frequency/presence penalties and newline stop. References remain separate in
`evaluator_references`; never send the enclosing packet to the model.

`score_ordered` preserves the creator regex and random fallback stream after the
same two-shot selection. For a selected subset this is a fresh, explicitly ordered
experiment, not the fallback RNG state after all preceding full-test items.
Fallback scores are retained but invalid as answer correctness; strict accuracy is
reported separately. No old caption/chat results, masks or prompts are rewritten.

This path is not runnable through current native Gym model APIs: they expose
Responses and chat completions, not raw text completions. Chat wrapping would
change the prompt semantics. Preparation therefore reports the missing
`raw_text_completion` capability instead of publishing a misleading runnable
benchmark config. A future guarded raw-completion route and actual provider
capability evidence are required. Matching the recipe with another model would
not reproduce the creator's text-davinci-002 performance.

## Separate native creator-recipe configuration

`creator-recipe.yaml` uses the shared `simple_completion` harness and the opt-in
raw model route. Set `SCIENCEQA_CREATOR_REPO` to the pinned source checkout,
`CAMPAIGN_SAMPLE_JSONL` to the original frozen IDs, and
`SCIENCEQA_COMPLETION_OUTPUT_ROOT` to a fresh attempt directory. Set
`SCIENCEQA_COMPLETION_MODEL` explicitly. The model config must enable
`raw_completions_enabled` and declare `raw_text_completion` capability only after
provider qualification. The frozen literal request contains the original recipe
sampling settings; general chat-generation overrides do not rewrite that request.

Per-row verification reports strict extraction accuracy. Invalid formatting is
masked, with `creator_raw_accuracy: null`, because the creator's random fallback
depends on earlier malformed outputs. Run `benchmarks.scienceqa.regrade_recipe`
on the completed result file and prepared `.packet.json` to calculate that raw
metric in the exact frozen order without model calls. It records scorer/output
hashes and a scorer lineage; it refuses missing or duplicate task IDs. Subset RNG
is explicitly not full-test RNG parity. Original caption/chat data and runs remain
unchanged. The raw route and this configuration require separate actual provider
qualification; no new scientific result is implied by local fixtures.
