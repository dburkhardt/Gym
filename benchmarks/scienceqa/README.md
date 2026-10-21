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
