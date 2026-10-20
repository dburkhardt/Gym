# SupraBench TBS base MCQ subset

Creator source https://github.com/Tianyi-Billy-Ma/SupraBench at bcb06eb6e1d283b70e943ed0c90aa9e6f97826e5. Uses all 2064 public `data/tbs/base.jsonl` records, exact pre-rendered questions and original TBS answer parser. This covers only the strongest-binding TBS base task; no claim of coverage of BAP/SID/HGD/VQA or few-shot variants. Correct-molecule names and affinities never enter model inputs beyond the options already supplied in the creator question.

Creator accuracy excludes unparsed answers. Run aggregate.py on traces for that metric and the all-attempt diagnostic; do not label ordinary mean Gym reward creator accuracy when parse failures occur. The original loose letter parser is retained and can parse stray letters from unrelated prose; this is a protocol limitation.

Use pinned upstream checkout as CAMPAIGN_UPSTREAM_REPO. Prepare with `python -m benchmarks.scientific_agent_campaign.deterministic suprabench --repo SOURCE --output OUTPUT`; set CAMPAIGN_SAMPLE_JSONL and stage with `python benchmarks/suprabench/prepare.py` before loading config.yaml. Gold-reference and empty controls verify scoring, not model performance.
