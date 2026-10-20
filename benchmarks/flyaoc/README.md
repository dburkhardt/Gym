# FlyAOC maintained migration

Preparation preserves the frozen 50 gene IDs, dataset revision and input bytes. Choose the original first 5, 20 or 50 tasks with `FLYAOC_SAMPLE_SIZE`; provide the pinned `inputs-50.jsonl` as `FLYAOC_FROZEN_INPUTS`. `FLYAOC_DATA` selects the output directory. Prompts preserve the historical gene identifier, symbol, paper IDs and public gene summary. Reference annotations are not published in solver inputs.

This is preparation only. The historical pipeline used the creator agent, paper/corpus tools and semantic ontology metrics; replacing it with single-turn question answering would change the benchmark. A maintained native agent, pinned corpus cache, shared guarded inference for all helper roles, saved-output scoring and native Gym configuration remain required. No runnable config is published yet. Source provenance currently has an archive short revision, not a verified full upstream Git commit.
