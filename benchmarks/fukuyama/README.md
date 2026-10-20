# FukuyamaBench pathway inference

Pinned creator repository: https://github.com/HaCTang/ReactionMechanismReasoning/tree/63bb79f912f0b2de593996b80ffeea894f6f1a59

This adapter evaluates the creator's text pathway-inference task on all 319 public mechanisms (A 78, B 131, C 110). It retains the creator YAML system/user prompts and mapped starting reactants and conditions. Mechanism pathways and checkpoint references remain verifier metadata. The scorer imports the creator's JSON extraction and lenient checkpoint pathway comparison, with exact-match as the scalar reward and the other creator metrics retained. This is the text pathway task, not diagram recognition.

Stage pinned repository as `CAMPAIGN_UPSTREAM_REPO`; prepare seeded 5/20/50 rows with `python -m benchmarks.scientific_agent_campaign.deterministic fukuyama --repo "$CAMPAIGN_UPSTREAM_REPO" --output PATH`. Set `CAMPAIGN_SAMPLE_JSONL` to one sample, then `python benchmarks/fukuyama/prepare.py` before native Gym startup. Load `benchmarks/fukuyama/config.yaml` with the campaign Nemotron model configuration. Dependencies are shared resource requirements, RDKit, and PyYAML. Twenty gold-pathway controls all score 1; empty controls all score 0. No model execution is claimed by these controls.
