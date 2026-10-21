# Saved scientific-response recovery control

`saved_recovery_control.py` runs the maintained SimpleAgent and checkpoint control plane in separate baseline, interrupted and restored processes. It returns an unchanged historical QCBench model response locally, checkpoints the native completed activation before verification, kills that process, restores the session in a fresh process, and executes the real pinned creator verifier through its maintained LocalProvider subprocess path. The existing wait-mode verification contract is unchanged. A saved known-correct QCBench answer must score one, while the recovered original budget-exhausted answer retains zero and its original disposition. A distinct MolJSON row is actually scored before interruption and its completed artifact must remain byte-identical.

The driver uses the existing test agent factory and mocked transport around actual native component/checkpoint methods, not the full CLI batch/coordinator. The completed MolJSON row proves byte preservation, not heterogeneous scheduler restoration. It proves real scientific subprocess grading after clean-process restoration. It does not prove in-flight scorer replay, mutable tool/container restoration, general unattended heterogeneous recovery, or new model performance. No external provider is constructed. Existing source checks, reference metadata and 600-second scoring limit remain unchanged; the control supervisor bounds each phase separately.

From the repository root, with the preserved campaign assets:

```sh
PYTHONPATH=. ../../outputs/broad-eval-campaign/foundation/harbor-upgrade/venv/bin/python resources_servers/campaign_creator_scoring/tests/saved_recovery_control.py \
  --input ../../outputs/broad-eval-campaign/foundation/qcbench-native-validity-regrade50-v1/results.jsonl \
  --creator ../qcbench-upstream \
  --completed-input ../../outputs/broad-eval-campaign/foundation/moljson-saved20/cohort1/original-results.jsonl \
  --completed-creator ../moljson-data-upstream \
  --output /absolute/new/control-directory
```

Use a fresh output directory. Local process inspection permission is required to assert no scorer descendants remain. The original artifacts are read-only inputs. This intentionally does not classify the historical model calls as newly guarded or reconciled.
