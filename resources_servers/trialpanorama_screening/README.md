# TrialPanorama measurement qualification

The scalar reward remains the pinned creator's micro accuracy. Its parser synthesizes all-excluded decisions after malformed output, and its evaluator averages only supplied recognized studies. These behaviors can award perfect raw credit without a complete answer.

The separate `complete_output_micro_accuracy_v1` metric requires explicit JSON decisions for every unique reference PMID, without duplicate predictions; it accepts the creator's lowercase normalization and confidence conversion. Unusable predictions receive zero in this named variant, remain attempted model failures, and remain in its reviewed denominator. This is not a replacement for the original creator score. Native scalar selection is conservatively masked when the original measurement lacks complete prediction evidence; the existing per-metric report retains eligible variant zeros.

Duplicate reference PMIDs make both metric dispositions unresolved, even when labels agree. Raw values remain available, but input ambiguity is not attributed to the model. Frozen sample24 contains such a duplicate; none of its three reference/empty/malformed controls is silently repaired or included in a reviewed mean.

Reviews bind sample and answer hashes, exact creator commit and source-file bytes, and the review implementation hash. A modified creator working tree is refused. This establishes parser/coverage qualification, not independent medical correctness or full-population performance. Saved15 controls are in `outputs/broad-eval-campaign/foundation/trialpanorama-validity` in the campaign workspace:12 eligible strict controls include8 model-failure zeros;3 reference-ambiguous controls remain unresolved.

## Native runtime and loopback control

Set `TRIALPANORAMA_SOURCE` to the exact pinned creator checkout; the maintained
resource config now binds this required path explicitly. The verifier checks both
commit20e97cbb823003fccc425732d593cf2bb205d163 and working-tree scorer bytes.

`tests/native_control.py` runs the supported native CLI with two saved response
fixtures and their original complete batch prompts. It uses actual simple-agent,
model capture and verifier services, a loopback-only provider, and the existing
component environment. Run from the Gym root with `--output` (fresh directory),
`--saved` (historical native JSONL) and `--creator` (pinned checkout). Local
ports25300–25330 must be free. No package installation or external model key is
required; use an already qualified Gym interpreter.

The accepted local control produced two single-attemptHTTP200 captures and two
healthy native results. Raw rewards1.0 and0.9 were preserved; the first remains
masked by its measurement review. Five owned service processes were absent after
exit0. This qualifies the tested shared runtime path, not provider capacity,
scientific correctness, or new model performance. Saved fixture source identities
and raw captures remain in the campaign's `foundation/trialpanorama-native-control-v1`.
