# FGBench campaign

Use shared creator server with pinned upstream repo and seeded sample env variables like QCBench. The solver receives the exact public question as a zero-shot user message. This is a documented prompt variant pending comparison with creator model-specific notebooks.

**Never report mean Gym reward as FGBench performance.** Boolean task rewards are creator accuracy; numeric transport reward is negative squared error and invalid values use0 solely for serialization. Use `aggregate.py` on saved rollouts: it calls the original creator aggregate and reports six separate accuracy/RMSE/validity metrics. The creator excludes numeric zero outputs and zero references from RMSE; invalid rate must accompany it. Five examples do not cover all six strata; use20 before aggregate reporting.

Native reporting now projects each row into its own creator stratum: accuracy over all boolean answers, RMSE over creator-valid numeric pairs, and valid-rate over every observed answer in that stratum. Numeric per-row values retain squared error; the existing disposition report applies the declared `sqrt_mean` aggregation and returns null when no valid pair exists. Raw transport reward remains unchanged, but generic scalar aggregation is disabled in this benchmark config.

Run `aggregate.py` with `--native-report PATH` to compare saved creator outputs with the same native disposition projection. This qualifies implementation parity, not the scientific correctness of every benchmark question. The campaign remains a zero-shot prompt variant. Reports from different model/settings or source versions stay separate. No combined FGBench scalar is defined.
