# FGBench campaign

Use shared creator server with pinned upstream repo and seeded sample env variables like QCBench. The solver receives the exact public question as a zero-shot user message. This is a documented prompt variant pending comparison with creator model-specific notebooks.

**Never report mean Gym reward as FGBench performance.** Boolean task rewards are creator accuracy; numeric transport reward is negative squared error and invalid values use0 solely for serialization. Use `aggregate.py` on saved rollouts: it calls the original creator aggregate and reports six separate accuracy/RMSE/validity metrics. The creator excludes numeric zero outputs and zero references from RMSE; invalid rate must accompany it. Five examples do not cover all six strata; use20 before aggregate reporting.
