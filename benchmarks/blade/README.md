# BLADE native integration staging

Pinned creator source: `6118fa8d5007b91aa8c91c518182db82446a4547`. The maintained preparation entrypoint validates 186 released files and preserves the existing three biological dataset selections: crofoot, panda_nuts, amtl. It stages native source plus public CSV/info into a solver context, with annotations/MCQ references outside that context. It preserves the original native info/question rather than replacing the scientific task with a text QA prompt. Repeated preparation is deterministic and rejects contaminated solver contexts.

This is **preparation, not yet an executable maintained Gym integration**. No live inference, Docker build or scientific score is claimed. There is deliberately no nominal runnable YAML that silently swaps the native solver for Terminus or substitutes a scalar scorer.

The next hook must retain `RunLLMMultiRun(use_agent=True, use_data_desc=True, num_runs=1, use_code_cache=False)` with its native ten-turn behavior. It must run within the shared sandbox/Harbor lifecycle, not execute Jupyter on the verifier host. Scoring runs separately via creator `run_eval_on_analyses`, with references provided only then. Preserve the full conceptual-variable/transform/model matching metric vector; it is not a single accuracy score. Solver, repair and matching roles must all use exact Nemotron3.5SuperVL through guarded Gym capture; the creator retrying OpenAI transport needs a no-hidden-resend binding before activation.

Historical CPU runtime image and actual artifacts establish possible native execution, not current shared-path readiness. Their Python3.11/Torch2.5.1 stack cannot simply import this Python3.13 Gym checkout. Do not activate the old bespoke Docker/proxy launcher. Existing sandbox, guarded transport and native phase hooks need an explicit compatibility bridge and acceptance controls.

Prepare with `BLADE_SOURCE=/path/to/pinned/source python -m benchmarks.blade.prepare`. Data is staged under this benchmark's data directory. Reference data must never be used as a solver image build context.
