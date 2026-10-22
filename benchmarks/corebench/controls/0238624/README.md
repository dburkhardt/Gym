# Full workflow control

Copy this directory into the prepared task's `solution` directory and add
`source.sha256` from the frozen medium-file manifest. Run with Harbor's oracle
agent through the existing Gym integration, retaining the original task resources
and time limits. Retain `/workspace` and `/results` as task artifacts.

The control invokes the original `code/run`: all data, three bundled frozen
submodels, the unseeded 75/25 split, and 150 epochs remain unchanged. It checks
frozen source hashes before and after execution and records the complete log.
Inspect that log for all 150 epochs and the final classification report before
claiming workflow completion. A successful environment import is insufficient.

The empty answer report is intentional. Any verifier zero is an operational
control result and must not enter the model-performance denominator. This control
does not establish agreement with published precision or recall; the original
unseeded procedure may vary across executions.
