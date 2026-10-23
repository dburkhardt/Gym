# SPDX-License-Identifier: Apache-2.0
"""Unscored terminal judge outcomes; never substitute a solver scientific zero."""

import hashlib
import json
from pathlib import Path

from nemo_gym.failure_kinds import JUDGE_FAILED
from nemo_gym.sandbox.lifecycle import _publish
from responses_api_agents.promptbio_bench.nemotron_proxy import MODEL, JudgeOutputBudgetExhausted, validate_result


class JudgeBudgetFailure(RuntimeError):
    def __init__(self, grading: Path, runs: list[Path]):
        super().__init__("PromptBio judge output budget exhausted; creator score unavailable")
        self.grading, self.runs = grading, runs

    def record(self, body: dict, rollout_id: str) -> dict:
        """Qualify only settled, captured length failures after owned cleanup exits."""
        failure = json.loads((self.grading / "judge-failure.json").read_text())
        if failure.get("disposition") != "judge_output_budget_exhausted" or failure.get("model") != MODEL:
            raise ValueError("Unqualified judge failure receipt")
        lifecycle = json.loads((self.grading / "judge-lifecycle.json").read_text())
        if lifecycle.get("cleanup_confirmed") is not True:
            raise RuntimeError("Judge cleanup unconfirmed; cannot persist settled budget outcome")
        source = json.loads((self.grading / "runtime-source.json").read_text())
        from responses_api_agents.promptbio_bench.runtime import SCORER_REVISION

        expected = hashlib.sha256(Path(__file__).with_name("guarded_scorer.py").read_bytes()).hexdigest()
        if source.get("implementation_sha256") != expected or source.get("scorer_revision") != SCORER_REVISION:
            raise ValueError("Judge failure source identity mismatch")
        captures = list((self.grading / "calls").glob("*.response.json"))
        if any(not path.name.split(".")[0].isdigit() for path in captures):
            raise ValueError("Invalid judge capture ordinal")
        captures.sort(key=lambda path: int(path.name.split(".")[0]))
        if not captures:
            raise ValueError("Judge budget failure lacks captured response")
        capture = json.loads(captures[-1].read_text())
        if (
            capture.get("status") != 200
            or hashlib.sha256(capture["body"].encode()).hexdigest() != failure["response_sha256"]
        ):
            raise ValueError("Judge budget response identity mismatch")
        try:
            validate_result(json.loads(capture["body"]))
        except JudgeOutputBudgetExhausted:
            pass
        else:
            raise ValueError("Judge budget receipt lacks length response")
        execution = json.loads((self.grading / "execution.json").read_text())
        if execution.get("error_type") or execution.get("return_code") != 1:
            raise ValueError("Judge execution is not a settled creator failure")
        for run in self.runs:
            lifecycle = json.loads((run / "solver-lifecycle.json").read_text())
            if lifecycle.get("cleanup_confirmed") is not True:
                raise RuntimeError("Solver cleanup unconfirmed")
        # Bind full retained evidence, including creator stderr/traceback and solver outputs.
        artifacts = {}
        for root in [*self.runs, self.grading]:
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.is_symlink() and path.name != "terminal-outcome.json":
                    with path.open("rb") as handle:
                        artifacts[str(path)] = hashlib.file_digest(handle, "sha256").hexdigest()
        record = {
            **{key: value for key, value in body.items() if key not in ("reward", "response")},
            "_ng_failure_class": JUDGE_FAILED,
            "_ng_failure_terminal": True,
            "_ng_failure_judge_error": str(self),
            "failure_stage": "judge",
            "failure_disposition": "judge_output_budget_exhausted",
            "creator_score": None,
            "scientific_score": None,
            "rollout_id": rollout_id,
            "solver_artifact_dirs": [str(run) for run in self.runs],
            "grading_artifact_dir": str(self.grading),
            "artifact_sha256": artifacts,
        }
        _publish(self.grading / "terminal-outcome.json", record)
        return record
