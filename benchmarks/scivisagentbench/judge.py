# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Receipt-preserving client for native SciVis evaluation; no scoring replacement."""
import json
from pathlib import Path
from types import SimpleNamespace
from benchmarks.scivisagentbench.transport import MODEL, validate_response

class JudgeClient:
    def __init__(self, sender, path: Path):
        self.sender, self.path = sender, Path(path)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        if request.get('model') != MODEL:
            raise ValueError('requested_judge_model_mismatch')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as f:
            f.write(json.dumps({'phase': 'dispatch', 'role': 'judge', 'request': request})+'\n')
        try:
            response = self.sender(request)
        except Exception as exc:
            with self.path.open('a') as f:
                f.write(json.dumps({'phase': 'error', 'error': repr(exc)})+'\n')
            raise
        with self.path.open('a') as f:
            f.write(json.dumps({'phase': 'response', 'response': response})+'\n')
        validate_response(response)
        if response['choices'][0]['message'].get('tool_calls'):
            raise ValueError('judge_must_return_text')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response['choices'][0]['message']['content']))])


def score_validity(parsed: dict, *, modality: str, goals: int) -> dict:
    """Separate validity annotation; retain creator parsed output and raw scores."""
    if 'error' in parsed:
        return {'valid': False, 'failure': 'judge_parse_or_transport_failed'}
    key = 'score' if modality == 'text' else 'total_score'
    value = parsed.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= goals*10:
        return {'valid': False, 'failure': 'missing_or_out_of_range_score'}
    if modality == 'vision':
        values = [parsed.get(f'goal_{i}_score') for i in range(1, goals+1)]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 10 for v in values):
            return {'valid': False, 'failure': 'incomplete_goal_coverage'}
        if abs(sum(values)-value) > 1e-6:
            return {'valid': False, 'failure': 'inconsistent_goal_total'}
    return {'valid': True, 'score': value, 'max_score': goals*10}
