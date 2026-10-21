# SPDX-License-Identifier: Apache-2.0
"""Creator regrade in the isolated verifier; never executes saved model code."""

import json
from pathlib import Path

from regrade import regrade


request = json.loads(Path("/results/request.json").read_text())
result = regrade(Path("/creator"), request["task_id"], Path("/workspace"), missing_result_is_model_output=True)
Path("/results/score.json").write_text(json.dumps(result, indent=2) + "\n")
