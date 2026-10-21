# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from resources_servers.scicode.measurement_validity import review_substep


def test_review_binds_full_step_and_canonical_data():
    step = json.loads((Path(__file__).parent / "fixtures/step79_4.json").read_text())
    digest = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
    assert review_substep(step, data_sha256=digest)["status"] == "unresolved"
    assert review_substep(step, data_sha256="changed") is None
    step["test_cases"][0] += "\n# changed scientific assertion"
    assert review_substep(step, data_sha256=digest) is None


def test_bounded_broadcast_proof():
    import numpy as np

    flat = np.array([1.0, 2.0, 3.0])
    column = flat[:, None]
    assert np.broadcast_shapes(column.shape, flat.shape) == (3, 3)
    assert not np.allclose(column, flat)
    assert np.allclose(column[:, 0], flat)
    assert not np.allclose((column + 1)[:, 0], flat)
    # Shape-only calculation: never allocate the 40000 x 40000 comparison.
    assert np.broadcast_shapes((40000, 1), (40000,)) == (40000, 40000)
