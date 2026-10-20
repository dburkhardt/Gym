# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from resources_servers.ether0_benchmark.grading import grade


def test_controls():
    solution = "product_eval!:!CCCO!:!reaction-prediction"
    assert grade("<answer>CCCO</answer>", solution=solution)["reward"] == 1
    assert grade("<answer>OCCC</answer>", solution=solution)["reward"] == 1
    assert grade("<answer>CCCC</answer>", solution=solution)["reward"] == 0
    assert grade("", solution=solution)["reward"] == 0
    assert grade("<answer>this is not smiles</answer>", solution=solution)["reward"] == 0
