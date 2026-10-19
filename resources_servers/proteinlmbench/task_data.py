# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-only ProteinLMBench labels."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class TaskData(BaseModel):
    expected_answer: int = Field(ge=1, le=10, json_schema_extra={"legacy_location": "verifier_metadata"})
    option_count: int = Field(ge=2, le=10, json_schema_extra={"legacy_location": "verifier_metadata"})

    @model_validator(mode="after")
    def valid_label(self) -> Self:
        if self.expected_answer > self.option_count:
            raise ValueError("Expected answer exceeds option count")
        return self
