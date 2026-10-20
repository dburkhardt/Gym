# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare only the pinned public evaluation set (not the SFT corpus)."""

import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download

from nemo_gym.dataset_staging import stage_file


REVISION = "f1397963c7f727a4a2f00cdd691e6e219c36e992"
SHA256 = "f2dce5c1b54e8ce73897a4af48ae8ba05507076d511b9d02d57cddf31c42ed96"
URL = f"https://huggingface.co/datasets/tsynbio/ProteinLMBench/resolve/{REVISION}/ProteinLMBench.json"
OUTPUT = Path("resources_servers/proteinlmbench/data/benchmark.jsonl")
PROMPT = """
Answer the multiple-choice question based solely on the provided context.\x20
If you are still unsure about the answer, output option 7.
Select only ONE correct option by its number. Start your response with 'The correct option is' followed by the option number ONLY. eg: "The correct option is Option X."
Think step by step.
    """


def convert(source: list[dict]) -> list[dict]:
    """Preserve release order, question text, variable options, and gold labels."""
    rows = []
    for i, item in enumerate(source):
        options = item["options"]
        expected = int(re.fullmatch(r"option (\d+)", item["answer"]).group(1))
        if not 1 <= expected <= len(options):
            raise ValueError(f"Invalid answer at row {i}")
        prompt = (
            PROMPT
            + "\n Question: \n"
            + item["question"]
            + "\n Options: \n"
            + "".join(o + "\n" for o in options)
            + "\nThe correct option is:"
        )
        rows.append(
            {
                "id": f"proteinlmbench-{i:04d}",
                "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
                "verifier_metadata": {"expected_answer": expected, "option_count": len(options)},
            }
        )
    return rows


def prepare() -> Path:
    """Download, checksum, validate, and write 944 deterministic evaluation tasks."""
    cached = Path(
        hf_hub_download(
            repo_id="tsynbio/ProteinLMBench",
            repo_type="dataset",
            filename="ProteinLMBench.json",
            revision=REVISION,
        )
    )
    source_path = OUTPUT.parent / "sources" / REVISION / "ProteinLMBench.json"
    stage_file(cached, source_path, expected_sha256=SHA256)
    payload = source_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("ProteinLMBench checksum mismatch; refusing changed source")
    source = json.loads(payload)
    if len(source) != 944:
        raise ValueError("Expected 944 public evaluation questions")
    rows = convert(source)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Conversion is deterministic; publish only a complete, validated task file.
    with TemporaryDirectory(dir=OUTPUT.parent) as temporary:
        converted = Path(temporary) / "tasks.jsonl"
        converted.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        stage_file(converted, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(prepare())
