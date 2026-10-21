# SPDX-License-Identifier: Apache-2.0
"""Stage frozen FlyAOC task inputs without exposing reference annotations."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.dataset_staging import stage_file


def prepare(source: Path | None = None, destination: Path | None = None, sample_size: int | None = None) -> Path:
    source = Path(source) if source is not None else Path(os.environ["FLYAOC_FROZEN_INPUTS"])
    destination = (
        Path(destination)
        if destination is not None
        else Path(os.environ.get("FLYAOC_DATA", str(Path(__file__).parent / "data")))
    )
    sample_size = sample_size if sample_size is not None else int(os.environ.get("FLYAOC_SAMPLE_SIZE", "5"))
    if sample_size not in (5, 20, 50):
        raise ValueError("Use frozen sample sizes 5, 20 or 50")
    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["input_sha256"]:
        raise ValueError("Frozen FlyAOC input checksum mismatch")
    records = [json.loads(line) for line in content.splitlines() if line.strip()]
    if [r["gene_id"] for r in records] != manifest["frozen_task_ids"]:
        raise ValueError("Frozen FlyAOC task ordering mismatch")
    rows = []
    for record in records[:sample_size]:
        payload = {key: record[key] for key in ("gene_id", "gene_symbol", "pmcids", "summary")}
        rows.append(
            {
                "task_id": record["gene_id"],
                "responses_create_params": {"input": [{"role": "user", "content": json.dumps(payload)}]},
                "verifier_metadata": {"gene_id": record["gene_id"]},
                "agent_ref": {"type": "responses_api_agents", "name": "flyaoc_agent"},
            }
        )
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        data = temporary / "benchmark.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in rows))
        published = stage_file(data, destination / data.name)
        receipt = temporary / "benchmark.source.json"
        receipt.write_text(
            json.dumps(
                {
                    "dataset_revision": manifest["dataset_revision"],
                    "input_sha256": manifest["input_sha256"],
                    "output_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                    "task_ids": [row["task_id"] for row in rows],
                    "status": "prepared_only; maintained native agent/runtime migration pending",
                },
                indent=2,
            )
            + "\n"
        )
        stage_file(receipt, destination / receipt.name)
    return published


if __name__ == "__main__":
    print(
        prepare(
            Path(os.environ["FLYAOC_FROZEN_INPUTS"]),
            Path(os.environ.get("FLYAOC_DATA", str(Path(__file__).parent / "data"))),
            int(os.environ.get("FLYAOC_SAMPLE_SIZE", "5")),
        )
    )
