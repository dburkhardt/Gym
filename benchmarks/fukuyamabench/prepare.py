# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare the FukuyamaBench benchmark split (sets B and C).

Delegates to the resources server's preparation script rather than duplicating
it, so the benchmark split is produced by the same tested path as the
environment split — including the manifest check that refuses to write a short
corpus.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
SERVER_SCRIPT = REPO_ROOT / "resources_servers" / "fukuyamabench" / "scripts" / "prepare_fukuyamabench.py"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "fukuyamabench_benchmark.jsonl"

# Sets B and C are the tiers carrying upstream's sub-20% published figures; set
# A is a materially easier tier and is not part of this benchmark.
SETS = ("B", "C")


def _load_server_prepare():
    spec = importlib.util.spec_from_file_location("prepare_fukuyamabench", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_frozen(source: Path) -> Path:
    from benchmarks.scientific_agent_campaign.stage_sample import prepare_sample
    from nemo_gym.dataset_staging import stage_file

    # Stage the ORIGINAL source first, so its checksum never describes an
    # intermediate normalized file. Publish only after all metadata validates.
    with tempfile.TemporaryDirectory(prefix="fukuyama-frozen-") as temporary:
        staged = prepare_sample(Path(temporary) / "sample.jsonl", source=source)
        rows = [json.loads(line) for line in staged.read_text().splitlines()]
        for row in rows:
            identity, tier = row.get("uuid"), row.get("stratum")
            if not isinstance(identity, str) or not identity or tier not in ("A", "B", "C"):
                raise ValueError("Frozen Fukuyama requires uuid and A/B/C stratum")
            if not identity.startswith(tier) or row["task_id"] != identity:
                raise ValueError("Frozen Fukuyama identity/stratum mismatch")
            meta = row["verifier_metadata"]
            pathway, checkpoints = meta.get("pathway"), meta.get("checkpoints")
            if not isinstance(pathway, list) or not pathway:
                raise ValueError("Malformed pathway")
            for step in pathway:
                if not isinstance(step, dict) or not isinstance(step.get("products"), list) or not step["products"]:
                    raise ValueError("Malformed pathway products")
                if any(not isinstance(product, str) or not product for product in step["products"]):
                    raise ValueError("Malformed pathway product")
            if not isinstance(checkpoints, list) or not checkpoints:
                raise ValueError("Malformed checkpoints")
            for group in checkpoints:
                if (
                    not isinstance(group, list)
                    or not group
                    or any(type(index) is not int or not 1 <= index <= len(pathway) for index in group)
                ):
                    raise ValueError("Malformed checkpoint indices")
            aliases = dict(gt_pathway=pathway, case_id=identity, case_set=tier, lenient=True, n_gt_steps=len(pathway))
            for key, value in aliases.items():
                if key in meta and meta[key] != value:
                    raise ValueError(f"Conflicting native metadata: {key}")
                meta.setdefault(key, value)
        output = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
        receipt_path = staged.with_suffix(".source.json")
        receipt = json.loads(receipt_path.read_text())
        receipt["output_sha256"] = hashlib.sha256(output).hexdigest()
        receipt["normalizations"].append("fukuyama_native_metadata_aliases")
        staged.write_bytes(output)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        stage_file(staged, OUTPUT_FPATH, expected_sha256=receipt["output_sha256"])
        stage_file(receipt_path, OUTPUT_FPATH.with_suffix(".source.json"))
    return OUTPUT_FPATH


def prepare(limit: int | None = None) -> Path:
    if source := os.environ.get("FUKUYAMA_FROZEN_SAMPLE"):
        if limit is not None:
            raise ValueError("--limit cannot resample a frozen sample")
        return _prepare_frozen(Path(source))
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    server_prepare = _load_server_prepare()

    argv = ["prepare_fukuyamabench", "--output", str(OUTPUT_FPATH), "--sets", *SETS]
    if limit is not None:
        argv += ["--limit", str(limit)]

    original_argv = sys.argv
    try:
        sys.argv = argv
        server_prepare.main()
    finally:
        sys.argv = original_argv
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the FukuyamaBench benchmark split")
    parser.add_argument("--limit", type=int, default=None, help="Max rows, for a smoke subset")
    args = parser.parse_args()
    prepare(limit=args.limit)
