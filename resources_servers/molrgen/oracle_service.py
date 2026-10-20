# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serialized pinned creator scorer, CPU Vina1.1.2 variant; raw oracle receipts retained."""

import json
import math
import pathlib
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import ray
from molrgen.reward import (
    GenerationVerifierConfigModel,
    MolecularVerifier,
    MolecularVerifierConfigModel,
    PyscreenerConfigModel,
)


ray.init(num_cpus=3, object_store_memory=100 * 1024 * 1024, include_dashboard=False)
config = MolecularVerifierConfigModel(
    parsing_method="boxed",
    reward="property",
    generation_verifier_config=GenerationVerifierConfigModel(
        path_to_mappings="/data/molgendata",
        rescale=True,
        generation_verifier_ncpus=1,
        oracle_kwargs=PyscreenerConfigModel(
            docking_oracle="pyscreener", software_class="vina", n_cpu=2, exhaustiveness=8
        ),
    ),
)
scorer = MolecularVerifier(config)
audit = pathlib.Path("/control/oracle-values.jsonl")
receipts = pathlib.Path("/control/service-receipts.jsonl")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ready","variant":"CPU Vina1.1.2"}')

    def do_POST(self):
        if self.path != "/get_reward":
            self.send_error(404)
            return
        start = time.time()
        before = len(audit.read_text().splitlines()) if audit.exists() else 0
        result = {"reward": None, "oracle_variant": "CPU PyScreener/Vina1.1.2", "config": config.model_dump()}
        try:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert len(request["query"]) == len(request["metadata"]) == 1
            result["request"] = request
            result["creator_response"] = scorer.get_score(request["query"], request["metadata"]).model_dump()
            raw = [json.loads(x) for x in audit.read_text().splitlines()[before:]] if audit.exists() else []
            result["raw_oracle_values"] = raw
            assert all(
                v is not None and isinstance(v, (float, int)) and math.isfinite(v)
                for row in raw
                for v in row["scores"]
            ), "Oracle returned missing/nonfinite value"
            value = result["creator_response"]["rewards"][0]
            assert math.isfinite(value)
            result["reward"] = value
        except Exception as exc:
            result["error"] = repr(exc)
            result["traceback"] = traceback.format_exc()
        result["elapsed_seconds"] = time.time() - start
        with receipts.open("a") as f:
            f.write(json.dumps(result) + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())


HTTPServer(("0.0.0.0", 37000), Handler).serve_forever()
