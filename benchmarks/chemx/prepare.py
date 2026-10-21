# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare two newly frozen article tasks using the creator 500-DPI image baseline."""

import base64
import hashlib
import json
import os
from pathlib import Path

from benchmarks.chemx.protocol import creator_definitions, validate_source
from nemo_gym.dataset_staging import stage_file


def prepare() -> Path:
    """Render one page at a time; fail rather than shrink or omit oversized inputs."""
    from pdf2image import convert_from_path
    from pypdf import PdfReader, PdfWriter

    source = Path(os.environ["CHEMX_SOURCE"])
    assets = Path(os.environ["CHEMX_ASSETS"])
    output = Path(os.environ["CHEMX_PREPARED_DIR"])
    if output.is_symlink():
        raise ValueError("Prepared directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("Prepared directory contains symlinks")
    manifest = validate_source(source)
    for name, digest in manifest["assets_sha256"].items():
        path = assets / name
        if path.is_symlink() or not path.resolve().is_relative_to(assets.resolve()):
            raise ValueError("Unsafe PDF asset")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"PDF hash mismatch: {name}")
    definitions = creator_definitions(source)
    schema = json.loads((source / "LLM/data/schemas/magnetic.json").read_text())
    rows, receipts = [], []
    for article in manifest["articles"]:
        directory = output / article["id"]
        directory.mkdir(exist_ok=True)
        writer = PdfWriter()
        counts = []
        for name in article["pdfs"]:
            reader = PdfReader(assets / name)
            counts.append({"file": name, "pages": len(reader.pages)})
            for page in reader.pages:
                pixels = float(page.mediabox.width) * float(page.mediabox.height) * (500 / 72) ** 2
                if pixels > 60_000_000:
                    raise ValueError("500-DPI page exceeds local 60M-pixel admission; do not downsample")
            writer.append(reader)
        merged = directory / "merged.pdf"
        with merged.open("wb") as stream:
            writer.write(stream)
        page_count = len(writer.pages)
        writer.close()
        images, image_receipts = [], []
        total_bytes = 0
        for page_index in range(page_count):
            pages = convert_from_path(
                merged, dpi=500, first_page=page_index + 1, last_page=page_index + 1, thread_count=1
            )
            if len(pages) != 1:
                raise ValueError("Renderer did not return the requested page")
            image_path = directory / f"out_{page_index}.jpg"
            page = pages[0]
            page.save(image_path, "JPEG")
            dimensions = list(page.size)
            page.close()
            data = image_path.read_bytes()
            total_bytes += len(data)
            if total_bytes * 4 / 3 > 64 * 1024**2:
                raise ValueError(
                    "Full 500-DPI inline image input exceeds 64MiB local admission; provider limits unqualified"
                )
            images.append(
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": "data:image/jpg;base64," + base64.b64encode(data).decode(),
                }
            )
            image_receipts.append(
                {
                    "page": page_index + 1,
                    "dimensions": dimensions,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
        rows.append(
            {
                "task_id": article["id"],
                "uuid": article["id"],
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": definitions["PROMPT_DESCRIPTION"] + definitions["INSTRUCTIONS"]},
                        {"role": "user", "content": [{"type": "input_text", "text": definitions["PROMPT"]}, *images]},
                    ],
                    "max_output_tokens": 10000,
                    "text": {"format": {"type": "json_schema", "name": "magnetic", "schema": schema, "strict": True}},
                },
                "verifier_metadata": {
                    "article_id": article["id"],
                    "protocol": manifest["version"],
                    "manifest_sha256": hashlib.sha256(
                        Path(__file__).with_name("frozen.json").read_bytes()
                    ).hexdigest(),
                },
            }
        )
        receipts.append(
            {
                "article_id": article["id"],
                "ordered_pdf_inputs": counts,
                "merged_pages": page_count,
                "image_pages": image_receipts,
                "dpi": 500,
                "image_bytes": total_bytes,
                "provider_capacity_qualified": False,
            }
        )
    candidate = output / "candidate.jsonl"
    candidate.write_text("".join(json.dumps(row) + "\n" for row in rows))
    destination = output / "benchmark.jsonl"
    stage_file(candidate, destination, expected_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
    (output / "render-receipt.json").write_text(json.dumps(receipts, indent=2))
    return destination


if __name__ == "__main__":
    print(prepare())
