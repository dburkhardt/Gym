# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, offline creator-tokenizer preparation; never replace frozen variants."""

import base64
import hashlib
import io
import json
import os
import zipfile
from importlib.metadata import version
from pathlib import Path

from benchmarks.csmed_ft.creator_protocol import PROTOCOL
from nemo_gym.dataset_staging import stage_file


TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
WINDOWS = {"gpt-4-0314": 8000, "gpt-3.5-turbo-0301": 4000, "gpt-3.5-turbo-16k": 16000}
QUESTION = (
    "Does the following scientific paper fulfill all eligibility criteria and should it be included in the systematic review? "
    "Answer 'Included' or 'Excluded'."
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_recipe(config: dict):
    """Require exact declared dependencies and cached tokenizer; no download fallback."""
    if config.get("recipe_model") not in WINDOWS:
        raise ValueError("Unknown creator tokenizer/context recipe; explicit supported recipe_model required")
    if config.get("application") != "creator-tokenizer-adaptation":
        raise ValueError("Explicit creator-tokenizer-adaptation declaration required; not Nemotron tokenizer parity")
    for package, expected in (("tiktoken", "0.4.0"), ("langchain", "0.0.191")):
        if version(package) != expected:
            raise ValueError(f"Require pinned {package}=={expected}")
    cache_dir = Path(config["tokenizer_cache_dir"]).resolve()
    if os.environ.get("TIKTOKEN_CACHE_DIR") != str(cache_dir):
        raise ValueError("TIKTOKEN_CACHE_DIR must explicitly match the declared offline asset directory")
    cache = cache_dir / hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()
    if digest(cache.read_bytes()) != config["tokenizer_asset_sha256"]:
        raise ValueError("Tokenizer asset hash mismatch")
    import tiktoken
    from langchain.text_splitter import TokenTextSplitter

    encoding = tiktoken.encoding_for_model(config["recipe_model"])
    if encoding.name != "cl100k_base" or encoding._mergeable_ranks != {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in cache.read_bytes().splitlines() if line)
    }:
        raise ValueError("Resolved encoding does not match declared cached tokenizer")
    return encoding, TokenTextSplitter


def chunk_paper(example: dict, *, recipe_model: str, encoding, splitter) -> list[str]:
    """Exact creator prompt construction, with original pre-truncation review length."""
    context = WINDOWS[recipe_model]
    review = ("Systematic review:\n" + example["review_title"] + "\n" + example["review_criteria"]).strip()
    paper = (
        "Candidate paper:\n" + example["title"] + "\n\n" + example["abstract"] + "\n\n" + example["main_text"]
    ).strip()
    review_splitter = splitter(chunk_size=context // 2, chunk_overlap=0, model_name=recipe_model)
    paper_splitter = splitter(
        chunk_size=max(context // 2, context - len(encoding.encode(review))),
        chunk_overlap=0,
        model_name=recipe_model,
    )
    review = review_splitter.split_text(review)[0]
    return [QUESTION + "\n\n" + review + "\n\n" + chunk + "\n\nAnswer: " for chunk in paper_splitter.split_text(paper)]


def prepare(config: dict, *, output: Path) -> Path:
    """Materialize selected paper/chunk identities from hashed original data."""
    generation = config.get("generation_protocol", {})
    if not all(field in generation for field in ("model", "temperature", "max_output_tokens")):
        raise ValueError("Explicit generation model/temperature/output cap required for adaptation")
    encoding, splitter = load_recipe(config)
    source = Path(config["dataset_zip"]).read_bytes()
    selected = Path(config["frozen_selection"]).read_bytes()
    if digest(source) != config["dataset_sha256"] or digest(selected) != config["selection_sha256"]:
        raise ValueError("Dataset or frozen selection hash mismatch")
    import pandas as pd
    from datasets import Value

    for package, expected in (("pandas", "1.5.3"), ("datasets", "2.8.0")):
        if version(package) != expected:
            raise ValueError(f"Require qualified creator loader {package}=={expected}")
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        papers = pd.read_csv(archive.open("CSMeD-FT/CSMeD-FT-test.csv")).to_dict("records")
        reviews = json.loads(archive.read("CSMeD-FT/CSMeD-FT-test_reviews_metadata.json"))
    by_identity = {}
    for index, paper in enumerate(papers):
        key = (paper["document_id"], paper["review_id"])
        if key in by_identity:
            raise ValueError("Ambiguous source paper/review identity")
        by_identity[key] = (index, paper)
    selection = [json.loads(line)["verifier_metadata"] for line in selected.splitlines() if line.strip()]
    keys = [(row["task_id"], row["review_id"]) for row in selection]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate selected paper identity")
    binding = digest(json.dumps(config, sort_keys=True).encode())
    rows = []
    for meta, key in zip(selection, keys):
        index, paper = by_identity[key]
        review = reviews[paper["review_id"]]
        if (
            meta["answer"] != paper["decision"]
            or meta.get("input_valid") is not True
            or pd.isna(paper["main_text"])
            or not paper["main_text"]
            or not review["criteria_text"]
        ):
            raise ValueError(f"Invalid or mismatched frozen source input: {key}")
        # Creator pandas -> datasets.Value("string") emits literal "nan" for
        # missing title/abstract; empty-string CSV normalization would change prompts.
        string = Value("string")
        example = {key: string.encode_example(paper[key]) for key in ("title", "abstract", "main_text")}
        example.update(
            review_title=string.encode_example(review["title"]),
            review_criteria=string.encode_example(review["criteria_text"]),
        )
        prompts = chunk_paper(example, recipe_model=config["recipe_model"], encoding=encoding, splitter=splitter)
        if not prompts:
            raise ValueError(f"No chunks for {key}")
        for ordinal, prompt in enumerate(prompts):
            identity = json.dumps([*key, ordinal], separators=(",", ":"))
            rows.append(
                {
                    "task_id": identity,
                    "responses_create_params": {
                        "input": [{"role": "user", "content": prompt}],
                        "temperature": generation["temperature"],
                        "max_output_tokens": generation["max_output_tokens"],
                    },
                    "verifier_metadata": {
                        "task_id": identity,
                        "paper_id": key[0],
                        "review_id": key[1],
                        "uuid": index,
                        "paper_chunk_id": ordinal,
                        "chunk_count": len(prompts),
                        "answer": paper["decision"],
                        "input_valid": True,
                        "protocol": PROTOCOL,
                        "recipe_sha256": binding,
                        "generation_protocol": generation,
                        "prompt_sha256": digest(prompt.encode()),
                    },
                }
            )
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / "chunks.jsonl"
        staged.write_text("".join(json.dumps(row) + "\n" for row in rows))
        stage_file(staged, output, expected_sha256=digest(staged.read_bytes()))
        receipt = Path(temporary) / "receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "config": config,
                    "recipe_sha256": binding,
                    "protocol": PROTOCOL,
                    "output_sha256": digest(staged.read_bytes()),
                    "preparation_sha256": digest(Path(__file__).read_bytes()),
                    "runtime_versions": {
                        name: version(name) for name in ("tiktoken", "langchain", "pandas", "datasets")
                    },
                    "loader_missing_string_semantics": "pandas_NaN_to_datasets_Value_string_literal_nan",
                    "papers": len(keys),
                    "chunks": len(rows),
                },
                indent=2,
            )
        )
        stage_file(receipt, output.with_suffix(".source.json"))
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(json.loads(args.recipe.read_text()), output=args.output))
