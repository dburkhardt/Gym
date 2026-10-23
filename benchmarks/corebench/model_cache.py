# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit reconstructed MiniLM cache; creator historical revision is unknown."""

from pathlib import Path


VARIANT = "3249574-minilm-7dbbc903-v1"
CREATOR = "registry.codeocean.com/published/e1a2bd8d-c6b2-4ddb-bde5-01301ec7e823@sha256:77ca489fc57bb923f33512ef89c993f0e705a93089ec53b1f2e4fbb51ddf441c"
FILES = {
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "config_sentence_transformers.json": "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "pytorch_model.bin": "c3a85f238711653950f6a79ece63eb0ea93d76f6a6284be04019c53733baf256",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def recipe(row: dict[str, object]) -> str:
    if (
        row.get("external_input_variant") != VARIANT
        or row["id"] != "capsule-3249574"
        or row["published_image"] != CREATOR
    ):
        raise ValueError("Unsupported pinned model cache variant")
    target = "/model-cache/sentence-transformers_all-MiniLM-L6-v2"
    return (
        "# Reconstructed pinned snapshot; historical creator revision unspecified\n"
        "COPY model-cache/ "
        + target
        + "/\n"
        + "".join(f"RUN echo '{sha}  {target}/{name}' | sha256sum -c -\n" for name, sha in FILES.items())
        + f"RUN chmod 0755 /model-cache && find {target} -type d -exec chmod 0755 {{}} + && find {target} -type f -exec chmod 0644 {{}} +\n"
        + "ENV SENTENCE_TRANSFORMERS_HOME=/model-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
    )


def stage(source: Path, destination: Path) -> None:
    import os
    from tempfile import TemporaryDirectory

    from nemo_gym.dataset_staging import stage_file

    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Pinned model cache requires a directory")
    actual = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(FILES) or any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Pinned model cache tree differs")
    published = destination / "model-cache"
    if published.exists() or published.is_symlink():
        raise ValueError("Refusing existing model cache destination")
    with TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        clean = temporary / "clean"
        clean.mkdir()
        for index, (name, sha) in enumerate(FILES.items()):
            checked = temporary / "checked" / str(index)
            stage_file(source / name, checked, expected_sha256=sha)
            target = clean / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(checked, target)
        # Caller owns the surrounding stage_context lock. Publish only after all
        # files verify, so an interrupted or rejected input cannot leave leftovers.
        os.rename(clean, published)
