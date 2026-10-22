# SPDX-License-Identifier: Apache-2.0
"""Exact reviewed source transitions; never rewrite historical receipts.

188bc960c -> b395eb6de changes only dockerfile()'s explicit bionic allowlist
for capsule1906954. Capsule8197429's default recipe and all probes are identical.
The full current implementation hash must match this review before using it.
907059fd2 adds only the capsule1683542 and capsule3137115 allowlist branches
relative to b395eb6de. The full819 payload still matches its original digest.
"""

import hashlib
import json
from pathlib import Path


PRIOR_IMPLEMENTATION = "c6e4ce4f173d80780107693cb272307b16952b2492720744b64a358ba57c016d"
REVIEWED_IMPLEMENTATION = "3dd5667fcd8de0d360df04d85d074ff6856dd92c8a41b2fb248432c77c87a403"
PRIOR_QUALIFICATION = "950b338f7d7210f3b40636524c2c0a57ce6059c890ccbe1831fc97a30fe3bb3b"
REVIEWED_819_IMPLEMENTATIONS = frozenset(
    {
        REVIEWED_IMPLEMENTATION,
        "e756392591a45a0cf3afb685b40ee5f552dcd9908b59ed24a5e56352d16b8417",
    }
)


def reviewed_compatibility(row, receipt):
    from benchmarks.corebench import image_controls as controls
    from benchmarks.corebench.prepare import RUNTIME_SHA256

    transition = reviewed_five_retained_controls(row, receipt)
    if transition is not None:
        return transition
    transition = reviewed_201_after_891(row, receipt)
    if transition is not None:
        return transition
    transition = reviewed_313_addition(row, receipt)
    if transition is not None:
        return transition
    implementation = hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest()
    if (
        row["id"] != "capsule-8197429"
        or row.get("harness_install_variant") is not None
        or receipt.get("qualification_sha256") != PRIOR_QUALIFICATION
        or implementation not in REVIEWED_819_IMPLEMENTATIONS
    ):
        return None
    payload = {
        "implementation_sha256": PRIOR_IMPLEMENTATION,
        "schema": "corebench-image-qualification-v2",
        "recipe": controls.recipe_digest(row),
        "source": controls.source_digest(row),
        "instruction": row["instruction_sha256"],
        "reference": row["reference_sha256"],
        "scorer": row["scorer_sha256"],
        "harness_archive": RUNTIME_SHA256,
        "scientific_probe": controls.SCIENTIFIC_PROBE,
        "harness_probe": controls.HARNESS_PROBE,
    }
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != PRIOR_QUALIFICATION:
        return None
    return {
        "review": "core819-default-bionic-allowlist-only-v1",
        "original_qualification_sha256": PRIOR_QUALIFICATION,
        "original_implementation_sha256": PRIOR_IMPLEMENTATION,
        "reviewed_implementation_sha256": implementation,
    }


# Exact eight-line dockerfile allowlist addition only; no probes or previous recipes changed.
PRE_313_ENGINE = "e6da1cd94c28988ee19adbf20b01a4da47e1cda3ae3d6fb032e865f909d9e8f7"
POST_313_ENGINE = "e756392591a45a0cf3afb685b40ee5f552dcd9908b59ed24a5e56352d16b8417"
PRE_313_CAPSULES = frozenset({"capsule-1683542", "capsule-6746514"})


def reviewed_313_addition(row, receipt):
    from benchmarks.corebench import image_controls as controls
    from benchmarks.corebench.prepare import RUNTIME_SHA256

    if (
        row["id"] not in PRE_313_CAPSULES
        or hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest() != POST_313_ENGINE
    ):
        return None
    try:
        recipe = controls.recipe_digest(row)
    except ValueError:
        return None
    payload = {
        "implementation_sha256": PRE_313_ENGINE,
        "schema": "corebench-image-qualification-v2",
        "recipe": recipe,
        "source": controls.source_digest(row),
        "instruction": row["instruction_sha256"],
        "reference": row["reference_sha256"],
        "scorer": row["scorer_sha256"],
        "harness_archive": RUNTIME_SHA256,
        "scientific_probe": controls.SCIENTIFIC_PROBE,
        "harness_probe": controls.HARNESS_PROBE,
    }
    prior = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if receipt.get("qualification_sha256") != prior:
        return None
    return {
        "review": "core313-exact-allowlist-addition-v1",
        "original_qualification_sha256": prior,
        "original_implementation_sha256": PRE_313_ENGINE,
        "reviewed_implementation_sha256": POST_313_ENGINE,
    }


CORE201_PRIOR_QUALIFICATION = "bf0195888eba838520bdbe612206d1d88fdfbc61f9154a78971193516fda12da"
CORE201_REVIEWED_ENGINE = "97f2f6c69fe679588073d131fc82bf2ad70d7bec1d177e70fb5662adfe457248"


def reviewed_201_after_891(row, receipt):
    """Exact 201 transition across unrelated opt-in 891 layout/cache additions.

    cee28bc2e -> b098895b2 changes dockerfile/stage_context only. The complete
    probe, qualification, source/mode and cleanup implementations are unchanged.
    Reconstruct the historical payload using CURRENT content; an engine hash
    alone never accepts changed recipes, probes, sources or scoring identities.
    """
    from benchmarks.corebench import image_controls as controls
    from benchmarks.corebench.prepare import RUNTIME_SHA256

    implementation = hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest()
    if (
        row["id"] != "capsule-2011424"
        or implementation != CORE201_REVIEWED_ENGINE
        or receipt.get("qualification_sha256") != CORE201_PRIOR_QUALIFICATION
        or any(row.get(k) is not None for k in (
            "filesystem_layout_variant", "external_input_variant", "harness_install_variant"
        ))
    ):
        return None
    payload = {
        "implementation_sha256": POST_313_ENGINE,
        "schema": "corebench-image-qualification-v2",
        "recipe": controls.recipe_digest(row),
        "source": controls.source_digest(row),
        "instruction": row["instruction_sha256"],
        "reference": row["reference_sha256"],
        "scorer": row["scorer_sha256"],
        "harness_archive": RUNTIME_SHA256,
        "scientific_probe": controls.SCIENTIFIC_PROBE,
        "harness_probe": controls.HARNESS_PROBE,
    }
    reconstructed = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if reconstructed != CORE201_PRIOR_QUALIFICATION:
        return None
    return {
        "review": "core201-unchanged-probes-after-opt-in-891-v1",
        "original_qualification_sha256": reconstructed,
        "original_implementation_sha256": POST_313_ENGINE,
        "reviewed_implementation_sha256": implementation,
    }


# Exact historical engine, qualification digest, and required recipe variant.
RETAINED_CONTROL_TRANSITIONS = {'capsule-0201225': ('3dd5667fcd8de0d360df04d85d074ff6856dd92c8a41b2fb248432c77c87a403',
                     'b267d7c8cfd770b2b2b7cb96a0ad8332b140726458cf1b30373dbd70072de769',
                     None),
 'capsule-1683542': ('e6da1cd94c28988ee19adbf20b01a4da47e1cda3ae3d6fb032e865f909d9e8f7',
                     'ec38fa99f3060910a93188812c9053668aba090289af2d655289c056b3a6a536',
                     'ubuntu-bionic-isolated-v1'),
 'capsule-1906954': ('3dd5667fcd8de0d360df04d85d074ff6856dd92c8a41b2fb248432c77c87a403',
                     'e7d0e304b457ee6d37d245a53f59d05d8de962b9ad31bd45984e4afd2d73e86f',
                     'ubuntu-bionic-isolated-v1'),
 'capsule-3894632': ('3dd5667fcd8de0d360df04d85d074ff6856dd92c8a41b2fb248432c77c87a403',
                     '84c9ac65d5973c324f2e04b295610ece077e6856581550e919dbdda95f500b00',
                     None),
 'capsule-6746514': ('e6da1cd94c28988ee19adbf20b01a4da47e1cda3ae3d6fb032e865f909d9e8f7',
                     '550439aa379ad9979fe2689ff80244d0ef14434adc1c9436a35964f37df31890',
                     None)}


def reviewed_five_retained_controls(row, receipt):
    """Retain five exact control payloads across unrelated recipe allowlists.

    All source/mode/probe/qualification/cleanup functions are AST-identical to each listed engine. Only dockerfile/stage_context changed; the
    current per-row recipe must reproduce the original complete payload.
    CORE613 is deliberately excluded: its source-probe invocation changed.
    """
    from benchmarks.corebench import image_controls as controls
    from benchmarks.corebench.prepare import RUNTIME_SHA256

    prior = RETAINED_CONTROL_TRANSITIONS.get(row["id"])
    implementation = hashlib.sha256(Path(controls.__file__).read_bytes()).hexdigest()
    if prior is None or implementation != CORE201_REVIEWED_ENGINE:
        return None
    old_engine, original_digest, variant = prior
    if (
        receipt.get("qualification_sha256") != original_digest
        or row.get("harness_install_variant") != variant
        or row.get("filesystem_layout_variant") is not None
        or row.get("external_input_variant") is not None
    ):
        return None
    try:
        recipe = controls.recipe_digest(row)
    except ValueError:
        return None
    payload = {
        "implementation_sha256": old_engine,
        "schema": "corebench-image-qualification-v2",
        "recipe": recipe,
        "source": controls.source_digest(row),
        "instruction": row["instruction_sha256"],
        "reference": row["reference_sha256"],
        "scorer": row["scorer_sha256"],
        "harness_archive": RUNTIME_SHA256,
        "scientific_probe": controls.SCIENTIFIC_PROBE,
        "harness_probe": controls.HARNESS_PROBE,
    }
    reconstructed = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if reconstructed != original_digest:
        return None
    return {
        "review": "core-five-original-content-preserving-transitions-v1",
        "original_qualification_sha256": reconstructed,
        "original_implementation_sha256": old_engine,
        "reviewed_implementation_sha256": implementation,
    }
