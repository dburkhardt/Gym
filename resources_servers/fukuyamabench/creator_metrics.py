# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator functions copied from HaCTang/ReactionMechanismReasoning.

Source: eval/eval_infer_pathway.py at 63bb79f912f0b2de593996b80ffeea894f6f1a59.
Upstream Apache-2.0 license retained in CREATOR_LICENSE. Functions intentionally
preserve upstream parsing, invalid-component dropping and gold canonicalization.
Only formatting/type annotation modernization by repository lint is permitted.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem


def remove_atom_mapping(smiles: str) -> str:
    """
    Remove atom mapping numbers from SMILES.
    e.g., [CH3:1][O:2] -> [CH3][O] -> CO
    """
    if not smiles:
        return smiles

    # Pattern to match atom mapping numbers like :1, :2, :10, etc.
    # Handles both [C:1] and [CH3:1] formats
    pattern = r":\d+"
    return re.sub(pattern, "", smiles)


def get_checkpoint_products(gt_pathway: List[Dict], step_ids: List[int]) -> List[List[str]]:
    """
    Get products from ground truth pathway for given step IDs.

    Args:
        gt_pathway: Ground truth pathway steps
        step_ids: List of step IDs (1-indexed) that are equivalent for this checkpoint

    Returns:
        List of product lists, one for each step ID
    """
    result = []
    for step_id in step_ids:
        idx = step_id - 1  # Convert to 0-indexed
        if 0 <= idx < len(gt_pathway):
            products = gt_pathway[idx].get("products", [])
            result.append(products)
    return result


def extract_pathway_from_response(text: str) -> Optional[List[Dict]]:
    """
    Extract pathway steps from model output.
    Expected format: JSON array with step objects containing product_smiles.
    """
    if not text:
        return None

    # Method 1: Find JSON array in ```json ... ``` code block
    json_blocks = re.findall(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    for json_str in reversed(json_blocks):
        try:
            result = json.loads(json_str)
            if isinstance(result, list) and len(result) > 0:
                # Validate structure
                if any(isinstance(step, dict) and ("product_smiles" in step or "products" in step) for step in result):
                    return result
        except json.JSONDecodeError:
            continue

    # Method 2: Find any JSON array in the text (after ## Result section)
    result_section = re.search(r"##\s*Result\s*\n", text)
    search_text = text[result_section.end() :] if result_section else text

    # Find array start positions
    start_positions = [m.start() for m in re.finditer(r"\[", search_text)]

    for start in start_positions:
        bracket_count = 0
        for i in range(start, len(search_text)):
            if search_text[i] == "[":
                bracket_count += 1
            elif search_text[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    try:
                        candidate = search_text[start : i + 1]
                        result = json.loads(candidate)
                        if isinstance(result, list) and len(result) > 0:
                            if any(
                                isinstance(step, dict) and ("product_smiles" in step or "products" in step)
                                for step in result
                            ):
                                return result
                    except json.JSONDecodeError:
                        pass
                    break

    # Method 3: Try to parse individual step objects
    steps = []
    step_matches = re.finditer(r'\{[^{}]*"step_id"[^{}]*\}', text, re.DOTALL)
    for match in step_matches:
        try:
            step = json.loads(match.group())
            if "product_smiles" in step or "products" in step:
                steps.append(step)
        except json.JSONDecodeError:
            continue

    if steps:
        return steps

    return None


def get_product_smiles_from_step(step: Dict) -> List[str]:
    """
    Extract product SMILES from a step dictionary.
    Handles atom-mapped SMILES by removing mapping numbers.
    """
    if "product_smiles" in step:
        smiles = step["product_smiles"]
        if isinstance(smiles, str):
            # Remove atom mapping and split by '.' for multiple molecules
            unmapped = remove_atom_mapping(smiles)
            return [s.strip() for s in unmapped.split(".") if s.strip()]
        elif isinstance(smiles, list):
            # If already a list, remove mapping from each
            return [remove_atom_mapping(s) for s in smiles if s]
    elif "products" in step:
        products = step["products"]
        if isinstance(products, list):
            result = []
            for p in products:
                if isinstance(p, str):
                    result.append(remove_atom_mapping(p))
                elif isinstance(p, dict) and "smiles" in p:
                    result.append(remove_atom_mapping(p["smiles"]))
            return result
    return []


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES using RDKit."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonicalize_smiles_set(smiles_list: List[str]) -> set:
    """Canonicalize a list of SMILES and return as set."""
    result = set()
    for smi in smiles_list:
        canon = canonicalize_smiles(smi)
        if canon:
            result.add(canon)
    return result


def compare_step_products(pred_products: List[str], gt_products: List[str], lenient: bool = True) -> Tuple[bool, str]:
    """
    Compare predicted products with ground truth products.

    Args:
        pred_products: List of predicted SMILES
        gt_products: List of ground truth SMILES
        lenient: If True, prediction is correct if it's a subset of ground truth
                 (i.e., model predicted main products but not leaving groups)

    Returns:
        Tuple of (is_match, error_type)
    """
    pred_canon = canonicalize_smiles_set(pred_products)
    gt_canon = canonicalize_smiles_set(gt_products)

    if not pred_canon and not gt_canon:
        return False, "both_invalid"
    if not pred_canon:
        return False, "invalid_pred"
    if not gt_canon:
        return False, "invalid_gt"

    # Check if predicted products match ground truth exactly
    if pred_canon == gt_canon:
        return True, None

    # Lenient mode: prediction is correct if it's a non-empty subset of ground truth
    # This handles cases where model predicts main products but not leaving groups
    if lenient and pred_canon.issubset(gt_canon) and len(pred_canon) > 0:
        return True, "subset_match"

    # Check if prediction is a superset (model predicted extra products)
    if gt_canon.issubset(pred_canon):
        return False, "superset_match"

    # Partial overlap
    if pred_canon & gt_canon:  # non-empty intersection
        return False, "partial_match"

    return False, "mismatch"


def compare_pathways_with_checkpoints(
    pred_pathway: List[Dict], gt_pathway: List[Dict], checkpoints: List[List[int]], lenient: bool = True
) -> Dict:
    """
    Compare predicted pathway with ground truth using checkpoint-based evaluation.

    This method allows model to predict finer-grained steps than ground truth.
    Each checkpoint can be satisfied by matching any of its equivalent step IDs.
    Checkpoints must be matched in order using the predicted steps in order.

    Args:
        pred_pathway: List of predicted step dictionaries
        gt_pathway: List of ground truth step dictionaries
        checkpoints: List of checkpoints, each containing equivalent step IDs
        lenient: If True, allow subset matching for products

    Returns:
        Dictionary with comparison metrics
    """
    num_checkpoints = len(checkpoints)
    pred_length = len(pred_pathway)

    checkpoint_matches = []
    checkpoints_correct = 0

    # Track which predicted step matched each checkpoint
    pred_idx = 0  # Current position in predicted pathway

    for ckpt_idx, ckpt_step_ids in enumerate(checkpoints):
        # Get all possible products for this checkpoint (from equivalent GT steps)
        possible_products_list = get_checkpoint_products(gt_pathway, ckpt_step_ids)

        ckpt_match = {
            "checkpoint": ckpt_idx + 1,
            "equivalent_steps": ckpt_step_ids,
            "match": False,
            "matched_pred_step": None,
            "matched_gt_step": None,
            "error": None,
        }

        # Search through remaining predicted steps in order
        found = False
        while pred_idx < pred_length and not found:
            pred_products = get_product_smiles_from_step(pred_pathway[pred_idx])

            # Check if this predicted step matches any equivalent GT step
            for gt_step_id, gt_products in zip(ckpt_step_ids, possible_products_list):
                is_match, error_type = compare_step_products(pred_products, gt_products, lenient)

                if is_match:
                    ckpt_match["match"] = True
                    ckpt_match["matched_pred_step"] = pred_idx + 1
                    ckpt_match["matched_gt_step"] = gt_step_id
                    ckpt_match["pred_products"] = pred_products
                    ckpt_match["gt_products"] = gt_products
                    checkpoints_correct += 1
                    found = True
                    pred_idx += 1  # Move to next predicted step
                    break

            if not found:
                pred_idx += 1  # Try next predicted step

        if not found:
            ckpt_match["error"] = "checkpoint_not_found"

        checkpoint_matches.append(ckpt_match)

    # All checkpoints passed = exact match
    exact_match = checkpoints_correct == num_checkpoints

    return {
        "valid": True,
        "exact_match": exact_match,
        "length_match": pred_length == len(gt_pathway),
        "steps_correct": checkpoints_correct,  # For backward compatibility
        "checkpoints_correct": checkpoints_correct,
        "steps_total": len(gt_pathway),
        "checkpoints_total": num_checkpoints,
        "pred_length": pred_length,
        "step_accuracy": checkpoints_correct / num_checkpoints if num_checkpoints > 0 else 0,
        "checkpoint_accuracy": checkpoints_correct / num_checkpoints if num_checkpoints > 0 else 0,
        "step_matches": checkpoint_matches,  # For backward compatibility
        "checkpoint_matches": checkpoint_matches,
        "use_checkpoints": True,
    }
