"""Patch a copied NatureBench verifier for the explicit Nemotron judge variant.

Never run against the immutable downloaded task package. Scientific grading,
judge prompts and verdict parsing remain unchanged.
"""
MODEL = "nvidia/nvidia/nemotron-3.5-super-vl"


def patch_judge(source: str) -> str:
    old = '    return "openai" if model.lower().startswith("gpt-") else "anthropic"'
    new = f'''    if model != {MODEL!r}:
        raise ValueError("Campaign judge model is not allowed")
    return "openai"'''
    assert source.count(old) == 1, "Unexpected upstream provider dispatch"
    source = source.replace(old, new)
    old = "    return resolved_provider, resolved_model, resolved_base_url, resolved_api_key"
    new = f'''    expected_url = os.environ.get("CAMPAIGN_JUDGE_BASE_URL")
    if resolved_model != {MODEL!r} or resolved_provider != "openai":
        raise ValueError("Campaign judge model/provider is not allowed")
    if not expected_url or resolved_base_url.rstrip("/") != expected_url.rstrip("/"):
        raise ValueError("Campaign judge endpoint must be explicitly configured")
{old}'''
    assert source.count(old) == 1, "Unexpected upstream configuration"
    source = source.replace(old, new)
    old = '    output = payload.get("output", []) if isinstance(payload, dict) else []'
    new = f'''    if not isinstance(payload, dict) or payload.get("model") != {MODEL!r}:
        return False, "judge_response_model_mismatch"
    if payload.get("status") != "completed":
        return False, "judge_response_incomplete"
{old}'''
    assert source.count(old) == 1, "Unexpected upstream response parser"
    return source.replace(old, new)
