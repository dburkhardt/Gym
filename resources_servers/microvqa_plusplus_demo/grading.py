"""Explicit demo-only strict numbered-choice variant; no creator scorer released."""
import re

def grade(text: str, *, correct_index: int, **metadata: object) -> dict:
    value=text.strip()
    predicted=int(value) if re.fullmatch(r"[1-9]",value) else None
    return {'reward':float(predicted==correct_index),'diagnostics':{'predicted_index':predicted,'parse_ok':predicted is not None,'protocol':'15-row released demo, strict numbered-choice variant'}}
