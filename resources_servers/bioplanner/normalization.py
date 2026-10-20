"""Explicit transport-format handling; never change generated function arguments."""
import re

def extract_single_code_block(text):
    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    if len(blocks)==1:return blocks[0].strip(),'single_fenced_python_block'
    return text,'raw_no_unambiguous_single_block'

def scaled_edit_distance(distance,gold):
    return None if distance is None else distance/len(gold.split('\n'))
