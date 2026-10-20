"""Validate already frozen creator rows; no data acquisition or resampling."""
import json,os
from pathlib import Path
p=Path(os.environ['GENEHOP_SAMPLE'])
for line in p.read_text().splitlines():
    row=json.loads(line)
    assert row['responses_create_params']['input']==[{'role':'user','content':row['verifier_metadata']['question']}]
