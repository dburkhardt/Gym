# SPDX-License-Identifier: Apache-2.0
"""Creator parsed-only accuracy and all-attempt strict diagnostic."""
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1])]
valid=[r for r in rows if r['scoring_details']['valid']]
print(json.dumps({'creator_accuracy':sum(r['reward'] for r in valid)/len(valid) if valid else None,'n_parsed':len(valid),'n_total':len(rows),'all_attempt_accuracy':sum(r['reward'] for r in rows)/len(rows)}))
