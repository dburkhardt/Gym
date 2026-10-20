"""Run native creator chemistry libraries outside the long-lived HTTP process."""
import contextlib
import json
import sys
from resources_servers.ether0_benchmark.grading import grade
if __name__ == '__main__':
    payload = json.load(sys.stdin)
    # Creator native libraries can print diagnostics; reserve stdout for JSON IPC.
    with contextlib.redirect_stdout(sys.stderr):
        result = grade(**payload)
    print("ETHER0_RESULT_JSON:" + json.dumps(result), flush=True)
