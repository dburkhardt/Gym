"""Run native creator chemistry libraries outside the long-lived HTTP process."""
import json
import sys
from resources_servers.ether0_benchmark.grading import grade
if __name__ == '__main__':
    print(json.dumps(grade(**json.load(sys.stdin))))
