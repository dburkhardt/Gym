# SPDX-License-Identifier: Apache-2.0
"""Evidence-bound permission to qualify an OOM-limited creator runtime.

A successful workflow control is not image qualification or a solver score.
Network, reference isolation and harness checks remain mandatory image qualification.
Artifacts remain external, hash-bound files and must be available at validation.
"""
import hashlib
import json
from pathlib import Path


def validate_resource_review(row, resources, review):
    from benchmarks.corebench.image_controls import source_digest

    if not isinstance(review, dict) or review.get('version') != 'core-resource-review-v1':
        raise ValueError('Observed OOM requires an evidence-bound resource review')
    expected = {
        'capsule': row['id'],
        'creator_image': row['published_image'],
        'source_sha256': source_digest(row),
        'runtime_environment_variant': row['runtime_environment_variant'],
    }
    if any(review.get(k) != v for k, v in expected.items()):
        raise ValueError('Resource review capsule/image/source/environment differs')
    artifacts = review.get('artifacts', {})
    required = {'binding', 'result', 'lifecycle', 'supervisor', 'container', 'source_manifest', 'memory'}
    if set(artifacts) != required:
        raise ValueError('Resource review artifact set incomplete')
    evidence = {}
    for role in sorted(required):
        item = artifacts[role]
        if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
            raise ValueError('Resource review artifact binding malformed')
        try:
            with Path(item['path']).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != item['sha256']:
                raise ValueError('Resource review artifact hash differs: ' + role)
            evidence[role] = json.loads(Path(item['path']).read_text())
        except (OSError, TypeError, json.JSONDecodeError) as error:
            raise ValueError('Resource review artifact unavailable: ' + role) from error
    binding = evidence['binding']
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError('Control binding capsule/image/source/environment differs')
    attempt = review.get('attempt_id')
    if not isinstance(attempt, str) or not attempt or binding.get('attempt_id') != attempt:
        raise ValueError('Resource review attempt identity differs')
    if binding.get('control_artifacts') != {k: artifacts[k]['sha256'] for k in required - {'binding'}}:
        raise ValueError('Control evidence set differs from attempt binding')
    control = binding.get('resources', {})
    if (
        type(control.get('cpus')) is not int or control['cpus'] <= 0
        or type(control.get('memory_mb')) is not int or control['memory_mb'] <= 0
        or type(resources.get('cpus')) is not int or resources['cpus'] != control['cpus']
        or type(resources.get('memory_mb')) is not int or resources['memory_mb'] < control['memory_mb']
    ):
        raise ValueError('Resource review CPU/memory allocation insufficient or different')
    actual_manifest = evidence['source_manifest']
    expected_manifest = {x['path']: x['sha256'] for x in row['medium_file_manifest']}
    if actual_manifest != expected_manifest:
        raise ValueError('Observed control source manifest differs')
    result, lifecycle, supervisor = (evidence[k] for k in ('result', 'lifecycle', 'supervisor'))
    if (not {'return_code', 'error_type', 'model_calls', 'scientific_score'} <= result.keys()
        or type(result['return_code']) is not int or result['return_code'] != 0 or result['error_type'] is not None
        or type(result['model_calls']) is not int or result['model_calls'] != 0 or result['scientific_score'] is not None):
        raise ValueError('Resource control is not a successful model-free workflow')
    if (lifecycle.get('cleanup_confirmed') is not True or supervisor.get('cleanup_confirmed') is not True
        or type(supervisor.get('return_code')) is not int or supervisor['return_code'] != 0
        or 'error' not in supervisor or supervisor['error'] is not None
        or supervisor.get('timed_out') is not False or supervisor.get('harness_forced_kill') is not False):
        raise ValueError('Resource control terminal/cleanup incomplete')
    container = evidence['container']
    sandbox_id = lifecycle.get('handle', {}).get('sandbox_id')
    if (not isinstance(sandbox_id, str) or not sandbox_id
        or sandbox_id not in (container.get('Id'), container.get('Name', '').lstrip('/'))
        or container.get('Config', {}).get('Labels', {}).get('nemo-gym.run') != attempt
        or lifecycle.get('handle', {}).get('ownership_labels', {}).get('nemo-gym.run') != attempt):
        raise ValueError('Resource control container/lifecycle attempt differs')
    host = container.get('HostConfig', {})
    creator_digest = row['published_image'].split('@')[-1]
    if (container.get('Image') != creator_digest
        or host.get('Memory') != control['memory_mb'] * 1024**2
        or host.get('MemorySwap') != host.get('Memory')
        or host.get('NanoCpus') != control['cpus'] * 10**9):
        raise ValueError('Actual control image/resources differ')
    env = dict(value.split('=', 1) for value in container.get('Config', {}).get('Env', []) if '=' in value)
    if any(env.get(k) != v for k, v in row['runtime_environment_variant'].items()):
        raise ValueError('Actual control environment differs')
    memory = evidence['memory']
    if (type(memory.get('peak_bytes')) is not int or not 0 < memory['peak_bytes'] <= host['Memory']
        or type(memory.get('oom')) is not int or memory['oom'] != 0
        or type(memory.get('oom_kill')) is not int or memory['oom_kill'] != 0):
        raise ValueError('Resource control still exceeded memory or lacks telemetry')
    return review
