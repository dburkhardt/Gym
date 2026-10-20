# SPDX-License-Identifier: Apache-2.0
"""Commands for bounded image qualification; never dispatch an LLM or install at run time."""
import hashlib
import json

# This command runs identically in creator and prepared images. Scientific
# interpreters keep their original PATH, executable and installed distributions.
SCIENTIFIC_PROBE = r'''set -eu
for command in python python2 python3 R Rscript; do
  if command -v "$command" >/dev/null 2>&1; then
    executable=$(command -v "$command")
    printf '%s\n' "$command:$executable"
    sha256sum "$executable"
    case "$command" in
      python*) "$executable" -c 'import sys; print(sys.version); print(sys.executable); print(sys.path)'; "$executable" -m pip freeze 2>/dev/null || true ;;
      *) "$executable" --version 2>&1 ;;
    esac
  fi
done
'''

HARNESS_PROBE = r'''set -eu
/opt/corebench-harness/bin/python3 -c 'import numpy, scipy, openai, backoff, tqdm; print("harness-python-ok")'
tmux -V
asciinema --version
test ! -e /tests/answers.json
test ! -e /solution
test ! -e /workspace/answers.json
echo reference-isolation-ok
'''


def dockerfile(row):
    """Source-preserving overlay; image-specific controls remain mandatory.

    The provided python directory is the separately hash-verified recovered
    runtime. No scientific Python executable or global PATH is replaced.
    """
    env = '\n'.join('ENV '+k+'='+json.dumps(v) for k,v in row['runtime_environment_variant'].items())
    return f'''FROM {row["published_image"]}
COPY python /opt/corebench-harness
RUN /opt/corebench-harness/bin/python3 -m pip install numpy==1.26.4 scipy==1.14.1 openai==1.109.1 backoff==2.2.1 tqdm==4.67.1 asciinema==2.4.0
RUN command -v tmux >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-upgrade --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*)
RUN if ! command -v asciinema >/dev/null 2>&1; then printf '#!/bin/sh\\nexec /opt/corebench-harness/bin/python3 -m asciinema "$@"\\n' > /usr/local/bin/asciinema && chmod 755 /usr/local/bin/asciinema; fi
COPY capsule /workspace/{row["id"]}
RUN mkdir -p /workspace/{row["id"]}/results
RUN for name in code data results; do if [ -e "/$name" ] || [ -L "/$name" ]; then rmdir "/$name"; fi; ln -s "/workspace/{row["id"]}/$name" "/$name"; done
{env}
WORKDIR /workspace
'''


def recipe_digest(row):
    return hashlib.sha256(dockerfile(row).encode()).hexdigest()

def source_digest(row):
    return hashlib.sha256(json.dumps(row["medium_file_manifest"], sort_keys=True).encode()).hexdigest()


def stage_context(row, medium_root, reference_root, runtime_archive, destination):
    """Build context from preserved, checked sources; no Docker/network action."""
    import os
    import tarfile
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from benchmarks.corebench.prepare import RUNTIME_SHA256
    from benchmarks.corebench.prepare_frozen20 import verify_sources, digest
    from nemo_gym.dataset_staging import stage_file

    source, _ = verify_sources(row, medium_root, reference_root)
    if digest(runtime_archive) != RUNTIME_SHA256:
        raise ValueError('Recovered harness runtime hash differs')
    if destination.exists():
        raise ValueError('Refusing existing image context')
    with TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        with tarfile.open(runtime_archive) as archive:
            for member in archive.getmembers():
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or path.parts[0] != 'python' or not (member.isfile() or member.isdir() or member.issym()):
                    raise ValueError('Unsafe runtime member')
            archive.extractall(temporary, filter='data')
        for f in row['medium_file_manifest']:
            src=source/f['path']; dst=destination/'capsule'/f['path']
            stage_file(src,dst,expected_sha256=f['sha256']);dst.chmod(src.stat().st_mode & 0o777)
        for src in (temporary/'python').rglob('*'):
            dst=destination/src.relative_to(temporary)
            if src.is_symlink():
                dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(os.readlink(src))
            elif src.is_file():
                stage_file(src,dst,expected_sha256=digest(src));dst.chmod(src.stat().st_mode & 0o777)
        recipe=temporary/'Dockerfile';recipe.write_text(dockerfile(row));stage_file(recipe,destination/'Dockerfile')
    return destination
