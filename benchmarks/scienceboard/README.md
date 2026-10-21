# ScienceBoard image evidence admission

This is a model-free preparation gate, **not a runnable benchmark registration**.
It reads saved Docker metadata and an operator-owned receipt from the exact creator
sanitization audit. No Docker command, download, VM conversion, model, or worker is
started. There was no general image-admission registry to extend; this
candidate-specific check does not introduce one or replace shared sandbox ownership.

Pinned source: `OS-Copilot/ScienceBoard@21b5eb0f1d00898f7de9e84c3d33a87c2f2b3c3e`.
The official `docker/export/README.md` requires the reference guest's `sci_bench`
snapshot, sanitizing overlay and flattened image. The public reference VM is not
itself the qualified Docker image. Do not distribute intermediate/rootfs layers.

For an already authorized, locally available image, save `docker image inspect`
output. Execute the pinned creator `scripts/audit_distribution.py --image <sha256 ID>`
in a resource-admitted execution window, and record its exit code and exact JSON
stdout. The admission command itself never runs this audit. The receipt format is:

```json
{"source_revision":"21b5eb0f1d00898f7de9e84c3d33a87c2f2b3c3e",
 "script_sha256":"40d420cb4f749092080138ea8926bcdbdb7a65cc20f21152b1b5a3b9aa923d80",
 "image_id":"sha256:<64 lowercase hex characters>", "returncode":0,
 "report":{"image":"sha256:<same ID>","passed":true,"finding_count":0,"findings":[]}}
```

`resources.json` must declare `cpus`, `memory_gib`, `free_disk_gib`, `shm_gib`.
This gate adopts the lower end of the creator
[recommended capacity](https://github.com/OS-Copilot/ScienceBoard/blob/21b5eb0f1d00898f7de9e84c3d33a87c2f2b3c3e/docs/cloud-quickstart.md)
as a conservative admission policy: 8 CPUs, 24 GiB memory, 80 GiB free disk
and 2 GiB shared memory. It is not an experimentally established minimum. These declarations are not live capacity observations.

```bash
python -m benchmarks.scienceboard.admission --inspect inspect.json --audit audit.json --resources resources.json
```

Missing/malformed evidence exits 2 with `blocked`; valid evidence still returns
`runnable: false` and `image_evidence_admitted_runtime_unqualified`. Receipts are
operator attestations, not cryptographic proof of producer honesty. The evidence
hash binds metadata, audit and placement inputs. Actual application initialization,
frozen task binding, model transport and creator verifier controls remain required.
The fixtures test the gate and do not claim a real image passed.
