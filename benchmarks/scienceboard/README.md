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

## Maintained preparation and desktop lifecycle boundary

`prepare.py` now stages the exact creator `scripts/materialize_smoke_subset.py`
20-task list. This is **a new, explicitly named creator-parity20 selection**:
the campaign previously recorded no frozen ScienceBoard task IDs. The checked-in
manifest binds all creator Python modules, relevant runtime definitions and each
exact task JSON. Instructions become native input rows; initializer and evaluator
JSON stay in a separate trusted host-side directory and must not be mounted into
the solver. Task steps, snapshots and verifiers are unchanged. No synthetic task
or shortened scientific protocol is introduced.

```bash
SCIENCEBOARD_SOURCE_ROOT=/path/to/pinned/ScienceBoard python -m benchmarks.scienceboard.prepare
```

`managed_desktop.creator_desktop_factory` validates the existing image-evidence
packet and source imports, and returns a constructor with the pinned creator
`DockerDesktopEnv` interface. An isolated creator worker can assign it to
`sci.docker.DockerDesktopEnv` before constructing the unchanged creator Tester.
Creation, resource limits and exact-owned cleanup use Gym's shared `Sandbox`;
no independent docker launcher or cleanup registry is added. Reset removes the
owned sandbox and starts fresh state. Health **and** PNG readiness are required.
The controller and scientific verification remain the creator implementations.

This adapter preserves the native entrypoint, Mesa environment, guest hostname,
2GiB shared memory and bridge networking. It requests8 CPUs/24GiB RAM using the
existing admission recommendation; actual minimum performance is unmeasured.
No host network, published port, secret environment or host task/reference mount
is used. Linux-host reachability of its private bridge address is required because
native application clients use fixed guest ports. A proxy-free creator process is
required until its module-level HTTP clients have a qualified scoped transport.

**Not yet a runnable Gym benchmark:** no config registration is supplied until
an actual native guarded multimodal agent transport and full Tester result mapping
are qualified. The lifecycle controls use substituted Sandbox/HTTP backends; actual
image startup, GUI initializers, actions, creator verifiers, source/asset isolation,
and cancellation while creator threads are active remain untested. Custom image
entrypoint also disables the shared Docker provider's sleep-based TTL; successful
`close` is checked in local controls, but abrupt-controller termination requires a
separate owned-resource supervisor/cleanup proof before live admission. The
17GB reference VM route is not equivalent to an admitted sanitized Docker image.

## Guarded creator model transport and result mapping

`guarded_model.py` binds only the pinned `sci.base.model.requests` reference in
an isolated synchronous worker. The unchanged creator `Model._request_openai`
constructs its complete payload; the facade routes it through shared
`GuardedSyncChatTransport` to an authenticated task UDS bridge. Screenshot bytes,
text/order and sampling keys are retained. The creator's480-second model timeout
bounds connection plus full-body receipt. Failed/ambiguous calls poison subsequent
requests; a captured null action is retained but cannot trigger the creator's
internal retry loop. No upstream credential is exposed to the creator model.

This is a transport adapter, not yet the complete native agent server: its caller
must supply the task lineage, source/configuration digest and owned bridge, bind
all roles, and supervise creator/desktop cleanup. Protocol differences are explicit:
transport errors do not use the upstream three-try retry wrapper; native null
content stops after the preserved attempt. No sampling or screenshot changes are
introduced.

`result_mapping.py` preserves exact `result.out`0/1 bytes/hash. Missing output is
null and masked, never zero. A known transport/creator exception keeps any raw
native result but masks its execution disposition. Absent an independent scientific
review, ordinary native results remain unreviewed rather than accepted quality.
This mapper requires the eventual runner to report exceptions instead of relying
only on Tester's aggregate counts (Tester catches exceptions as skipped tasks).

Offline controls now exercise the **actual pinned** creator Model serializer and
parser with a real PIL image, and actual VManager enter/reset/exit using substituted
Docker/HTTP. The local Gym interpreter imports these via a small isolated dependency
overlay; it is Python3.13, whereas creator setup recommends3.11. Full scientific
application/runtime qualification remains necessary before claiming compatibility.

The named protocol is `creator-openai-guarded-single-attempt-no-overflow-trimming-v1`.
Exact first-request payload equality does **not** certify unchanged full creator
retry semantics: native Agent defaults to three tries after Model.access returns
None, and its overflow handler recursively trims context. This adapter raises on
non200 before Agent's overflow handler and latches no-resend on malformed returned
actions. Both retry suppression and overflow-trimming suppression are explicit
variants; future execution identity must include this variant name.

The no-resend latch is separate from `outcome`. A captured null/length response is
`model_output_budget`; null/nonlength is `model_invalid_action`. Both remain in
attempted model-failure denominators and are not masked as infrastructure. An
absent creator score remains null, never an invented zero. Ambiguous timeout is
`transport_unresolved`, unscored and masked. Malformed result artifacts retain
their bytes/hash and map to explicit verifier-artifact failure.
