# Native service ownership

Native CLI runs retain source/config/host-bound ownership receipts under their output directory. The default process mode remains unchanged and fails closed when it cannot establish ownership of a potentially related process.

On a qualified Linux host with cgroup v2 and a working systemd user manager, explicitly select `NEMO_GYM_SERVICE_OWNERSHIP_MODE=systemd` before starting a new native CLI run. Each service starts in a fresh `nemogym-<random-token>.scope`; its exact unit, InvocationID and ControlGroup are recorded before the service command executes. Environment, working directory, standard input/output and command exit status are preserved without writing environment values into receipts or systemd arguments. Normal shutdown and later ownership reconciliation stop only those dedicated scopes. They never stop the SSH session scope or fall back to a global environment scan.

The mode is part of receipt compatibility: existing process-mode runs cannot be silently migrated. Missing user-manager access, changed unit identity, a populated cgroup after stop, or a surviving recorded root leaves cleanup unverified. This is containment of ordinary descendants, including children that create new POSIX sessions; it is not protection against deliberately moving processes out of the cgroup. It does not restore scientific agent state or resolve uncertain provider requests.

Run the explicit Linux control using an existing qualified interpreter and source checkout:

```sh
NEMO_GYM_TEST_SYSTEMD=1 PYTHONPATH="$PWD" NEMO_GYM_EXTRA_ROOTS="$PWD" \
  python -m pytest -q tests/unit_tests/test_systemd_service_ownership_linux.py
```

This creates only short-lived local fixture scopes. It checks command environment/cwd/stdin/exit failure, coordinator SIGKILL, a detached child, independent-process reconciliation, changed-invocation refusal, and an unrelated surviving sentinel. It makes no model calls. Systemd 249.11 qualification passed both controls; native model/agent/scorer CLI integration is a separate acceptance gate.
