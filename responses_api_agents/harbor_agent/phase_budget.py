# SPDX-License-Identifier: Apache-2.0
"""Declared Harbor phase budgets; operational I/O is explicitly not time-bounded."""

import hashlib
import inspect
import json
import math
from pathlib import Path


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source(obj):
    path = Path(inspect.getfile(obj)).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def resolve_phase_budget(config, task):
    """Resolve the installed Harbor's exact timeout properties without starting it.

    Only normal single-step trials are supported. Healthchecks, skills/files,
    artifact transfers, hooks, and environment stop have no collective upstream
    timeout, so their allowance must be supplied separately, never inferred here.
    """
    from harbor.agents.factory import AgentFactory
    from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode
    from harbor.trial.single_step import SingleStepTrial
    from harbor.trial.trial import Trial

    if task.has_steps or config.user_agent is not None or config.source_trial is not None or config.install_only:
        raise ValueError("Phase admission supports only normal single-step trials without user agents or regrade")
    if config.agent.load_trajectory or config.agent.resume_trajectory:
        raise ValueError("Phase admission does not support trajectory-loading setup")
    # Invoke the pinned implementation, not a second copy of override/max and
    # multiplier arithmetic. No Trial.__init__, directories, agent, or container.
    trial = object.__new__(SingleStepTrial)
    trial.config = config
    trial.task = task
    trial._init_timeouts()
    phases = {
        "agent_environment_start": trial.environment_build_timeout_sec,
        "agent_setup": trial.agent_setup_timeout_sec,
        "agent_execution": trial.agent_timeout_sec,
    }
    # SingleStepTrial collects before verification (even when verification is
    # disabled). Main and sidecar hooks run sequentially, using int(timeout_sec).
    # This is one normal collection pass; interrupted recovery can repeat it.
    hooks = task.config.verifier.collect
    if hooks:
        limits = [hook.timeout_sec for hook in hooks]
        if any(not math.isfinite(value) or int(value) <= 0 for value in limits):
            raise ValueError("Collect hooks require finite positive effective timeouts")
        phases["artifact_collect_hooks"] = sum(int(value) for value in limits)
    mode = resolve_task_verifier_mode(task.config)
    if not config.verifier.disable:
        if mode == VerifierEnvironmentMode.SEPARATE:
            phases["verifier_environment_start"] = trial.environment_build_timeout_sec
        phases["verifier_execution"] = trial.verifier_timeout_sec
    if any(value is None or not math.isfinite(value) or value <= 0 for value in phases.values()):
        raise ValueError("Phase admission requires finite positive timeouts for all enabled phases")
    agent_cls = AgentFactory.get_agent_class_from_config(config.agent)
    return {
        "version": 1,
        "phases_seconds": phases,
        "declared_phase_sum_seconds": sum(phases.values()),
        "verifier_mode": str(mode.value),
        "task_config_sha256": hashlib.sha256(task.paths.config_path.read_bytes()).hexdigest(),
        "trial_config_sha256": _hash(config.model_dump(mode="json", context={"redact_sensitive_env": False})),
        "sources": {
            "trial": _source(Trial),
            "single_step": _source(SingleStepTrial),
            "verifier_mode": _source(resolve_task_verifier_mode),
            "agent_mro": [_source(cls) for cls in agent_cls.__mro__ if cls.__module__ != "builtins"],
        },
        "operational_allowance_covers": [
            "healthcheck",
            "skills/files",
            "artifact transfers",
            "hooks",
            "repeated artifact collection during recovery",
            "environment stop",
            "result publication",
        ],
        "end_to_end_runtime_bound_proven": False,
    }


def validate_admission_budget(report, *, drain_seconds, operational_allowance_seconds):
    for value in (drain_seconds, operational_allowance_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Admission drain and operational allowance must be explicit finite positive seconds")
    required = report["declared_phase_sum_seconds"] + operational_allowance_seconds
    if drain_seconds < required:
        raise ValueError(
            f"Harbor admission drain {drain_seconds}s is below declared phases plus allowance {required}s"
        )
    return {
        **report,
        "operational_allowance_seconds": operational_allowance_seconds,
        "minimum_declared_drain_seconds": required,
        "supplied_drain_seconds": drain_seconds,
    }


def validate_job_admission(job, *, drain_seconds, operational_allowance_seconds):
    """Validate exact downloaded tasks before Job.run can start containers/calls."""
    from harbor.models.task.task import Task

    if len(job._trial_configs) != 1:
        raise ValueError("Phase admission supports exactly one resolved trial per job")
    reports = []
    for config in job._trial_configs:
        download = job._task_download_results[config.task.get_task_id()]
        task = Task(download.path, disable_verification=config.verifier.disable)
        reports.append(resolve_phase_budget(config, task))
    if not reports:
        raise ValueError("Phase admission requires at least one resolved trial")
    path = job.config.jobs_dir / job.config.job_name / "gym-phase-budget.json"
    path.write_text(
        json.dumps(
            {
                "trials": reports,
                "drain_seconds": drain_seconds,
                "operational_allowance_seconds": operational_allowance_seconds,
            },
            indent=2,
        )
    )
    return [
        validate_admission_budget(
            report, drain_seconds=drain_seconds, operational_allowance_seconds=operational_allowance_seconds
        )
        for report in reports
    ]
