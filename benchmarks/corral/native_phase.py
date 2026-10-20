# SPDX-License-Identifier: Apache-2.0
"""One creator task or saved-state scoring phase; no benchmark scheduler.

Caller owns the isolated runtime, guarded agent transport, and DB snapshot.
This module deliberately does not construct an unguarded model client.
"""

import json
from dataclasses import asdict
from pathlib import Path


async def phase(*, task_id: str, execution_id: str, root: Path, agent=None, saved_state: dict | None = None):
    from corral.observability.base import NoOpObserver
    from corral.orchestration.evaluation import evaluate_task
    from corral.orchestration.models import EvaluateTaskInput, RetryPolicy, RunTaskInput
    from corral.orchestration.registry import RuntimeRegistry
    from corral.persistence import ShardedCommitStore
    from corral.run import execute_task
    from corral.runtime.environment_loader import load_environment_group

    manifest = json.loads(Path(__file__).with_name("frozen_sources.json").read_text())
    if task_id not in manifest["frozen5"] + manifest["preserved_expansion"]:
        raise ValueError("Task is outside preserved Corral selection")
    environments = load_environment_group("retrosynthesis", env_kwargs={"level": 3})
    if environments[task_id].current_task.dependencies():
        raise ValueError("Dependent task requires explicit native dependency-state binding")
    if saved_state is None and agent is None:
        raise ValueError("Solver phase requires an explicitly guarded creator agent")
    if saved_state is not None and (saved_state["execution_id"] != execution_id or agent is not None):
        raise ValueError("Regrade requires matching saved execution and no solver agent")
    store = ShardedCommitStore(root / "state")
    registry = RuntimeRegistry(
        agents={"react": agent} if agent is not None else {},
        environments={task_id: environments[task_id]},
        workspace_manager_factory=store.workspace_manager,
    )
    original_error = None
    try:
        if saved_state is None:
            result = await execute_task(
                state_store=store,
                registry=registry,
                retry_policy=RetryPolicy(maximum_attempts=1),
                task=RunTaskInput(
                    execution_id=execution_id,
                    task_id=task_id,
                    environment_id=task_id,
                    agent_id="react",
                    max_iterations=50,
                    model="openai/nvidia/nvidia/nemotron-3.5-super-vl",
                ),
            )
        else:
            result = await evaluate_task(
                EvaluateTaskInput(
                    execution_id=execution_id,
                    environment_id=task_id,
                    task_id=task_id,
                    commit_hash=saved_state["commit_hash"],
                    branch_id=saved_state["branch_id"],
                ),
                state_store=store,
                registry=registry,
                observer=NoOpObserver(),
            )
        return asdict(result)
    except BaseException as error:
        original_error = error
        raise
    finally:
        cleanup_error = None
        try:
            registry.close()
        except BaseException as error:
            cleanup_error = error
        try:
            await store.aclose()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            if original_error is not None:
                original_error.add_note(f"Corral native cleanup failed: {cleanup_error!r}")
            else:
                raise cleanup_error
