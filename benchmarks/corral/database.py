# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned, disconnected clone of the frozen creator PostgreSQL database.

The task receives only a Unix socket and an unprivileged SELECT role. The
original database is never connected to or modified by this component.
"""

import hashlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.lifecycle import _owned_sandbox


IMAGE = "sha256:3a9633b67d61e169ddcffd531dda5fc702ac823abc5089d4b3031d30c5d2c41e"


@asynccontextmanager
async def database_clone(
    *, snapshot: Path, expected_sha256: str, socket_dir: Path, receipt: Path, run_id: str | None = None
):
    snapshot = snapshot.resolve(strict=True)
    with snapshot.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("Corral database snapshot checksum mismatch")
    socket_dir.mkdir(parents=True, exist_ok=False)
    sandbox = AsyncSandbox(
        {"docker": {"create": {"network": "none"}}},
        SandboxSpec(
            image=IMAGE,
            ttl_s=5040,
            resources=SandboxResources(cpu=1, memory_mib=2048),
            metadata={"workload": "gym-corral-database", "run": run_id or uuid.uuid4().hex},
            provider_options={"volumes": [f"{snapshot}:/snapshot.dump:ro", f"{socket_dir.resolve()}:/db-socket"]},
        ),
    )
    async with _owned_sandbox(sandbox, receipt):
        setup = await sandbox.exec(
            "mkdir /tmp/owned-pg && chown postgres /tmp/owned-pg /db-socket && "
            "runuser -u postgres -- /usr/lib/postgresql/17/bin/initdb -D /tmp/owned-pg --auth-local=peer --auth-host=reject && "
            "runuser -u postgres -- /usr/lib/postgresql/17/bin/pg_ctl -D /tmp/owned-pg "
            "-o \"-c listen_addresses='' -c unix_socket_directories='/db-socket'\" -l /tmp/postgres.log start",
            timeout_s=120,
            preserve_background_services=True,
        )
        if setup.return_code != 0:
            raise RuntimeError(f"Corral clone initialization failed: {setup.stderr}")
        restore = await sandbox.exec(
            "runuser -u postgres -- createdb -h /db-socket reactions_production_db && "
            "runuser -u postgres -- pg_restore -h /db-socket -d reactions_production_db --no-owner --exit-on-error /snapshot.dump",
            timeout_s=1200,
            preserve_background_services=True,
        )
        if restore.return_code != 0:
            raise RuntimeError(f"Corral clone restore failed: {restore.stderr}")
        sql = """DO $$ BEGIN
IF EXISTS (SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
WHERE p.prosecdef AND n.nspname NOT IN ('pg_catalog','information_schema')) THEN
RAISE EXCEPTION 'Unexpected creator SECURITY DEFINER function'; END IF;
IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname NOT LIKE 'pg_%'
AND nspname NOT IN ('public','information_schema')) THEN
RAISE EXCEPTION 'Unexpected creator custom schema'; END IF;
END $$;
CREATE ROLE corral_task LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION;
REVOKE ALL ON DATABASE reactions_production_db FROM PUBLIC;
GRANT CONNECT ON DATABASE reactions_production_db TO corral_task;
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO corral_task;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO corral_task;
ALTER ROLE corral_task SET default_transaction_read_only = on;
"""
        auth = await sandbox.exec(
            "set -e; runuser -u postgres -- psql -h /db-socket -v ON_ERROR_STOP=1 -d reactions_production_db <<'SQL'\n"
            + sql
            + "SQL\n"
            + "printf 'local all postgres reject\\nlocal reactions_production_db corral_task trust\\nlocal all all reject\\n' > /tmp/owned-pg/pg_hba.conf\n"
            + "runuser -u postgres -- /usr/lib/postgresql/17/bin/pg_ctl -D /tmp/owned-pg reload",
            timeout_s=30,
            preserve_background_services=True,
        )
        if auth.return_code != 0:
            raise RuntimeError(f"Corral restricted role setup failed: {auth.stderr}")
        # Reload acknowledgement is asynchronous; never expose the socket while
        # a peer-authenticated administrator connection still succeeds.
        boundary = await sandbox.exec(
            "sleep 1; if runuser -u postgres -- psql -h /db-socket -U postgres -d reactions_production_db -c 'SELECT 1'; then exit 91; fi; "
            "psql -h /db-socket -U corral_task -d reactions_production_db -v ON_ERROR_STOP=1 -Atc 'SELECT count(*) FROM reactions'",
            timeout_s=15,
            preserve_background_services=True,
        )
        if boundary.return_code != 0 or (boundary.stdout or "").strip() != "427683":
            raise RuntimeError("Corral restricted socket boundary or snapshot cardinality check failed")
        yield sandbox
