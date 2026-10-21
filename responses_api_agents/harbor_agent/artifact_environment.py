# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in regular-file artifact export for Linux Docker environments."""

import asyncio
import hashlib
import json
import os
import shlex
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from uuid import uuid4

from harbor.environments.docker.docker import DockerEnvironment

from nemo_gym import state_bundle


class BundledArtifactDockerEnvironment(DockerEnvironment):
    """Use native import_path/kwargs; only explicitly selected main-service trees change."""

    def __init__(self, *args, bundle_artifact_sources: list[str], bundle_python: str = "python", **kwargs):
        if not bundle_artifact_sources or any(
            not isinstance(p, str)
            or not p.startswith("/")
            or ".." in PurePosixPath(p).parts
            or str(PurePosixPath(p)) != p
            or p == "/"
            for p in bundle_artifact_sources
        ):
            raise ValueError("Bundle sources must be explicit normalized absolute non-root paths")
        self._bundle_sources = frozenset(bundle_artifact_sources)
        self._bundle_python = bundle_python
        super().__init__(*args, **kwargs)
        if self._is_windows_container:
            raise ValueError("Bundle artifact export supports Linux Docker only")

    async def download_dir(self, source_dir, target_dir):
        if source_dir in self._bundle_sources:
            return await self._download_bundle(source_dir, Path(target_dir))
        return await super().download_dir(source_dir, target_dir)

    async def service_download_dir(self, source_dir, target_dir, *, service=None):
        if source_dir in self._bundle_sources:
            if service not in (None, "main"):
                raise ValueError("Bundle export supports the main service only")
            return await self._download_bundle(source_dir, Path(target_dir))
        return await super().service_download_dir(source_dir, target_dir, service=service)

    async def service_download_dir_with_exclusions(self, source_dir, target_dir, exclude, *, service=None):
        if source_dir in self._bundle_sources:
            raise ValueError("Required bundle export does not support exclusions")
        return await super().service_download_dir_with_exclusions(
            source_dir=source_dir, target_dir=target_dir, exclude=exclude, service=service
        )

    async def _download_bundle(self, source: str, target: Path):
        # Never erase or merge a previous partial capture. Native Harbor creates
        # an empty destination before calling this method.
        if any(p.is_symlink() for p in (target, *target.parents)) or (
            target.exists() and (not target.is_dir() or any(target.iterdir()))
        ):
            raise FileExistsError(f"Artifact target is not an empty real directory: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        remote = "/tmp/gym-artifact-" + uuid4().hex
        module = Path(state_bundle.__file__).resolve()
        digest = hashlib.sha256(module.read_bytes()).hexdigest()
        code = (
            "import hashlib,json,runpy;from pathlib import Path;"
            f"p=Path({remote + '.py'!r});"
            f"assert hashlib.sha256(p.read_bytes()).hexdigest()=={digest!r};"
            "m=runpy.run_path(str(p));"
            f"receipt=m['pack_bundle'](Path({source!r}),Path({remote + '.tar'!r}));"
            f"Path({remote + '.json'!r}).write_text(json.dumps(receipt))"
        )
        primary = None
        try:
            await self.upload_file(module, remote + ".py")
            result = await self.exec(shlex.quote(self._bundle_python) + " -c " + shlex.quote(code), user="root")
            if result.return_code:
                raise RuntimeError(f"Artifact bundle export failed: {result.stderr}")
            with TemporaryDirectory(prefix=".artifact-bundle-", dir=target.parent) as directory:
                stage = Path(directory)
                await self.download_file(remote + ".tar", stage / "bundle.tar")
                await self.download_file(remote + ".json", stage / "manifest.json")
                manifest = json.loads((stage / "manifest.json").read_text())
                await asyncio.to_thread(state_bundle.unpack_bundle, stage / "bundle.tar", manifest, stage / "tree")
                receipt = {"source": source, "state_bundle_sha256": digest, "manifest": manifest}
                # Retain the exact verified manifest independently of Harbor's
                # existing artifact manifest, whose semantics remain unchanged.
                receipt_path = target.with_name(target.name + ".bundle-receipt.json")
                with receipt_path.open("x") as stream:
                    json.dump(receipt, stream)
                if target.exists():
                    target.rmdir()  # refuses a concurrently populated destination
                os.rename(stage / "tree", target)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                cleanup = await self.exec(
                    "rm -f -- " + " ".join(shlex.quote(remote + suffix) for suffix in (".py", ".tar", ".json")),
                    user="root",
                )
                if cleanup.return_code:
                    raise RuntimeError("Artifact bundle temporary-file cleanup failed")
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"Artifact bundle cleanup also failed: {type(error).__name__}")
