# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Creator-lock adapter: no Gym networking, sandbox or provider imports."""

import base64

from benchmarks.bioagentbench.runtime_channel import RuntimeClient


def runtime_class(*, socket, token, deadline, expected_config):
    from verifiers.v1.runtimes.base import ProgramResult, Runtime
    from verifiers.v1.runtimes.docker import DockerRuntimeInfo

    client = RuntimeClient(socket, token, deadline=deadline)

    class RemoteRuntime(Runtime):
        is_local = True  # Creator interception remains its native loopback endpoint.

        def __init__(self, config, name=None):
            super().__init__(name)
            if config.model_dump() != expected_config:
                raise ValueError("Creator runtime differs from parent-admitted configuration")
            self.config = config
            self.info = DockerRuntimeInfo(**config.model_dump())

        async def start(self):
            self.info.id = (await client.call("start"))["id"]

        async def teardown(self):
            await client.call("teardown")

        async def run(self, argv, env):
            return ProgramResult(**await client.call("run", argv=argv, env=env))

        async def run_background(self, argv, env, log):
            await client.call("run_background", argv=argv, env=env, log=log)

        async def read(self, path):
            return base64.b64decode(await client.call("read", path=path), validate=True)

        async def write(self, path, data):
            await client.call("write", path=path, data=base64.b64encode(data).decode())

    return RemoteRuntime
