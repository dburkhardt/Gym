# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fault fixture: a tracked tool session's state disappears before export."""

from resources_servers.example_session_state_mgmt.app import StatefulCounterResourcesServer


class MissingStateCounter(StatefulCounterResourcesServer):
    async def export_session_states(self, session_ids):
        self.session_id_to_counter.clear()
        return await super().export_session_states(session_ids)


if __name__ == "__main__":
    MissingStateCounter.run_webserver()
