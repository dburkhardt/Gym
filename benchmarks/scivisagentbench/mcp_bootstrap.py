# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve postponed annotations before pinned MCP1.9.4 inspects native tools.

MCP1.9.4 calls issubclass on future-annotation strings. Resolving the existing
annotations changes neither tools nor argument types; creator source stays intact.
"""
import runpy
import sys
import typing
from pathlib import Path
from mcp.server.fastmcp import FastMCP

_original = FastMCP.add_tool

def add_tool(self, fn, *args, **kwargs):
    fn.__annotations__ = typing.get_type_hints(fn)
    return _original(self, fn, *args, **kwargs)

FastMCP.add_tool = add_tool
source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source.parent))
sys.argv = [str(source), *sys.argv[2:]]
runpy.run_path(str(source), run_name='__main__')
