# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify actual Docker-exec native MCP channels without inference."""
import asyncio,json,sys
from pathlib import Path
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
ROOT='/home/ubuntu/projects/broad-eval-campaign/scivisagentbench/runtime-ram'
CONTAINER='scivis-native-persistent'
servers=[{'command':'docker','args':['exec','-i',CONTAINER,ROOT+'/env/bin/python','/control/mcp_bootstrap.py','/source/src/napari_mcp/napari_mcp_server.py']},
 {'command':'docker','args':['exec','-i',CONTAINER,ROOT+'/node-env/bin/node',ROOT+'/node-tools/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js','/workspace']}]
async def main():
 records=[]
 for i,server in enumerate(servers):
  async with stdio_client(StdioServerParameters(**server)) as (read,write),ClientSession(read,write) as session:
   await session.initialize();tools=await session.list_tools()
   name,args=('list_layers',{}) if i==0 else ('read_text_file',{'path':'/workspace/viewer-ready.json'})
   result=await session.call_tool(name,args);assert not result.isError
   denied=await session.call_tool('open_file' if i==0 else 'read_text_file',{'file_path':'/workspace/GS/dataset_002_1.png'} if i==0 else {'path':'/workspace/GS/dataset_002_1.png'})
   assert denied.isError or 'File not found' in str(denied)
   records.append({'tools':len(tools.tools),'response':result.model_dump(mode='json'),'hidden_reference_denied':True})
 Path(sys.argv[1]).write_text(json.dumps({'llm_calls':0,'records':records,'native_servers':servers,'status':'passed'},indent=2))
asyncio.run(main())
