# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate native scientific runtime for host-only grading, one HTTP send per call."""
import asyncio,json,sys,urllib.request,urllib.error
from pathlib import Path
from benchmarks.scivisagentbench.evaluate import evaluate
source,task_path,output,endpoint= sys.argv[1:]
task=json.loads(Path(task_path).read_text())
def sender(body):
 request=urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer placeholder','X-Campaign-Benchmark':'SciVisAgentBench'})
 try:
  with urllib.request.urlopen(request,timeout=1200) as response:return json.load(response)
 except urllib.error.HTTPError as exc:
  raw=exc.read().decode('utf-8',errors='replace')
  return {'http_status':exc.code,'raw_error_body':raw,'error':'native judge HTTP failure; no retry'}
asyncio.run(evaluate(Path(source),task['source_case'],Path(task['judge_case_dir']),Path(task['judge_data_dir']),task['agent_mode'],sender,Path(output)))
