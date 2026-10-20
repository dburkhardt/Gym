# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child self-identifies durably before parent may release native execution."""
import json,os,sys,time
from pathlib import Path
def persist(path,value):
    # Child uses the pinned creator runtime, which deliberately lacks aiohttp.
    with path.open('x') as stream:
        json.dump(value,stream);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def identity(pid):
 stat=Path(f'/proc/{pid}/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
 return {'pid':pid,'pgid':os.getpgid(pid),'start_ticks':int(fields[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'command_hex':Path(f'/proc/{pid}/cmdline').read_bytes().hex()}

def hold_then_exec(attempt,command,identity_fn=identity,execute=os.execv,timeout=60):
 attempt=Path(attempt);me=identity_fn(os.getpid());assert me['pid']==me['pgid']
 persist(attempt/'child-identity.json',dict(me,status='awaiting-parent-release',native_command=command,time=time.time()))
 deadline=time.time()+timeout
 while not (attempt/'child-release.json').exists():
  if time.time()>deadline:return 75
  time.sleep(.05)
 release=json.loads((attempt/'child-release.json').read_text());assert release['pid']==me['pid'] and release['start_ticks']==me['start_ticks'] and release['boot_id']==me['boot_id']
 execute(command[0],command)
 return 0
if __name__=='__main__':raise SystemExit(hold_then_exec(sys.argv[1],sys.argv[2:]))
