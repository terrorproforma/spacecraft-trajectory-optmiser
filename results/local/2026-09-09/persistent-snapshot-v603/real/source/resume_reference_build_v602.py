"""Retain the first compile failure and add its missing pinned QDLDL include."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

root = Path(__file__).resolve().parent / 'persistent-replay-20260909'
prepared = json.loads((root/'preparation.json').read_text())
include = Path('/home/angus/build-qoco-scaled-pool-v540/source/lib/qdldl/include')
command = prepared['qoco_build']['command']
command.insert(5, '-I'+str(include))
assert not (root/'qoco_snapshot_replay').exists()
environment = dict(os.environ, LD_LIBRARY_PATH='/home/angus/build-qoco-scaled-pool-v540/final:/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
shutil.copyfile(__file__, root/'source'/Path(__file__).name)
began = time.perf_counter()
with (root/'qoco-build-retry.log').open('x') as log:
    result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=90)
record = dict(complete=result.returncode==0, gpu_calls=0, command=command,
              returncode=result.returncode, seconds=time.perf_counter()-began,
              added_headers={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in include.glob('*.h')})
if record['complete']:
    binary = root/'qoco_snapshot_replay'
    record['binary'] = dict(path=str(binary), bytes=binary.stat().st_size,
                            sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
(root/'qoco-build-retry.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
raise SystemExit(result.returncode)
