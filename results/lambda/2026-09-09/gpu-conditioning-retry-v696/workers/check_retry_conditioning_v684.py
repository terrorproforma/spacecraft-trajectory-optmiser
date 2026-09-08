from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys

root=Path('/home/angus/spacepdhcg-retry-conditioning-v683')
build=json.loads((root/'report.json').read_text());assert build['complete'] and build['success']
out=root/'validation-v684';out.mkdir(exist_ok=False)
source=root/'repo/cpp/cuda/tests/qoco_snapshot_replay.cu'
text=source.read_text().replace('extern "C" int qoco_gpu_begin_reduction_scope();','extern "C" int qoco_gpu_begin_reduction_scope();\nextern "C" int qoco_gpu_numeric_retry_conditioning(void*,const int*);')
text=text.replace('double *values{},*dorigin{},*doffset{};', 'double *values{},*dorigin{},*doffset{};\n    int* retry{}; CHECK(cudaMalloc(&retry,sizeof(int))); CHECK(cudaMemset(retry,0,sizeof(int)));\n    CHECK(qoco_gpu_numeric_retry_conditioning(update,retry));')
text=text.replace('for(int repeat=0;repeat<repeats;++repeat) {','for(int repeat=0;repeat<repeats;++repeat) {\n        const int mode=repeat%3; CHECK(cudaMemcpy(retry,&mode,sizeof(int),cudaMemcpyHostToDevice));')
text=text.replace('qoco_gpu_destroy_numeric_update(update);qoco_cleanup(solver);','CHECK(qoco_gpu_numeric_retry_conditioning(update,nullptr)); CHECK(cudaFree(retry));\n    qoco_gpu_destroy_numeric_update(update);qoco_cleanup(solver);')
(out/'retry_snapshot_replay.cu').write_text(text)
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))}
env.update(SPACEPDHCG_TEST_QOCO_IPM_GRAPH='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH=str(root/'final')+':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64')
command=['/usr/local/cuda-12.8/bin/nvcc','--default-stream','per-thread','-arch=sm_120','-std=c++17',*['-I'+str(root/'qoco'/n) for n in ['include','algebra/cuda','lib/qdldl/include','lib/amd']],str(out/'retry_snapshot_replay.cu'),'-L'+str(root/'final'),'-lqoco','-o',str(out/'retry_snapshot_replay')]
with (out/'build.log').open('x') as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
r.check_returncode()
qp=Path('results/local/2026-09-09/return-qp-v680/input-qp.txt').resolve()
with Path('/home/angus/.spacepdhcg-gpu.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with (out/'replay.log').open('x') as log:r=subprocess.run([str(out/'retry_snapshot_replay'),str(qp),'12'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
sys.path.insert(0,'/home/angus/spacepdhcg-return-qp-v676/replay-v678')
from audit_helper import problem,audit
d=problem(qp)
records=[json.loads(line[10:]) for line in (out/'replay.log').read_text().splitlines() if line.startswith('QP_REPLAY ')]
checks=[dict(mode=r['repeat']%3,status=r['status'],iterations=r['iterations'],**audit(d,r)) for r in records]
report=dict(complete=True,returncode=r.returncode,checks=checks,build_command=command,libraries=build['libraries'],source_sha256=hashlib.sha256((out/'retry_snapshot_replay.cu').read_bytes()).hexdigest())
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(returncode=r.returncode,modes={mode:dict(repeats=sum(c['mode']==mode for c in checks),qualified=sum(c['mode']==mode and c['qualified'] for c in checks)) for mode in range(3)})))
