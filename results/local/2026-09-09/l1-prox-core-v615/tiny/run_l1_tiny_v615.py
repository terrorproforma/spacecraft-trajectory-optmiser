"""One finite reviewed L1 test process; actual GPU launch remains root-controlled."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--build-root',type=Path,required=True)
parser.add_argument('--manifest-sha256',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=args.build_root.resolve();manifest_path=root/'manifest.json'
assert digest(manifest_path)==args.manifest_sha256,'manifest identity mismatch'
manifest=json.loads(manifest_path.read_text());assert manifest['complete']
binary=root/'build/cuda-tests/persistent_l1_test';core=root/'build/cuda/libspacepdhcg_cuda.so'
assert digest(binary)==manifest['persistent_l1_test_sha256']
assert digest(core)==manifest['library_sha256']
for name in manifest['owned_paths']:assert digest(root/'repo'/name)==manifest['source_sha256'][name]
output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,output/Path(__file__).name)
report={'complete':False,'gpu_calls_started':0,'maximum_executions':1,'maximum_solve_api_calls':9,
        'expected_optimization_iterations':10,'maximum_requested_iterations':13,
        'manifest_sha256':args.manifest_sha256,'source_commit':manifest['frozen_commit'],
        'core_sha256':digest(core),'test_sha256':digest(binary),'runner_sha256':digest(Path(__file__)),
        'lock_path':'/home/angus/.spacepdhcg-gpu.lock','lock_policy':'LOCK_EX|LOCK_NB','cases':[]}
def save():(output/'report.json').write_text(json.dumps(report,indent=2))
save();start=time.perf_counter()
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_','PDHCG_','QOCO_','LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES']='0'
with open(report['lock_path'],'a+') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        report['status']='lock_busy_zero_GPU_calls';save();raise SystemExit(75)
    report['lock_acquired']=True;report['pid']=os.getpid();save()
    print(json.dumps({'status':'running','pid':os.getpid(),'output':str(output)}),flush=True)
    try:
        inventory=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        (output/'compute-processes-before.txt').write_text(inventory.stdout+inventory.stderr)
        assert inventory.returncode==0 and not inventory.stdout.strip(),'another compute process is active'
        command=[str(binary)];entry={'command':command,'started':True};report['cases'].append(entry)
        report['gpu_calls_started']=1;save();begin=time.perf_counter();path=output/'tiny.log'
        with path.open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        entry.update(returncode=result.returncode,wall_seconds=time.perf_counter()-begin,log_sha256=digest(path));save()
        assert result.returncode==0,path.read_text()[-5000:]
        records=[(line.split(' ',1)[0],json.loads(line.split(' ',1)[1])) for line in path.read_text().splitlines()]
        entry['records']=[{'prefix':prefix,'record':record} for prefix,record in records]
        summaries=[v for p,v in records if p=='L1_SUMMARY'];assert len(summaries)==1
        summary=summaries[0];assert summary['complete'] and summary['solve_calls']==9 and summary['expected_optimization_iterations']==10
        cases=[v for p,v in records if p=='L1_TEST'];assert len(cases)==9
        assert sum(x['iterations'] for x in cases)==10
        entry['strict_json_parsed']=True
        report['actual_solve_api_calls']=9;report['actual_optimization_iterations']=10
        report['complete']=True;report['status']='passed'
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:
        report['wall_seconds']=time.perf_counter()-start;report['lock_scope_ending']=True;save()
print(json.dumps({'status':report['status'],'report':str(output/'report.json'),'sha256':digest(output/'report.json')}),flush=True)
