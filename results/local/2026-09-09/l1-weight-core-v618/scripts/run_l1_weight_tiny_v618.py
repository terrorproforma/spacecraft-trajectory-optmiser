"""Prepared finite unit/weighted correctness batch; launch remains root-controlled."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
parser=argparse.ArgumentParser()
parser.add_argument('--build-root',type=Path,required=True)
parser.add_argument('--manifest-sha256',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def strict_json(s):
    def bad(value):raise ValueError('nonfinite JSON token '+value)
    return json.loads(s,parse_constant=bad)
expected_case_modes={
    'unit':[('positive_prox',0),('disabled_original_scaling_refresh',0),('fresh_original_control',0),
        ('negative_prox',0),('positive_zero_prox',0),('negative_zero_prox',0),
        ('original_weak_seed_zero_step',0),('cancel_before_initial',0),('nonfinite_initial',0)],
    'weighted':[('weighted_quarter_prox',1),('weight_switch_unit',1),('fresh_unit_l1_control',0),
        ('weighted_four_prox',1),('weighted_positive_zero',1),('weighted_negative_zero',1),
        ('cancel_global_prox',2),('weighted_weak_seed_zero_step',1),('cancel_global_weak_seed_zero_step',2),
        ('cancel_before_initial',2),('nonfinite_initial',1),
        ('weighted_primal_step_overflow',1),('weighted_dual_step_overflow',1)]}
root=args.build_root.resolve();manifest_path=root/'manifest.json'
assert digest(manifest_path)==args.manifest_sha256,'manifest identity mismatch'
manifest=strict_json(manifest_path.read_text());assert manifest['complete']
assert manifest['source_identity_kind']=='sha256_tree_not_git_commit' and manifest['frozen_commit'] is None
binary=root/'build/cuda-tests/persistent_l1_test';core=root/'build/cuda/libspacepdhcg_cuda.so'
assert digest(binary)==manifest['persistent_l1_test_sha256'] and digest(core)==manifest['library_sha256']
for name,sha in manifest['source_sha256'].items():assert digest(root/'repo'/name)==sha,name
tree=''.join(k+':'+v+'\n' for k,v in manifest['source_sha256'].items())
assert hashlib.sha256(tree.encode()).hexdigest()==manifest['source_tree_sha256']
output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,output/Path(__file__).name)
report={'complete':False,'gpu_executions_started':0,'maximum_executions':2,'maximum_solve_api_calls':22,
    'expected_optimization_iterations':22,'maximum_requested_iterations':31,
    'manifest_sha256':args.manifest_sha256,'source_commit':manifest['compiled_source_commit'],
    'source_tree_sha256':manifest['source_tree_sha256'],'source_commit_scope':'uncommitted_frozen_source_tree',
    'base_commit':manifest['base_commit'],'workspace_parent_commit':manifest['workspace_parent_commit'],
    'core_sha256':digest(core),'test_sha256':digest(binary),'runner_sha256':digest(Path(__file__)),
    'lock_path':'/home/angus/.spacepdhcg-gpu.lock','lock_policy':'LOCK_EX|LOCK_NB','cases':[]}
def save():(output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
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
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        (output/'gpu.txt').write_text(gpu.stdout+gpu.stderr);assert gpu.returncode==0
        for name,extra,prefix,calls,updates,cap in [('unit',[],'L1_SUMMARY',9,10,13),
                ('weighted',['--weight-tests'],'L1_WEIGHT_SUMMARY',13,12,18)]:
            command=[str(binary)]+extra;entry={'name':name,'command':command,'started':True};report['cases'].append(entry)
            report['gpu_executions_started']+=1;save();begin=time.perf_counter();path=output/(name+'.log')
            with path.open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            entry.update(returncode=result.returncode,wall_seconds=time.perf_counter()-begin,log_sha256=digest(path));save()
            records=[]
            for line in path.read_text().splitlines():
                if line.startswith(('L1_TEST ','L1_SUMMARY ','L1_VALIDATION ','L1_WEIGHT_')):
                    p,v=line.split(' ',1);records.append({'prefix':p,'record':strict_json(v)})
            entry['records']=records;save()
            assert result.returncode==0,path.read_text()[-5000:]
            summaries=[r['record'] for r in records if r['prefix']==prefix];assert len(summaries)==1
            summary=summaries[0]
            assert summary=={'complete':True,'solve_calls':calls,'maximum_requested_iterations':cap,'expected_optimization_iterations':updates}
            cases=[r['record'] for r in records if r['prefix']=='L1_TEST'];assert len(cases)==calls
            assert [(x['case'],x['weight_mode']) for x in cases]==expected_case_modes[name]
            assert sum(x['iterations'] for x in cases)==updates
            entry['strict_json_parsed']=True
        report['actual_solve_api_calls']=22;report['actual_optimization_iterations']=22
        report['complete']=True;report['status']='passed'
    except BaseException as exc:
        report['status']='failed';report['error']=repr(exc);raise
    finally:
        report['wall_seconds']=time.perf_counter()-start;report['lock_scope_ending']=True;save()
print(json.dumps({'status':report['status'],'report':str(output/'report.json'),'sha256':digest(output/'report.json')}),flush=True)
