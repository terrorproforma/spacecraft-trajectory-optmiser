from pathlib import Path
import subprocess,os,json,shutil,tarfile,hashlib,ast,fcntl,time
root=Path('/home/ubuntu/spacepdhcg-lagrange-precision-v503');repo=root/'repo';baseline=Path('/home/ubuntu/spacepdhcg-gpu-execution-v328/repo')
report=dict(pid=os.getpid(),complete=False,stages=[],campaigns=[])
def save():(root/'report.json').write_text(json.dumps(report,indent=2))
save()
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');report['stage']='waiting_for_gpu';save()
fcntl.flock(lock,fcntl.LOCK_EX)
try:
 repo=Path('/home/ubuntu/spacepdhcg-resident-options-v500/repo')
 report['source_base']='resident-options-v500/repo'
 cmake='/home/ubuntu/spacepdhcg/v1/.venv/bin/cmake';py='/home/ubuntu/spacepdhcg/v1/.venv/bin/python'
 core=Path('/home/ubuntu/spacepdhcg-resident-options-v497/core-build/cuda/libspacepdhcg_cuda.so');qoco=Path('/home/ubuntu/spacepdhcg-step-final-v359/final/libqoco.so')
 env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
 env['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0'
 env.update(PYTHONPATH=str(repo/'src'),SPACEPDHCG_GTOC12_CUDA_LIBRARY=str(core),SPACEPDHCG_QOCO_LIBRARY=str(qoco),SPACEPDHCG_GTOC12_GPU_TESTS='1',SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):runtime=ast.literal_eval(node.value)
 env['LD_LIBRARY_PATH']=str(core.parent)+':'+str(qoco.parent)+':'+runtime+'/lib:/usr/local/cuda/lib64'
 def run(name,cmd,timeout=900,environment=None,cwd=None):
  start=time.perf_counter()
  with (root/(name+'.log')).open('x') as log:
   child=subprocess.Popen(cmd,cwd=cwd or repo,env=environment or env,stdout=log,stderr=subprocess.STDOUT)
   report['child_pid']=child.pid;report['stage']=name;save()
   try:code=child.wait(timeout=timeout)
   except subprocess.TimeoutExpired:child.kill();child.wait();raise
  report['stages'].append(dict(name=name,returncode=code,seconds=time.perf_counter()-start,command=cmd));save()
  assert code==0,(name,code)
 report['runtime_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [core,qoco]};save()
 (root/'diagnose.py').write_text("from pathlib import Path\nimport dataclasses,json,os,sys,time\nimport numpy as np\nfrom types import SimpleNamespace\nsys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']\nsys.path.insert(0,str(Path('tests').resolve()))\nfrom test_gtoc12_gpu_discretisation import synthetic_boundary\nfrom spacepdhcg.gtoc12.low_thrust import ScvxSettings,solve_leg,certify_leg\nfrom spacepdhcg.gtoc12.gpu_execution import using_gpu_execution\nroot=Path(sys.argv[1]);root.mkdir(exist_ok=False);rows=[]\na=synthetic_boundary();rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])\nb=dataclasses.replace(a,departure_position=rotation@a.departure_position,departure_velocity=rotation@a.departure_velocity,arrival_position=rotation@a.arrival_position,arrival_velocity=rotation@a.arrival_velocity,arrival_epoch=a.arrival_epoch-.5,initial_mass=a.initial_mass-50)\npolicy=SimpleNamespace(gpu_execution='graph',outer_loop_backend='cuda',workers=1)\ndef encode(x):\n if isinstance(x,np.ndarray):return x.tolist()\n if isinstance(x,np.generic):return x.item()\n raise TypeError(type(x).__name__)\ndef save():(root/'solves.json').write_text(json.dumps(rows,indent=2,default=encode))\nwith using_gpu_execution(policy):\n for precision in ['default','tighter']:\n  settings=ScvxSettings(discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',outer_loop_backend='cuda',hold='lagrange',max_iterations=40,time_limit_s=30)\n  if precision=='tighter':settings=dataclasses.replace(settings,step_tolerance=1e-9,objective_tolerance=1e-8,defect_tolerance=5e-10,clarabel_tolerance=1e-10)\n  for origin in [0,1]:\n   os.environ['SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN']=str(origin)\n   os.environ['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH']='0'\n   for pool in [0,1]:\n    os.environ['SPACEPDHCG_TEST_GTOC12_QOCO_POOL']=str(pool)\n    for trial in range(12):\n     for name,boundary in [('A',a),('B',b)]:\n      start=time.perf_counter();solution=solve_leg(boundary,settings);certificate=certify_leg(solution)\n      rows.append(dict(precision=precision,origin=origin,pool=pool,trial=trial,boundary=name,status=solution.status,iterations=solution.iterations,certificate=dataclasses.asdict(certificate),certified=certificate.within_tolerance,mass=certificate.final_mass_kg,seconds=time.perf_counter()-start,history=solution.history,reports=solution.solver_reports));save()\nprint(len(rows),'solves recorded',flush=True)\n")
 for name,library in [('prior',Path('/home/ubuntu/spacepdhcg-workspace-pool-v491/core-build/cuda/libspacepdhcg_cuda.so')),('candidate',core)]:
  case=env.copy();case['SPACEPDHCG_GTOC12_CUDA_LIBRARY']=str(library);case['LD_LIBRARY_PATH']=str(library.parent)+':'+env['LD_LIBRARY_PATH']
  run(name,[py,str(root/'diagnose.py'),str(root/name)],900,environment=case)
 report['complete']=True
except Exception as e:report['error']=repr(e)
save();print('complete',report['complete'],report.get('error'),flush=True)
