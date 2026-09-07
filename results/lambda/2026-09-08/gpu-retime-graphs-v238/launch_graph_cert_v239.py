from pathlib import Path
import os, subprocess, ast, json
root=Path('/home/ubuntu/spacepdhcg-graph-cert-v239');root.mkdir(exist_ok=False)
script=r'''from pathlib import Path
import os,sys,json,time,fcntl,hashlib,traceback
root=Path('/home/ubuntu/spacepdhcg-graph-cert-v239');repo=Path('/home/ubuntu/spacepdhcg-retime-graph-v238/repo')
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(repo/'src'))
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.search import RoutePlan,SearchSettings
from spacepdhcg.gtoc12.retiming import Retimer
from spacepdhcg.gtoc12.returnsweep import ReturnSweep
import numpy as np
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.pipeline import refine_route,write_route_artifacts
from spacepdhcg.gtoc12.low_thrust import ScvxSettings
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
from spacepdhcg.gtoc12.official import run_official_verifier
lock=open('/home/ubuntu/.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
report=dict(pid=os.getpid(),start=time.time(),complete=False,rows=[],source_sha256={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in ['src/spacepdhcg/gtoc12/retiming.py','cpp/cuda/src/native_qoco_adapter.cpp']})
try:
 cat=load_catalogue();original=RoutePlan.from_summary(json.loads(Path('/home/ubuntu/spacepdhcg-native-campaign-v209/output/ship_01/refinements.json').read_text())[0]['plan'])
 with using_lambert_backend('cuda') as gpu:
  flown=json.loads((repo/'input-return.json').read_text())['legs'][-1];assert flown['certified']
  timer=Retimer(cat,SearchSettings());timer.set_return_sweep(ReturnSweep(flown['from'],flown['mass_before'],np.array([flown['t0']]),np.array([flown['tf']-flown['t0']]),np.ones((1,1),dtype=bool),np.ones((1,1),dtype=bool),np.array([[flown['delta_v_km_s']]]),np.array([[flown['propellant_kg']]])))
  result=timer.retime(original)
 report['retime_gpu']=dict(gpu.telemetry);report['retiming']=result.summary();assert result.improved
 (root/'plan.json').write_text(json.dumps(result.plan.summary(),indent=2))
 settings=ScvxSettings(node_days=2.,max_iterations=40,seed_backend='cuda',discretisation_backend='cuda',assembly_backend='cuda',convex_solver_backend='qoco',outer_loop_backend='cuda')
 route=refine_route(result.plan,cat,scvx=settings);report['refinement']=route.summary()
 if route.certified:
  artifacts=write_route_artifacts(route,cat,root/'output');solution=Path(artifacts['solution'])
  report['independent']=Gtoc12Verifier(cat,bonus=load_bonus_table()).verify_file(solution).summary()
  report['official']=run_official_verifier(solution).summary()
 report['complete']=True
except Exception:report['error']=traceback.format_exc()
report['end']=time.time();(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='refinement'}),flush=True)
'''
(root/'run.py').write_text(script)
env={k:v for k,v in os.environ.items() if not k.startswith('SPACEPDHCG_TEST_')}
env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SPACEPDHCG_GTOC12_CUDA_LIBRARY='/home/ubuntu/spacepdhcg-retime-graph-v238/core-build/cuda/libspacepdhcg_cuda.so',SPACEPDHCG_QOCO_LIBRARY='/home/ubuntu/spacepdhcg-diagnose-v174/final/libqoco.so',SPACEPDHCG_GTOC12_DATA='/home/ubuntu/spacepdhcg/gtoc12/benchmarks/gtoc12/data',LD_LIBRARY_PATH='/home/ubuntu/spacepdhcg-retime-graph-v238/core-build/cuda:/home/ubuntu/spacepdhcg-diagnose-v174/final:/usr/local/cuda/lib64')
for node in ast.parse(Path('/home/ubuntu/spacepdhcg-diagnose-v174/diagnose.py').read_text()).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in node.targets):env['LD_LIBRARY_PATH']+=':'+ast.literal_eval(node.value)+'/lib'
for name,cmd in [
 ('release-comparison',['/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/home/ubuntu/spacepdhcg-retime-graph-v238/repo/benchmark_graph_release_v238.py',str(root/'release-comparison'),'/home/ubuntu/spacepdhcg-native-campaign-v209/output/ship_01/refinements.json','/home/ubuntu/spacepdhcg-retime-graph-v238/repo/input-return.json','/home/ubuntu/spacepdhcg-sweeps-v234/repo/src/spacepdhcg/gtoc12/gpu_retime.py','/home/ubuntu/spacepdhcg-sweeps-v234/core-build/cuda/libspacepdhcg_cuda.so',env['SPACEPDHCG_GTOC12_CUDA_LIBRARY']]),
 ('boundary-profile',['/home/ubuntu/spacepdhcg/v1/.venv/bin/python','/home/ubuntu/spacepdhcg-retime-graph-v238/repo/profile_retime_boundary_v239.py',str(root/'boundary-profile.json'),'/home/ubuntu/spacepdhcg-native-campaign-v209/output/ship_01/refinements.json'])
]:
 env['PYTHONPATH']='/home/ubuntu/spacepdhcg-retime-graph-v238/repo/src'
 with (root/(name+'.log')).open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
with (root/'runner.log').open('x') as log:p=subprocess.Popen(['/home/ubuntu/spacepdhcg/v1/.venv/bin/python',str(root/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'pid').write_text(str(p.pid));print(json.dumps(dict(pid=p.pid,root=str(root))))
