"""Two predeclared penalty probes of the saved failed boundary, unchanged physics gates."""
from pathlib import Path
import argparse,dataclasses,fcntl,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parent;HOME=Path.home();PRIOR=HOME/'spacepdhcg-family-refinement-v851'
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
sys.path.insert(0,str(HOME/'spacepdhcg-return-cache-v844/repo/src'))
import numpy as np
from spacepdhcg.gtoc12.low_thrust import LegBoundary,ScvxSettings,solve_leg
from spacepdhcg.gtoc12.pipeline import clamp_thrust
from spacepdhcg.gtoc12.gpu_verifier import certify_legs_cuda
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution

def write(path,value):
    value=json.loads(json.dumps(value,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v)),parse_constant=lambda x:x)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))
report=dict(pid=os.getpid(),complete=False,success=False,probes=[],native_solves=0,
            protocol='Single original failed ship-18 leg; virtual_weight 100000 and 1000000, cold native initialization, original boundary/mass, no repeated baseline. All original dynamics/KKT/certification gates retained. No fleet promotion by this experiment.')
def save():write(ROOT/'report.tmp',report);(ROOT/'report.tmp').replace(ROOT/'report.json')
save();started=time.perf_counter()
try:
    old=json.loads((PRIOR/'report.json').read_text());assert old['complete'] and old['success'] and old['native_solves']==3
    b=json.loads((PRIOR/'solve-002/boundary.json').read_text())
    for k in ('departure_position','departure_velocity','arrival_position','arrival_velocity'):b[k]=np.asarray(b[k])
    boundary=LegBoundary(**b)
    settings=ScvxSettings(**old['settings'])
    assert settings.virtual_weight==10000. and settings.defect_tolerance==5e-9
    report.update(boundary_sha256=hashlib.sha256((PRIOR/'solve-002/boundary.json').read_bytes()).hexdigest(),
        baseline_defect=json.loads((PRIOR/'solve-002/solution.json').read_text())['max_defect'],
        core_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']).read_bytes()).hexdigest(),
        qoco_sha256=hashlib.sha256(Path(os.environ['SPACEPDHCG_QOCO_LIBRARY']).read_bytes()).hexdigest())
    assert report['core_sha256']==old['core_sha256'] and report['qoco_sha256']==old['qoco_sha256']
    write(ROOT/'boundary.json',dataclasses.asdict(boundary));save()
    with (HOME/'.spacepdhcg-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with using_gpu_execution(argparse.Namespace(gpu_execution='graph',outer_loop_backend='cuda',workers=1)):
            for penalty in (100000.,1000000.):
                out=ROOT/str(int(penalty));out.mkdir()
                chosen=dataclasses.replace(settings,virtual_weight=penalty)
                write(out/'settings.json',dataclasses.asdict(chosen));report.update(stage='solve',penalty=penalty);report['native_solves']+=1;save()
                sol=solve_leg(boundary,chosen)
                np.savez_compressed(out/'raw.npz',epochs=sol.node_epochs_mjd,thrust=sol.thrust_n,states=sol.states_scaled,
                    departure_vinf=sol.departure_vinf_km_s,arrival_vinf=sol.arrival_vinf_km_s)
                write(out/'solution.json',{k:v for k,v in dataclasses.asdict(sol).items() if k not in ('boundary','node_epochs_mjd','thrust_n','states_scaled','departure_vinf_km_s','arrival_vinf_km_s')})
                record=dict(virtual_weight=penalty,status=sol.status,defect=sol.max_defect,iterations=sol.iterations,
                    accepted=sol.accepted_iterations,seconds=sol.solve_seconds,certification=None)
                if sol.status in ('converged','iteration_limit') and sol.max_defect<=chosen.defect_tolerance:
                    clamp_thrust(sol)
                    certificate=certify_legs_cuda([sol])[0]
                    record['certification']=dataclasses.asdict(certificate)
                    np.savez_compressed(out/'certified-controls.npz',epochs=sol.node_epochs_mjd,thrust=sol.thrust_n)
                report['probes'].append(record);save()
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report.update(complete=True,seconds=time.perf_counter()-started);save()
