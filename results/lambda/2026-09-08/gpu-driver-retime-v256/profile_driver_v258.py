from pathlib import Path
import os,sys,time,json,fcntl
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.search import RoutePlan
from spacepdhcg.gtoc12.retiming import Retimer
from spacepdhcg.gtoc12.returnsweep import ReturnSweep
from spacepdhcg.gtoc12 import lambert
out=Path(sys.argv[1]);source=Path(sys.argv[2]);sweep=Path(sys.argv[3]);enabled=sys.argv[4]=='device'
plan=RoutePlan.from_summary(json.loads(source.read_text())[0]['plan'])
flown=json.loads(sweep.read_text())['legs'][-1]
lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=[]
with lambert.using_lambert_backend('cuda') as gpu:
    gpu.retime_cuda_driver=enabled
    timer=Retimer(load_catalogue());timer.set_return_sweep(ReturnSweep(flown['from'],flown['mass_before'],np.array([flown['t0']]),np.array([flown['tf']-flown['t0']]),np.ones((1,1),bool),np.ones((1,1),bool),np.array([[flown['delta_v_km_s']]]),np.array([[flown['propellant_kg']]])))
    reference=timer.retime(plan)
    before=dict(gpu.telemetry)
    assert gpu.library.cudaProfilerStart()==0
    for i in range(20):
        start=time.perf_counter();result=timer.retime(plan);elapsed=time.perf_counter()-start
        assert result.objective_after==reference.objective_after
        rows.append(dict(seconds=elapsed,price_rounds=result.price_rounds,mass_rounds=result.mass_rounds))
    assert gpu.library.cudaProfilerStop()==0
    counters={k:gpu.telemetry[k]-before[k] for k in ['completed_retime_dp_calls','completed_retime_forward_calls','completed_retime_driver_calls']}
out.write_text(json.dumps(dict(scope='20 warm complete retained-table retimings; timings instrumented by Nsight',gpu_driver=enabled,rows=rows,counters=counters),indent=2))
