from pathlib import Path
import os,sys,time,json,fcntl
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
import numpy as np
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.search import RoutePlan
from spacepdhcg.gtoc12.retiming import Retimer,build_visits,orders_of
from spacepdhcg.gtoc12 import lambert
out=Path(sys.argv[1]);source=Path(sys.argv[2]);plan=RoutePlan.from_summary(json.loads(source.read_text())[0]['plan'])
lock=open(Path.home()/'.spacepdhcg-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=[]
with lambert.using_lambert_backend('cuda') as gpu:
    timer=Retimer(load_catalogue());visits=build_visits(*orders_of(plan));masses=timer._plan_masses(plan)
    timer._dp(visits,masses,.15);ws=gpu.retime_workspace;native=ws.evaluate
    times=[]
    def measured(*args):
        start=time.perf_counter();result=native(*args);times.append(time.perf_counter()-start);return result
    ws.evaluate=measured
    if os.environ.get('SPACEPDHCG_PROFILE_CAPTURE') == '1':
        assert gpu.library.cudaProfilerStart() == 0
    for i in range(100):
        start=time.perf_counter();timer._dp(visits,masses,[.01,.03,.1,.15,.5][i%5]);total=time.perf_counter()-start
        rows.append(dict(total=total,native=times[-1],python_boundary=total-times[-1]))
    if os.environ.get('SPACEPDHCG_PROFILE_CAPTURE') == '1':
        assert gpu.library.cudaProfilerStop() == 0
report=dict(scope='Warm cached unswept DP; native includes kernels, copies and synchronization. Python boundary is total minus ctypes native call, not pure CPU profiling.',rows=rows,medians={k:float(np.median([r[k] for r in rows[5:]])) for k in rows[0]})
out.write_text(json.dumps(report,indent=2));print(json.dumps(report['medians'],indent=2))
