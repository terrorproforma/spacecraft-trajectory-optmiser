import os, sys, runpy, json
sys.path.insert(0,'src')
os.environ['SPACEPDHCG_GTOC12_CUDA_LIBRARY']='/home/angus/build-spacepdhcg-verification-v175/libverify.so'
os.environ['SPACEPDHCG_GTOC12_GPU_TESTS']='1'
import numpy as np
from scipy.integrate import solve_ivp
from spacepdhcg.gtoc12.gpu_verifier import *
from spacepdhcg.gtoc12.verifier import LagrangeThrust, _thrust_dynamics, propagate_coast
from spacepdhcg.gtoc12 import constants as C
f=runpy.run_path('tests/test_gtoc12_gpu_verifier.py')['fixture']
legs,_,_=f(1,45)
times=np.linspace(3,37,37)*C.DAY_S
thrust=np.column_stack((0.15+0.12*np.sin(times/(20*C.DAY_S)),0.08*np.cos(times/(9*C.DAY_S)),np.linspace(-.02,.04,37)))
samples=np.zeros(37,dtype=SAMPLE);samples['seconds']=times;samples['thrust']=thrust
arcs=np.array([(0,37)],dtype=ARC);legs['arc_count']=1
with GpuVerifier(1,1,37) as gpu:out=gpu.propagate(legs,arcs,samples)[0]
r,v,_=propagate_coast(0,legs['initial'][0,:3],legs['initial'][0,3:6],2500,3)
fun=_thrust_dynamics(LagrangeThrust(times-times[0],thrust))
rows=[]
for tol,step in [(1e-12,np.inf),(1e-12,C.DAY_S/4),(2.3e-14,C.DAY_S/16)]:
 sol=solve_ivp(fun,(0,times[-1]-times[0]),np.r_[r,v,2500],method='DOP853',rtol=tol,atol=np.array([1e-7]*3+[1e-10]*3+[1e-9])*.01,max_step=step)
 state=sol.y[:,-1];rr,vv,_=propagate_coast(37,state[:3],state[3:6],state[6],45)
 rows.append(dict(tol=tol,max_step=step,nfev=sol.nfev,position=float(np.linalg.norm(out['final_state'][:3]-rr)),velocity=float(np.linalg.norm(out['final_state'][3:6]-vv)),mass=float(out['final_state'][6]-state[6])))
print(out);print(json.dumps(rows,indent=2))
