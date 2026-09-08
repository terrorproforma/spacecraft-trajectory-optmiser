from pathlib import Path
import numpy as np
root=Path('build/performance/step-snapshot-v351')
p=Path('/home/angus/build-qoco-step-snapshot-v351/snapshots/step-0000.bin')
with p.open('rb') as f:
 l,nsoc,m=np.fromfile(f,np.int32,3);alpha,factor=np.fromfile(f,np.float64,2);q=np.fromfile(f,np.int32,nsoc);u=np.fromfile(f,np.float64,m);du=np.fromfile(f,np.float64,m)
values=[];offsets=[0];ix=int(l)
for n in q:
 values.extend(u[ix:ix+n]);values.extend(du[ix:ix+n]);offsets.append(len(values));ix+=n
with (root/'inputs.bin').open('wb') as f:
 np.array([nsoc,len(values)],np.int32).tofile(f);np.array(offsets,np.int32).tofile(f);np.array(values).tofile(f)
