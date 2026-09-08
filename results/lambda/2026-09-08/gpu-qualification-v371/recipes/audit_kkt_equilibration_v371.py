from pathlib import Path
import numpy as np,subprocess,fcntl,json,hashlib
root=Path('build/performance/kkt-equilibration-probe-v371');root.mkdir(exist_ok=False)
binary='/home/angus/kkt-equilibration-probe-v371'
subprocess.run(['/usr/local/cuda-12.8/bin/nvcc','-O3','-arch=sm_120','--default-stream','per-thread','build/performance/kkt_equilibration_probe_v371.cu','-o',binary],check=True)
snap=Path('/home/angus/build-qoco-linear-snapshot-v333/snapshots/linear-1110-0052-1.bin')
with snap.open('rb') as f:
 n,nnz,*_=map(int,np.fromfile(f,np.int32,8));np.fromfile(f,np.float64,3)
 rows=np.fromfile(f,np.int32,n+1);cols=np.fromfile(f,np.int32,nnz)
 raw=np.fromfile(f,np.float64,nnz);rhs=np.fromfile(f,np.float64,n);sol=np.fromfile(f,np.float64,n)
assert np.isfinite(raw).all() and np.isfinite(rhs).all() and np.isfinite(sol).all()
indices=np.repeat(np.arange(n),np.diff(rows))
with (root/'inputs.bin').open('wb') as f:
 for a in [np.array([n,nnz],np.int32),rows,cols,raw,rhs,sol]:a.tofile(f)
with open('/home/angus/.spacepdhcg-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 subprocess.run([binary,str(root/'inputs.bin'),str(root/'outputs.bin')],check=True)
out=np.fromfile(root/'outputs.bin',np.float64);offset=0;reports=[]
for repeat in range(2):
 scale=out[offset:offset+n];offset+=n
 matrix=out[offset:offset+nnz];offset+=nnz
 b=out[offset:offset+n];offset+=n;x=out[offset:offset+n];offset+=n
 unchanged=out[offset:offset+nnz];offset+=nnz
 values=raw*(4. if repeat else 1.);expected=np.ones(n)
 for _ in range(4):
  maxima=np.zeros(n);v=np.abs((values*expected[indices])*expected[cols])
  np.maximum.at(maxima,indices,v);np.maximum.at(maxima,cols,v)
  keep=(maxima>0)&np.isfinite(maxima);expected[keep]/=np.sqrt(maxima[keep])
 np.testing.assert_allclose(scale,expected,rtol=5e-15,atol=0)
 np.testing.assert_allclose(matrix,(values*scale[indices])*scale[cols],rtol=5e-15,atol=0)
 np.testing.assert_array_equal(unchanged,values)
 np.testing.assert_allclose(b,rhs*scale,rtol=5e-15,atol=0)
 np.testing.assert_allclose(x,sol*scale,rtol=5e-15,atol=0)
 assert np.isfinite(scale).all() and (scale>0).all()
 reports.append(dict(repeat=repeat,scale_min=float(scale.min()),scale_max=float(scale.max()),raw_matrix_max=float(abs(values).max()),scaled_matrix_max=float(abs(matrix).max()),raw_values_unchanged=True))
assert offset==len(out)
report=dict(passed=True,rows=n,nonzeros=nnz,snapshot_sha256=hashlib.sha256(snap.read_bytes()).hexdigest(),cases=reports)
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
