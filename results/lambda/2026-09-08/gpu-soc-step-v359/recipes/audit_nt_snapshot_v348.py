from pathlib import Path
from decimal import Decimal as D,getcontext
import numpy as np,json,hashlib,sys
getcontext().prec=100
root=Path('/home/angus/build-qoco-nt-snapshot-v348');output=Path('build/performance/nt-snapshot-v348');output.mkdir(exist_ok=True)
def dot(a,b):return sum((x*y for x,y in zip(a,b)),D(0))
def determinant(a):return a[0]*a[0]-dot(a[1:],a[1:])
rows=[];hashes={};valid_s=[];valid_z=[];valid_q=[];labels=[];ref_nt=[];ref_wtw=[]
for path in sorted((root/'snapshots').glob('nt-*.bin')):
 hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
 with path.open('rb') as f:
  l,nsoc,m,compact,tri=np.fromfile(f,np.int32,5);q=np.fromfile(f,np.int32,nsoc)
  s=np.fromfile(f,np.float64,m);z=np.fromfile(f,np.float64,m)
  nt=np.fromfile(f,np.float64,compact);wtw=np.fromfile(f,np.float64,tri);assert f.read()==b''
 row=dict(file=path.name,soc_count=int(nsoc),nonpositive_lp_s=int(np.sum(s[:l]<=0)),nonpositive_lp_z=int(np.sum(z[:l]<=0)),nonfinite_inputs=int(np.sum(~np.isfinite(s))+np.sum(~np.isfinite(z))),noninterior_socs=0,nonfinite_scaling_socs=0,max_relative_nt_error=0.,max_relative_wtw_error=0.,min_relative_determinant=1.,worst_soc=None)
 ix=ni=mi=int(l)
 for c,n in enumerate(q):
  ss=s[ix:ix+n];zz=z[ix:ix+n];actual_nt=nt[ni:ni+n+1];actual_mat=wtw[mi:mi+n*(n+1)//2]
  ix+=n;ni+=n+1;mi+=n*(n+1)//2
  if not(np.isfinite(ss).all() and np.isfinite(zz).all()):continue
  sd=list(map(lambda x:D(float(x)),ss));zd=list(map(lambda x:D(float(x)),zz));ds=determinant(sd);dz=determinant(zd)
  if ds<=0 or dz<=0 or sd[0]<=0 or zd[0]<=0:row['noninterior_socs']+=1;continue
  row['min_relative_determinant']=min(row['min_relative_determinant'],float(ds/(sd[0]*sd[0])),float(dz/(zd[0]*zd[0])))
  sn=ds.sqrt();zn=dz.sqrt();gamma=((1+dot(sd,zd)/(sn*zn))/2).sqrt();eta=(sn/zn).sqrt()
  w=[(sd[j]/sn+(zd[j]/zn if j==0 else -zd[j]/zn))/(2*gamma) for j in range(n)]
  ideal_nt=np.array(list(map(float,[eta]+w)))
  ideal_mat=np.array([float(eta*eta*(2*w[j]*w[k]+(-1 if j==0 else 1) if j==k else 2*w[j]*w[k])) for j in range(n) for k in range(j+1)])
  valid_s.extend(ss.tolist());valid_z.extend(zz.tolist());valid_q.append(int(n));labels.append(f'{path.stem}:{c}');ref_nt.extend(ideal_nt.tolist());ref_wtw.extend(ideal_mat.tolist())
  if not(np.isfinite(actual_nt).all() and np.isfinite(actual_mat).all()):row['nonfinite_scaling_socs']+=1;continue
  ne=float(np.max(np.abs(actual_nt-ideal_nt))/np.max(np.abs(ideal_nt)))
  me=float(np.max(np.abs(actual_mat-ideal_mat))/np.max(np.abs(ideal_mat)))
  row['max_relative_nt_error']=max(row['max_relative_nt_error'],ne)
  if me>row['max_relative_wtw_error']:row['max_relative_wtw_error']=me;row['worst_soc']=c
 rows.append(row)
with (output/'inputs.bin').open('wb') as f:
 np.array([len(valid_q),sum(valid_q)],np.int32).tofile(f);np.array(valid_q,np.int32).tofile(f)
 np.array(valid_s).tofile(f);np.array(valid_z).tofile(f)
(output/'reference.json').write_text(json.dumps(dict(labels=labels,sizes=valid_q,nt=ref_nt,wtw=ref_wtw)))
sys.path.insert(0,str(Path('build/performance/qp-ir-v309').resolve()));from audit import problem,audit
records=[json.loads(s[10:]) for s in (root/'replay.log').read_text().splitlines() if s.startswith('QP_REPLAY ')]
report=dict(note='Synchronous host-dispatched diagnostic; not a performance benchmark.',snapshots=len(rows),valid_pairs=len(valid_q),qp_audit=audit(problem(Path('build/performance/qp-ir-v309/qp.txt')),records[0]),rows=rows)
(output/'report.json').write_text(json.dumps(report,indent=2));(output/'snapshot-sha256.json').write_text(json.dumps(hashes,indent=2))
print('snapshots',len(rows),'valid_pairs',len(valid_q),'QP_audit',report['qp_audit'])
print(json.dumps(rows[-8:],indent=2))
