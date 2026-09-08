"""CPU-only spectral and step-balance analysis of the two exact zero-Q captures."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

root=Path(__file__).resolve().parents[3]
inputs=root/'results/local/2026-09-09/persistent-common-kkt-v609/real/inputs'
auditor_path=root/'results/local/2026-09-09/persistent-common-kkt-v609/real/independent_auditor.py'
spec=importlib.util.spec_from_file_location('audit',auditor_path)
auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
array_sha=lambda a:hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
old=json.loads((root/'build/performance/objective-balance-v606/findings.json').read_text())
cases=[]
for capture in ('conditioning','difficult'):
    start=time.perf_counter();path=inputs/(capture+'.txt');q=auditor.load_snapshot(path)
    assert q['quadratic_numerical_nonzeros']==0 and not q['shift']
    K=sp.vstack((q['A'],q['G']),format='csc').astype(np.float64)
    # SOC row signs/permutations do not change K^T K, and cone-preserving Ruiz
    # makes every row scale inside each SOC identical in either ordering.
    n=q['n'];rows=K.shape[0];cols=np.repeat(np.arange(n),np.diff(K.indptr));indices=K.indices
    D=np.ones(n);R=np.ones(rows)
    for _ in range(10):
        scaled=np.abs(K.data)/(R[indices]*D[cols]);column_max=np.zeros(n);row_max=np.zeros(rows)
        np.maximum.at(column_max,cols,scaled);np.maximum.at(row_max,indices,scaled)
        offset=q['p']+q['l']
        for size in q['soc']:
            size=int(size);row_max[offset:offset+size]=np.max(row_max[offset:offset+size]);offset+=size
        D*=np.where(column_max>1e-12,np.sqrt(column_max),1)
        R*=np.where(row_max>1e-12,np.sqrt(row_max),1)
    prior=next(case for case in old['cases'] if case['capture']==capture)
    assert array_sha(D)==prior['D_sha256_le_float64'] and array_sha(R)==prior['R_sha256_le_float64']
    scaled=K.copy();scaled.data/=R[indices]*D[cols]
    rhs=np.concatenate((q['b'],q['h'])).astype(np.float64);c=np.asarray(q['c'],dtype=np.float64)
    B=1/(math.sqrt(sum(float(x)**2 for x in rhs/R))+1)
    O=1/(math.sqrt(sum(float(x)**2 for x in c/D))+1)
    assert abs(B-prior['bound_scale_B'])<1e-16 and abs(O-prior['objective_scale_O'])<1e-20
    v=np.ones(n)/math.sqrt(n);history=[]
    for iteration in range(20):
        r=scaled@v;rn=np.linalg.norm(r)
        if rn<=1e-12:estimate=0.;break
        g=scaled.T@(r/rn);estimate=np.linalg.norm(g);v=g/estimate
        history.append(float(estimate))
    gram=scaled.T@scaled
    rng=np.random.default_rng(610)
    values,vectors=spla.eigsh(gram,k=1,which='LA',tol=1e-12,maxiter=2000,v0=rng.normal(size=n))
    eigenvalue=float(values[0]);singular=math.sqrt(eigenvalue);eigenvector=vectors[:,0]
    eigen_residual=float(np.linalg.norm(gram@eigenvector-eigenvalue*eigenvector))
    absK=abs(scaled);upper=math.sqrt(float(np.max(np.asarray(absK.sum(axis=0))))*float(np.max(np.asarray(absK.sum(axis=1)))))
    eta=.9/max(1,float(estimate));tau=eta*O/(B*D**2);sigma=eta*B/(O*R**2)
    point=inputs/(capture+'-initial.txt');lines=point.read_text().splitlines();vectors={}
    assert lines[1]=='snapshot_sha256 '+sha(path)
    for line in lines[3:]:
        parts=line.split();vectors[parts[0]]=np.asarray(parts[2:],dtype=np.float64);assert int(parts[1])==len(vectors[parts[0]])
    xp=B*D*vectors['x'];yp=O*R*np.concatenate((vectors['y'],vectors['z']))
    xn=float(np.linalg.norm(xp));yn=float(np.linalg.norm(yp));oracle_weight=yn/xn
    cases.append({'capture':capture,'snapshot_sha256':sha(path),'point_sha256':sha(point),'n':n,'rows':rows,
                  'stored_K_entries':int(K.nnz),'scaled_K_sha256':array_sha(scaled.data),'D_sha256':array_sha(D),'R_sha256':array_sha(R),
                  'D_range':[float(D.min()),float(D.max())],'R_range':[float(R.min()),float(R.max())],
                  'bound_scale_B':B,'objective_scale_O':O,'O_over_B':O/B,
                  'native_power20_estimates':history,'power20_estimate':float(estimate),'largest_singular_value_numeric':singular,
                  'largest_eigenpair_residual_l2':eigen_residual,'safe_one_infinity_norm_upper_bound':upper,
                  'power20_over_numeric_norm':float(estimate)/singular,'eta':eta,
                  'eta_squared_numeric_norm_squared':eta*eta*eigenvalue,'eta_squared_safe_upper_squared':(eta*upper)**2,
                  'original_primal_step_range':[float(tau.min()),float(tau.max())],
                  'original_dual_step_range':[float(sigma.min()),float(sigma.max())],
                  'known_point_scaled_primal_l2':xn,'known_point_scaled_dual_l2':yn,
                  'oracle_distance_balancing_omega':oracle_weight,
                  'oracle_tau_multiplier':1/oracle_weight,'oracle_sigma_multiplier':oracle_weight,
                  'baseline_over_optimal_ergodic_distance_bound_coefficient':(xn*xn+yn*yn)/(2*xn*yn),
                  'cpu_seconds':time.perf_counter()-start})
report={'scope':'CPU source/scaling analysis only; no solver or GPU run, no speedup prediction',
        'script_sha256':sha(Path(__file__)),'auditor_sha256':sha(auditor_path),
        'source_sha256':{p:sha(root/p) for p in ('cpp/cuda/src/persistent_pdhcg.cu','cpp/cuda/src/cooperative_pdhg.cuh')},
        'arithmetic':'FP64 scaling and SciPy sparse products; native atomic/reduction ordering can differ',
        'norm_caveat':'eigenvalue is a numerical estimate with reported residual, not a formal spectral upper certificate; one/infinity bound is conservative',
        'coordinate_derivation':'Ktilde=R^-1 K D^-1; xtilde=B D x; ytilde=O R y; Qtilde=(O/B)D^-1 Q D^-1',
        'balance_derivation':'For zero Q, tau=eta/omega and sigma=eta*omega preserve tau*sigma and the PDHG spectral condition. An oracle distance-bound coefficient omega*||xtilde*||^2+||ytilde*||^2/omega is minimized by omega=||ytilde*||/||xtilde*||. This uses a known solution and is diagnostic, not a deployable heuristic.',
        'cases':cases}
output=Path(__file__).with_name('scaling-findings.json');output.write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({'path':str(output),'cases':[{k:r[k] for k in ('capture','power20_estimate','largest_singular_value_numeric','eta_squared_numeric_norm_squared','oracle_distance_balancing_omega','baseline_over_optimal_ergodic_distance_bound_coefficient')} for r in cases]}))
