"""CPU-only independent audit and publication of the completed eight-case run."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import tarfile

import numpy as np

live=Path(__file__).resolve().parents[2]
analytic=live/'build/performance/common-kkt-v609-package'
real=live/'build/performance/common-kkt-real-v609e'
destination=live/'results/local/2026-09-09/persistent-common-kkt-v609'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original_report=json.loads((real/'run/report.json').read_text())
assert original_report.get('complete') or original_report.get('failure'),'original runner is not terminal'
logs=sorted(real.rglob('*.log'))
replay_logs=[p for p in logs if 'PERSISTENT_REPLAY_META ' in p.read_text()]
assert len(replay_logs)==8,'wait for all eight originally authorized executions to finish'
assert len(set(p.name for p in replay_logs))==8,'duplicate case execution'
manifest=json.loads((real/'manifest.json').read_text())
assert sha(real/'manifest.json')==original_report['source_manifest_sha256']
assert sha(real/'run.py')==original_report['runner_sha256']
assert sha(real/'independent_auditor.py')=='0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
for name in manifest['owned_paths']:
    assert sha(live/name)==manifest['source_sha256'][name],name
for name,digest in original_report['inputs_sha256'].items():assert sha(real/'inputs'/name)==digest
old_index=json.loads((analytic/'sha256.json').read_text())
for name,entry in old_index.items():assert (analytic/name).stat().st_size==entry['bytes'] and sha(analytic/name)==entry['sha256']

spec=importlib.util.spec_from_file_location('independent_auditor',real/'independent_auditor.py')
auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
cases=[];issues=[]
def same_bits(a,b):return np.array_equal(np.asarray(a,dtype=np.float64).view(np.uint64),np.asarray(b,dtype=np.float64).view(np.uint64))
def only(records,prefix):
    values=[value for key,value in records if key==prefix];assert len(values)==1,prefix;return values[0]
for log in replay_logs:
    records=[]
    for line in log.read_text().splitlines():
        if line.startswith('PERSISTENT_REPLAY'):
            prefix,payload=line.split(' ',1);records.append((prefix,json.loads(payload,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))))
    meta=only(records,'PERSISTENT_REPLAY_META');final=only(records,'PERSISTENT_REPLAY')
    initial=[v for k,v in records if k=='PERSISTENT_REPLAY_INITIAL_POINT']
    bootstrap=[v for k,v in records if k=='PERSISTENT_REPLAY_BOOTSTRAP']
    prestep=[v for k,v in records if k=='PERSISTENT_REPLAY_PRESTEP']
    seeded=bool(initial);common='gpu_common_kkt' in final
    capture='conditioning' if log.name.startswith('conditioning') else 'difficult'
    snapshot=real/'inputs'/(capture+'.txt');problem=auditor.load_snapshot(snapshot)
    assert meta['input_sha256']==sha(snapshot) and not meta['shifted'] and not meta['fold_singleton_bounds']
    assert meta['library_sha256']==manifest['library_sha256']==original_report['core_sha256']
    assert meta['source_sha256']==manifest['compiled_snapshot_source_sha256']
    assert meta['source_commit']==manifest['frozen_commit']
    assert meta['requested_execution_blocks']==final['execution_blocks']
    assert final['execution_blocks'] in (0,2) and (seeded or final['execution_blocks']==0)
    assert len(initial)==len(bootstrap)==len(prestep)==int(seeded)
    audit=auditor.audit(problem,final,backend='persistent',coordinates='original')
    if audit['qualified']!=final['qualified_original']:issues.append(log.name+': native CPU and fresh Python qualification disagree')
    if audit['passes_common_kkt_gate']!=final['kkt_qualified_original']:issues.append(log.name+': native CPU and fresh Python common gate disagree')
    if common and final['gpu_common_kkt']['valid']:
        if final['gpu_common_kkt']['passes']!=audit['passes_common_kkt_gate']:issues.append(log.name+': current GPU and fresh Python common gate disagree')
    if common:
        assert final['gpu_common_kkt']['enabled'] and final['recovery_iterations']==0 and final['recovery_seconds']==0
        if not final['gpu_common_kkt']['valid']:
            assert final['termination']==3 and not final['solver_optimal'],'invalid common record cannot certify'
    seeded_audit=None;bits=None
    if seeded:
        assert common and final['iterations']==0 and final['termination']==1
        assert meta['initial_point_sha256']==sha(real/'inputs'/(capture+'-initial.txt'))==initial[0]['point_sha256']
        assert initial[0]['supplied_qualified'] and initial[0]['mapped_reference_qualified']
        seeded_audit=auditor.audit(problem,initial[0],backend='persistent',coordinates='original')
        assert seeded_audit['passes_common_kkt_gate']
        bits={key:same_bits(initial[0][key],final[key]) for key in ('x','y','z','x_solver')}
        assert all(bits.values()) and prestep[0]['native_seed_verified_unchanged']
        assert prestep[0]['seeded_iterations']==0 and prestep[0]['termination']==0
        assert bootstrap[0]['iterations']<=1 and final['gpu_common_kkt']['evaluated_iteration']==0
    gpu_delta={}
    if common and final['gpu_common_kkt']['valid']:
        mapping={'primal':'primal','dual':'dual','gap':'gap','block_complementarity_normalized':'complementarity_max_relative',
                 'primal_cone_violation':'primal_cone_violation','dual_cone_violation':'dual_cone_violation',
                 'stationarity_absolute':'dual_absolute','objective':'objective','dual_objective':'dual_objective'}
        gpu_delta={k:abs(final['gpu_common_kkt'][k]-audit[v]) for k,v in mapping.items()}
    cases.append({'name':log.stem,'log_member':log.relative_to(real).as_posix(),'log_sha256':sha(log),'capture':capture,
                  'seeded':seeded,'common':common,'execution_blocks':final['execution_blocks'],
                  'snapshot_sha256':sha(snapshot),'solve_API_calls':1+len(bootstrap),'bootstrap_iterations':sum(v['iterations'] for v in bootstrap),
                  'seeded_source_audit':seeded_audit,'initial_final_fp64_bits_equal':bits,
                  'fresh_independent_audit':audit,'current_GPU_metric_absolute_differences':gpu_delta,
                  'final':{k:v for k,v in final.items() if k not in ('x','x_solver','y','z','s')}})
assert sum(row['solve_API_calls'] for row in cases)==12
assert sum(row['bootstrap_iterations'] for row in cases)<=4
assert sum(row['seeded'] for row in cases)==4
assert all(row['fresh_independent_audit']['qualified'] for row in cases if row['seeded'])
summary={'complete':True,'scope':'original-equation conic stopping diagnostic, not nonlinear trajectory or throughput certification',
         'real_executions':len(cases),'real_solve_API_calls':sum(row['solve_API_calls'] for row in cases),
         'real_bootstrap_iterations':sum(row['bootstrap_iterations'] for row in cases),
         'real_actual_iterations':sum(row['final']['iterations'] for row in cases),
         'seeded_qualified':sum(row['fresh_independent_audit']['qualified'] for row in cases if row['seeded']),
         'cold_qualified':sum(row['fresh_independent_audit']['qualified'] for row in cases if not row['seeded']),
         'cold_deadline_requested':sum(row['final']['deadline_requested'] for row in cases if not row['seeded']),
         'fresh_python_vs_native_CPU_all_gates_agree':not issues,'issues':issues,
         'core_sha256':original_report['core_sha256'],'replay_sha256':original_report['executable_sha256'],
         'source_manifest_sha256':sha(real/'manifest.json'),'auditor_sha256':sha(real/'independent_auditor.py'),
         'longdouble_mantissa_bits':int(np.finfo(np.longdouble).nmant),'source_owned_files_match_live':True,'cases':cases}
destination.mkdir(exist_ok=False)
shutil.copytree(analytic,destination/'analytic')
(destination/'real').mkdir();(destination/'audit').mkdir()
shutil.copy2(__file__,destination/'audit/publish_common_kkt_v609.py')
raw_members={}
with tarfile.open(destination/'real/raw-logs.tar.gz','w:gz') as archive:
    for log in logs:
        name=log.relative_to(real).as_posix();raw_members[name]={'bytes':log.stat().st_size,'sha256':sha(log)}
        archive.add(log,arcname=name,recursive=False)
for path in real.rglob('*'):
    if not path.is_file() or path.suffix in ('.log','.pyc') or '__pycache__' in path.parts:continue
    target=destination/'real'/path.relative_to(real);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
(destination/'real/raw-log-members.json').write_text(json.dumps(raw_members,indent=2,allow_nan=False))
(destination/'audit/findings.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
with tarfile.open(destination/'real/raw-logs.tar.gz','r:gz') as archive:
    for name,entry in raw_members.items():
        content=archive.extractfile(name).read();assert len(content)==entry['bytes'] and hashlib.sha256(content).hexdigest()==entry['sha256']
lines=['# Optional common-KKT stopping on identical captured problems','',
       'The new GPU stopping option accepts all four already-qualified real-capture starts at zero optimization iterations, with bit-for-bit unchanged primal and original equality/conic dual vectors. Independent original-equation audits pass for both execution strategies. The four cold runs remain unqualified at their fixed deadlines; this experiment establishes an acceptance correction, not a cold-convergence or mission-score improvement.','',
       '| Capture | Start / rule | Kernel blocks | Termination | Iterations | Normalized gap | Native solve seconds | Independent qualification |',
       '|---|---|---:|---|---:|---:|---:|---|']
for row in cases:
    f=row['final'];a=row['fresh_independent_audit'];label=('qualified seed' if row['seeded'] else 'cold')+' / '+('common KKT' if row['common'] else 'native natural')
    lines.append(f"| {row['capture']} | {label} | {row['execution_blocks']} | {f['termination_name']} | {f['iterations']:,} | {a['gap']:.9g} | {f['solve_seconds']:.6f} | {'Pass' if a['qualified'] else 'Fail'} |")
lines+=['','Block setting 0 means the single-block kernel; 2 means two cooperative blocks. The cold comparison forces setting 0 in both arms. It does not benchmark native automatic selection, which can choose a much larger grid, and does not establish general throughput or state-of-the-art performance. Both arms retain 1e-9 requested solver tolerances, a 100,000-iteration cap and the same 60-second phase deadline. Returned iteration counts can differ because both runs were bounded by wall time, not equal work.','',
        'The common rule uses normalized original-equation primal, dual, gap and maximum block-complementarity gates of 1e-9, and separate absolute primal/dual cone gates of 1e-8. The old absolute natural-residual telemetry is still computed. A passing original-equation point need not pass that distinct native natural threshold. The new policy checks before an update, preserves cancellation precedence and disables recovery explicitly. It is opt-in, restricted to unshifted generic captures with free primal variables, and assumes a full symmetric convex Hessian; these two captures have zero numerical Hessians.','',
        'A GPU common record is a certificate for the current reported iterate only when valid=true. Cancellation between scheduled checks invalidates that record; cancellation at a completed check may leave current metrics valid while termination remains cancelled. Neither case grants solver acceptance. Every returned vector set in this bundle has a fresh independent CPU audit, including deadline outcomes. GPU compensated FP64 and CPU long-double arithmetic are compared at the same fixed gate, without claiming bit-identical arithmetic at arbitrary boundaries.','',
        f"The real experiment ran eight executables and {summary['real_solve_API_calls']} solve API calls, including {summary['real_bootstrap_iterations']} separately reported unseeded bootstrap iterations. The four seeded actual solves used zero iterations. The analytic bundle adds two executables, fourteen bounded solve calls and six actual iterations. Bootstrap, setup, solve-event, download and audit times are separate in raw records. Native solve time includes legacy natural-report work and common evaluation; common evaluation_clock_cycles measures block-zero clock cycles, not seconds. The strong block-count dependence of these diagnostics must not be presented as end-to-end trajectory throughput.",'',
        'Default kernels are separate compiled instantiations; their register/shared/stack footprints match the frozen v603 baseline. The optional kernels have separate occupancy limits. The final core preserves all non-persistent v603 object files by hash. No physics tolerances, production GTOC12 backend, fleet itinerary, score or incumbent changed in this experiment.','',
        'Evidence: [analytic source/build/tiny checks](analytic/README.md), [original real runner report](real/run/report.json), [fresh independent audit and counts](audit/findings.json), and [raw-log member hashes](real/raw-log-members.json). The compressed raw logs retain every original point and cold-result vector, metadata, bootstrap and pre-step record. Inputs and exact runners/auditor are retained in real/. Earlier compile failures and the superseded CPU-only runtime-branch prototype are preserved in analytic/earlier-builds/. The parent sha256.json indexes every file except itself.','']
if issues:lines+=['Audit issues requiring review:',*['- '+issue for issue in issues],'']
(destination/'README.md').write_text('\n'.join(lines))
index={p.relative_to(destination).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(destination.rglob('*')) if p.is_file()}
(destination/'sha256.json').write_text(json.dumps(index,indent=2))
print(json.dumps({'directory':str(destination),'files':len(index),'bytes':sum(v['bytes'] for v in index.values()),
                  'index_sha256':sha(destination/'sha256.json'),'seeded_qualified':summary['seeded_qualified'],
                  'cold_qualified':summary['cold_qualified'],'issues':issues}))
