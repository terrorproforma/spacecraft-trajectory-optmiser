"""Append completed real captured-QP evidence; no GPU execution or core changes."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import tarfile
import numpy as np

live=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
source=live/'build/performance/halpern-real-v612'
root=live/'results/local/2026-09-09/halpern-core-v612'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old_index=json.loads((root/'sha256.json').read_text())
for name,entry in old_index.items():
    p=root/name;assert p.stat().st_size==entry['bytes'] and digest(p)==entry['sha256'],name
report=json.loads((source/'run/report.json').read_text())
assert report['complete'] and len(report['cases'])==10 and report['solve_API_calls']==14 and report['bootstrap_iterations']==4
assert not report.get('failure')
target=root/'real';target.mkdir(exist_ok=False)
for name in ('manifest.json','run.py','independent_auditor.py'):shutil.copy2(source/name,target/name)
shutil.copy2(source/'run/report.json',target/'report.json')
shutil.copytree(source/'inputs',target/'inputs')
assert digest(target/'manifest.json')==report['manifest_sha256']==digest(root/'cpu/v612d/manifest.json')
assert digest(target/'run.py')==report['runner_sha256']
assert digest(target/'independent_auditor.py')=='0d944f768ad9c089499cc6e95cd1313677e153f650eced2c7d84b9815c00c7ff'
manifest=json.loads((target/'manifest.json').read_text())
assert report['core_sha256']==manifest['library_sha256'] and report['binary_sha256']==manifest['persistent_snapshot_replay_sha256']
for name,expected in report['inputs_sha256'].items():assert digest(target/'inputs'/name)==expected
module={'__name__':'halpern_publication_audit'}
exec(compile((target/'independent_auditor.py').read_bytes(),str(target/'independent_auditor.py'),'exec'),module)
problems={name:module['load_snapshot'](target/'inputs'/(name+'.txt')) for name in ('conditioning','difficult')}
audit=[];members={};cold=[];bootstrap_count=0
archive_path=target/'raw-logs.tar.gz'
with tarfile.open(archive_path,'w:gz',format=tarfile.GNU_FORMAT) as archive:
    for case in report['cases']:
        name=case['name']+'.log';path=source/'run'/name;data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==case['log_sha256'] and case['returncode']==0
        records={}
        for line in data.decode().splitlines():
            prefix,text=line.split(' ',1);records.setdefault(prefix,[]).append(json.loads(text))
        meta=records['PERSISTENT_REPLAY_META'][0];final=records['PERSISTENT_REPLAY'][0]
        assert len(records['PERSISTENT_REPLAY_META'])==len(records['PERSISTENT_REPLAY'])==1
        assert {k:v for k,v in final.items() if k not in ('x','x_solver','y','z','s')}==case['final']
        assert meta['input_sha256']==report['inputs_sha256'][case['capture']+'.txt']
        assert meta['source_commit']==manifest['frozen_commit'] and meta['source_sha256']==manifest['compiled_snapshot_source_sha256']
        assert meta['library_sha256']==report['core_sha256'] and meta['halpern_mode']==case['mode']
        assert meta['requested_execution_blocks']==final['execution_blocks']==report['execution_blocks']==128
        assert meta['stopping_policy']=='gpu_common_kkt_original_equations'
        fresh=module['audit'](problems[case['capture']],final,backend='persistent',coordinates='original')
        assert fresh==case['audit']
        assert fresh['qualified']==final['qualified_original']==final['gpu_common_kkt']['passes']
        assert final['gpu_common_kkt']['valid'] and final['gpu_common_kkt']['finite']
        assert final['recovery_iterations']==0 and final['recovery_seconds']==0 and not final['deadline_requested']
        seed_bits=None
        bootstrap=records.get('PERSISTENT_REPLAY_BOOTSTRAP',[]);bootstrap_count+=sum(v['iterations'] for v in bootstrap)
        assert bootstrap==case['bootstrap'] and records.get('PERSISTENT_REPLAY_PRESTEP',[])==case['pre']
        if case['seeded']:
            initial=records['PERSISTENT_REPLAY_INITIAL_POINT'][0]
            assert initial['supplied_qualified'] and initial['mapped_reference_qualified']
            assert len(bootstrap)==1 and bootstrap[0]['iterations']==1
            assert final['iterations']==0 and final['termination']==1 and fresh['qualified']
            assert records['PERSISTENT_REPLAY_PRESTEP'][0]['native_seed_verified_unchanged']
            seed_bits={key:np.array_equal(np.asarray(initial[key],dtype=np.float64).view(np.uint64),np.asarray(final[key],dtype=np.float64).view(np.uint64)) for key in ('x','y','z')}
            assert all(seed_bits.values())
        else:
            assert not bootstrap and final['iterations']==100000 and final['termination']==2 and not fresh['qualified']
            cold.append({'capture':case['capture'],'mode':case['mode'],'iterations':final['iterations'],
                'solve_seconds':final['solve_seconds'],'primal':fresh['primal'],'dual':fresh['dual'],'gap':fresh['gap'],
                'block_complementarity':fresh['complementarity_max_relative'],'primal_cone_violation':fresh['primal_cone_violation'],
                'qualified':fresh['qualified'],'halpern':final.get('halpern')})
        audit.append({'case':case['name'],'log_sha256':case['log_sha256'],'raw_report_agree':True,
            'fresh_audit':fresh,'seed_bit_preservation':seed_bits,'fixed_blocks':128})
        info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644
        archive.addfile(info,io.BytesIO(data));members[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
assert bootstrap_count==4 and len(cold)==6
with tarfile.open(archive_path) as archive:
    assert len(archive.getmembers())==10
    for name,entry in members.items():
        data=archive.extractfile(name).read();assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
(target/'raw-members.json').write_text(json.dumps(members,indent=2,sort_keys=True))
findings={'report_sha256':digest(target/'report.json'),'raw_archive_sha256':digest(archive_path),
    'auditor_sha256':digest(target/'independent_auditor.py'),'fresh_cpu_audit':'FP64 input/reconstruction and long-double original sparse products; unchanged 1e-9 relative/1e-8 cone gates',
    'longdouble_mantissa_bits':int(np.finfo(np.longdouble).nmant),'real_executions':10,'real_solve_api_calls':14,
    'bootstrap_iterations':4,'seeded_optimization_iterations':0,'cold_optimization_iterations':600000,
    'qualified_seed_cases':4,'qualified_cold_cases':0,'all_cpu_native_gpu_gate_verdicts_agree':True,
    'cases':audit,'cold_summary':cold,
    'scope':'Two captured conic LPs, not full nonlinear trajectories or GTOC12 fleet score. No default promotion or SOTA claim.'}
(root/'audit/real-findings.json').write_text(json.dumps(findings,indent=2))
shutil.copy2(__file__,root/'audit/append_halpern_real_v612.py')
table='| Capture | Mode | Native solve seconds | Relative gap | Qualified |\n|---|---|---:|---:|---|\n'
for row in cold:table+=f"| {row['capture']} | {row['mode']} | {row['solve_seconds']:.6f} | {row['gap']:.10g} | No |\n"
(target/'README.md').write_text('''# Completed real captured-problem comparison

Ten process executions completed once: four supplied-start checks plus six cold comparisons. They made 14 solve API calls because each supplied-start execution included one separately recorded one-iteration bootstrap; the actual seeded solves then accepted at zero iterations. There were four bootstrap iterations and 600,000 cold iterations, with no recovery or deadline cancellation.

Every arm used the same frozen v612d binary/core, original unshifted zero-Q captures, explicit 128 cooperative blocks on the RTX 5090, and the unchanged original-equation gate: relative primal, dual, gap and maximum block complementarity at most 1e-9, and primal/dual cone violation at most 1e-8. Cold calls had a 100,000-iteration cap and 30-second deadline. The supplied-start phases used a 20-second deadline. The logs retain separate setup, scaling, solve, download and audit times.

All four supplied starts remained qualified and preserved the original x/y/z FP64 bits. **None of the six cold calls qualified**; all reached exactly 100,000 iterations. The following are unitless conic gaps, not the GTOC12 kilogram score.

'''+table+'''
Adaptive restart reduced the conditioning gap substantially but still failed primal feasibility, cone membership and block complementarity (the latter was about 0.4059). On the difficult capture its gap was worse than the ordinary method. Both plain-Halpern gaps exceeded one. Lower recorded iteration-loop time is not a verified-solution speedup because every cold result failed the common gate. These single-run timings do not establish throughput or SOTA performance, and the algorithm remains opt-in.

The native solve timer includes iteration and residual-check work plus the empty recovery timer events; it excludes scaling, setup, output download and CPU audit. Default/off is the existing dual-first map. Plain/adaptive use a primal-first reflected map, so off-versus-plain changes multiple mechanisms. Plain-versus-adaptive isolates the restart/weight bundle.

[report.json](report.json) is the immutable completed runner report. [raw-logs.tar.gz](raw-logs.tar.gz) contains all ten full stdout/stderr logs, including original x/y/z/slack and native x_solver arrays; [raw-members.json](raw-members.json) binds each exact member. The four [inputs](inputs/conditioning.txt), [runner](run.py), [auditor](independent_auditor.py), and [manifest](manifest.json) are retained. [Fresh publication audits](../audit/real-findings.json) independently recomputed the saved-vector CPU gate and checked source/input/raw/report identities; all CPU, native diagnostic and GPU gate verdicts agree.
''')
readme=(root/'README.md').read_text()
pending='The captured-problem comparison is **not included in this version**. Its independently audited completed results will be appended separately when available; no result is inferred from the tiny fixtures. The recorded real-run plan uses identical captures and fixed grid/accuracy settings across same-binary common baseline, plain and adaptive modes.'
assert pending in readme
actual='''The [completed real comparison](real/README.md) adds ten executions and 14 solve API calls at fixed 128 blocks: four qualified starts accepted at zero steps (plus four separately counted bootstrap iterations), and six cold calls that all reached 100,000 iterations without qualifying. Adaptive relative gaps were 0.02554017084 / 0.01844017712, versus 0.9999243058 / 0.01122705047 for the same-binary ordinary method; both plain gaps exceeded one. The conditioning gap improved, but other gates still failed; the difficult gap worsened. The ten complete raw logs, original vectors, inputs, runner and fresh CPU audits are preserved. This does not establish a cold-solve or SOTA improvement, and the algorithm remains opt-in.'''
readme=readme.replace(pending,actual)
readme=readme.replace('The option does not change the production GTOC12 backend or establish a fleet-score, cold-convergence or SOTA improvement.',
    'The option does not change the production GTOC12 backend or establish a fleet-score, cold-convergence or SOTA improvement.')
(root/'README.md').write_text(readme)
index={p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'sha256.json'}
(root/'sha256.json').write_text(json.dumps(index,indent=2,sort_keys=True))
print(json.dumps({'root':str(root),'files':len(index),'bytes':sum(v['bytes'] for v in index.values()),'index_sha256':digest(root/'sha256.json'),'real_audit_sha256':digest(root/'audit/real-findings.json')}))
