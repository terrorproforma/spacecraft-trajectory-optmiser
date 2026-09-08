"""Publish completed source/CPU/tiny evidence, with no new GPU work."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

live=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
cpu=live/'build/performance/halpern-v612-cpu'
root=live/'results/local/2026-09-09/halpern-core-v612'
assert root.is_dir() and (root/'tiny/report.json').is_file()
assert not (root/'README.md').exists() and not (root/'cpu').exists()
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cpu_index=json.loads((cpu/'sha256.json').read_text())
for name,entry in cpu_index.items():
    path=cpu/name;assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],name
shutil.copytree(cpu,root/'cpu')
manifest=json.loads((root/'cpu/v612d/manifest.json').read_text())
assert digest(root/'cpu/v612d/manifest.json')=='c64efa7a6356d3a17f9f29e2ba2c5706c5bda180ac7cf8e6cf919ba6b422300a'
with tarfile.open(root/'cpu/v612d/source.tar.gz') as archive:
    source_matches={name:hashlib.sha256(archive.extractfile(name).read()).hexdigest()==manifest['source_sha256'][name]==digest(live/name)
                    for name in manifest['owned_paths']}
assert all(source_matches.values())
tiny=root/'tiny';report=json.loads((tiny/'report.json').read_text())
assert digest(tiny/'report.json')=='dbe2072d2873be88dde87e6d6c135ebe15cc11d37639d876acec18d239707acc'
assert report['complete'] and report['status']=='passed' and report['actual_solve_api_calls']==14
assert report['manifest_sha256']==digest(root/'cpu/v612d/manifest.json')
assert report['core_sha256']==manifest['library_sha256'] and report['test_sha256']==manifest['persistent_halpern_test_sha256']
assert report['runner_sha256']==digest(tiny/'run_halpern_tiny_v612.py')
expected={'mixed_exact_zero_step':(1,0),'two_step_proximal_output':(2,2),'disabled_default_transition':(2,1),
          'mixed_cold_proximal_step':(2,1),'first_restart_epoch':(2,201),'cancel_before_initial':(3,0),'nonfinite_initial':(4,0)}
audited=[];iterations=0;terminations={}
for case in report['cases']:
    path=tiny/(case['mode']+'.log');assert digest(path)==case['log_sha256']
    raw=[{'prefix':line.split(' ',1)[0],'record':json.loads(line.split(' ',1)[1])} for line in path.read_text().splitlines()]
    assert raw==case['records'] and case['returncode']==0
    actual=[entry['record'] for entry in raw if entry['prefix']=='HALPERN_TEST']
    assert len(actual)==7 and {entry['case'] for entry in actual}==set(expected)
    for entry in actual:
        assert (entry['termination'],entry['iterations'])==expected[entry['case']]
        iterations+=entry['iterations'];terminations[entry['termination']]=terminations.get(entry['termination'],0)+1
        if entry['case']=='mixed_exact_zero_step':assert entry['common_valid'] and entry['common_passes'] and entry['updates']==0
        if entry['case']=='cancel_before_initial':assert not entry['common_valid'] and not entry['common_passes']
        if entry['case']=='nonfinite_initial':assert not entry['finite'] and not entry['common_passes'] and entry['gap'] is None
        if entry['case']=='disabled_default_transition':assert not entry['halpern_valid']
        if entry['case']=='first_restart_epoch':
            if case['mode']=='adaptive':assert entry['restarts']==1 and entry['inner']==1 and entry['epoch_reference']==201 and entry['weight_updates']==1
            else:assert entry['restarts']==0 and entry['inner']==201 and entry['weight']==1
    audited.append({'mode':case['mode'],'raw_log_sha256':digest(path),'records_equal_report':True,'expected_outcomes_match':True})
assert iterations==410==report['actual_optimization_iterations']
audit={'scope':'Independent CPU inspection of existing logs/source; no GPU rerun or separate vector re-export.',
       'report_sha256':digest(tiny/'report.json'),'source_matches_frozen_archive_and_live':source_matches,
       'actual_executions':2,'actual_solve_api_calls':14,'actual_optimization_iterations':410,
       'native_termination_counts':terminations,'all_expected_outcomes_pass':True,'raw_cases':audited,
       'caveats':['The two zero-step exact mixed seeds qualify; eight deliberate bounded iteration-limit outcomes are not solver successes.',
          'Cancellation and nonfinite outcomes are expected test assertions. Invalid/disabled diagnostic zeros are not certificates.',
          'Tiny raw logs expose metrics and assertion outcomes, not all iterate arrays; independent scalar/SOC oracles and unchanged-point checks run inside the pinned test executable.',
          'No captured-problem result, throughput gain or fleet-score improvement is included in this package.']}
(root/'audit').mkdir();(root/'audit/tiny-findings.json').write_text(json.dumps(audit,indent=2))
shutil.copy2(__file__,root/'audit/publish_halpern_core_v612.py')
(root/'.gitattributes').write_text('* -text\n')
(root/'README.md').write_text('''# Halpern core diagnostic — local v612

The two bounded GPU test executions passed all **14 expected solver outcomes**, performing **410 optimization iterations**. The frozen implementation includes plain reflected Halpern and an adaptive restart/weight variant, enabled only for zero-Q diagnostic replays with the existing common original-equation KKT gates.

Both exact mixed equality/nonnegative/SOC seeds were accepted at zero iterations with unchanged primal/dual values. The remaining tests intentionally exercised eight iteration-limit outcomes, two cancellations before acceptance, and two nonfinite-input rejections. “Tests passed” does not mean every deliberately bounded problem was solved. The suite checks an independent two-step proximal-point oracle, the return to the ordinary solver after disabling Halpern, a cold SOC/Moreau projection step, and the first restart/new metric epoch at iterations 200/201. The adaptive fixture performed one weight update; plain mode retained unit weight.

The [tiny report](tiny/report.json) binds both raw logs, the runner, core and source manifest. [Independent log/source findings](audit/tiny-findings.json) verify its records and counts. Its 0.689-second wrapper time covers fixture setup and the complete bounded test sequence; it is not a throughput benchmark. The tests ran with two cooperative blocks per solve and have no bootstrap.

The final [v612d manifest](cpu/v612d/manifest.json) pins source commit `6fbff324b152b75e6c27e3d91b4dc0835c65ab93`. Core SHA256: `49d0eef5702a7dee319ef3b7747ded844bec0f39313300405f9d6f30fc61730e`; replay: `872ef9d6fe84e85aee2431e2d92caa1ad43bf2293ef9fc91bd9ee3ba938717f2`; test: `6da1ba381a566a23923da26d21cc9e72e9e6eb8159399c800350f6a79e73bed7`. The source archive includes the nine owned files, each matched against the live source during packaging. Binaries and caches are omitted; unchanged non-persistent linked objects remain individually identified in the manifest.

[CPU and compiler evidence](cpu/README.md) preserves all a–d attempts, including the superseded compiler resource regressions and the first configure-path failure. The final [resource comparison](cpu/resource-comparison.json) restores the existing listed kernels' register, stack and shared-memory footprints. This is compiler evidence, not a runtime speed claim. CPU tests, six hidden-device JSON parser cases, four unsupported-input rejections and both CMake registrations also passed.

See [the design](cpu/DESIGN.md) for the primal-first map, reflected/anchored state, original-equation acceptance, and restart/weight rules. The existing spectral heuristic is retained as an experimental precondition, not a general stability proof. Default-versus-plain changes both ordering and reflected state; plain-versus-adaptive isolates the restart/weight bundle. The option does not change the production GTOC12 backend or establish a fleet-score, cold-convergence or SOTA improvement.

The captured-problem comparison is **not included in this version**. Its independently audited completed results will be appended separately when available; no result is inferred from the tiny fixtures. The recorded real-run plan uses identical captures and fixed grid/accuracy settings across same-binary common baseline, plain and adaptive modes.

Every file is bound by the flat [sha256.json](sha256.json) index, excluding that index itself. The nested CPU index remains unchanged, and `.gitattributes` preserves the recorded bytes in Git.
''')
for path in root.rglob('*'):
    assert path.name!='__pycache__' and path.suffix.lower() not in ('.pyc','.so','.o','.exe'),path
index={p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file()}
(root/'sha256.json').write_text(json.dumps(index,indent=2,sort_keys=True))
print(json.dumps({'root':str(root),'indexed_files':len(index),'bytes':sum(v['bytes'] for v in index.values()),'index_sha256':digest(root/'sha256.json')}))
