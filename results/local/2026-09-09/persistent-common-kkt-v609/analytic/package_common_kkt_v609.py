"""Package final diagnostic source/CPU/tiny evidence without running CUDA."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile

live=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
parent=Path('/home/angus')
destination=live/'build/performance/common-kkt-v609-package'
destination.mkdir(exist_ok=False)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=parent/'spacepdhcg-common-kkt-v609d';e=parent/'spacepdhcg-common-kkt-v609e'
for name,source in [('core-build',d),('replay-build',e)]:
    out=destination/name;out.mkdir()
    for path in source.iterdir():
        if path.is_file():shutil.copy2(path,out/path.name)
shutil.copytree(d/'tiny',destination/'tiny')
shutil.copytree(d/'fixtures',destination/'fixtures')
shutil.copy2(__file__,destination/'package_common_kkt_v609.py')
resources={}
for name in ('spacepdhcg-persistent-replay-v603','spacepdhcg-common-kkt-v609c','spacepdhcg-common-kkt-v609d'):
    path=parent/name/'build/cuda/libspacepdhcg_cuda.so'
    command=['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(path)]
    output=subprocess.check_output(command,text=True)
    resource_dir=destination/'resources';resource_dir.mkdir(exist_ok=True)
    target=resource_dir/(name+'.txt');target.write_text(output)
    lines=output.splitlines();selected=[lines[i:i+2] for i,line in enumerate(lines) if 'Function' in line and 'solve_kernel' in line]
    resources[name]={'command':command,'core_sha256':digest(path),'raw_sha256':digest(target),'selected':selected}
(destination/'resources/comparison.json').write_text(json.dumps(resources,indent=2))
for version in ('a','b','c'):
    source=parent/('spacepdhcg-common-kkt-v609'+version);out=destination/'earlier-builds'/('v609'+version);out.mkdir(parents=True)
    manifest=json.loads((source/'manifest.json').read_text())
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.json','.log','.py'):shutil.copy2(path,out/path.name)
    # A compact exact source delta is sufficient with the recorded frozen base;
    # none of these earlier build attempts launched GPU work.
    with tarfile.open(out/'owned-source.tar.gz','w:gz') as archive:
        for name in manifest['owned_paths']:archive.add(source/'repo'/name,arcname=name,recursive=False)
final=json.loads((e/'manifest.json').read_text());core_manifest=json.loads((d/'manifest.json').read_text())
for name in final['owned_paths']:
    assert digest(live/name)==final['source_sha256'][name],name
    if name!='cpp/cuda/tests/persistent_snapshot_replay.cu':assert final['source_sha256'][name]==core_manifest['source_sha256'][name]
tiny=json.loads((d/'tiny/report.json').read_text())
records=[item['record'] for case in tiny['cases'] for item in case['records'] if item['prefix']=='COMMON_KKT_TEST']
assert tiny['complete'] and len(records)==14 and sum(row['iterations'] for row in records)==6
summary={'complete':True,'gpu_executables':2,'solve_api_calls':14,'actual_iterations':6,'bootstrap_iterations':0,
         'all_asserted_outcomes_passed':True,'real_captured_QP_calls_in_this_bundle':0,
         'core_sha256':final['library_sha256'],'replay_sha256':final['persistent_snapshot_replay_sha256'],
         'test_sha256':core_manifest['persistent_common_kkt_test_sha256'],
         'source_core_fields_unchanged_by_replay_selector':True,'live_owned_source_matches_frozen':True,
         'earlier_builds':{'v609a':'compile rejected missing math_constants include; zero GPU calls',
                           'v609b':'compile rejected implicit initializer under -Werror; zero GPU calls',
                           'v609c':'CPU checks passed; runtime branch increased default registers; superseded before GPU'}}
(destination/'summary.json').write_text(json.dumps(summary,indent=2))
readme='''# Optional GPU common-KKT stopping diagnostic

This bundle contains the implemented diagnostic policy, CPU checks and two bounded tiny GPU executions. All 14 expected API outcomes passed across the single-block and two-block strategies, with six total optimization iterations. The exact mixed PSD-Q/equality/nonnegative/SOC seed was accepted at iteration zero without changing x or the dual. The disabled policy retained its existing one-iteration behavior.

The tests also cover separate equality versus combined original-G normalization, block-complementarity cancellation, unsupported box/lower/two-sided domains, nonfinite values, cancellation before initial certification, reset/reseed invalidation and disable/update/revalidate/re-enable. These are analytic diagnostic tests, not trajectory certifications or a throughput benchmark. No real captured-QP GPU call is included here; the parent publication links that comparison separately.

The option is `--common-kkt-stop`; the default stopping rule remains the absolute native natural residual. The new rule uses normalized primal/equality-plus-conic-equation residual, dual stationarity, primal/dual objective gap and maximum per-scalar/per-SOC complementarity at 1e-9, plus separate absolute primal/dual cone membership at 1e-8. Finite checks apply throughout the compensated FP64 products and reductions. Natural-residual telemetry remains available. These compensated calculations are not bit-identical to the independent long-double auditor at an arbitrary threshold boundary, so external original-equation verification remains required.

The replay accepts this option only for unshifted generic captures. Shifted captures, folded variable bounds and changed audit tolerances are rejected before CUDA. The C API supports free primal variables, an equality prefix, upper-only scalar rows and completely covered contiguous standard affine SOCs. Full symmetric PSD Q is a caller precondition, not something this gate proves. This does not integrate a new GTOC12 trajectory backend or claim global optimality for arbitrary nonconvex input.

`--execution-blocks 0` selects the single-block kernel; `--execution-blocks 2` selects two cooperative blocks. The setter runs before common-policy setup and rejects an oversized grid rather than silently changing it. Metadata records the requested count and each completed replay records the accepted count. Omitting the option preserves native automatic selection and records null. Common-policy recovery is disabled, with zero recovery time; default recovery launch and numerical defaults remain intact.

The GPU evaluator runs parallel column/row gathers, cone reductions and a parallel tree reduction. CSR entry maps are compiled once during explicit setup without dropping structural zeros. Its optional scratch never borrows the iteration's products or changes iterates. An initial check precedes the first update, and cancellation takes precedence over certification. `evaluation_clock_cycles` is accumulated block-zero device clock cycles, not elapsed seconds. Existing solve-event time includes the evaluator; setup time includes policy validation and retained CSR construction.

The first runtime-branch build increased default register counts. It was superseded before any GPU test by separate kernel instantiations. `resources/` preserves raw cuobjdump output: default cooperative/single-block register counts are restored to 80/148, matching v603, while the opt-in kernels use 96/204. Matching resource counts alone do not prove runtime parity. The final core links every non-persistent object from immutable v603 unchanged; `core-build/manifest.json` records those object hashes and exact compile/link commands.

`core-build/` contains frozen v609d source and build logs; `replay-build/` contains v609e's sole replay-selector change linked to the identical core. `fixtures/`, `tiny/` and `summary.json` retain the complete successful batch. `earlier-builds/` retains both compile failures and the superseded CPU-only prototype; they performed zero GPU calls. The replay still uses its explicitly reported one-iteration unseeded bootstrap before a supplied seed, full reset/import and residual-only measurement. That bootstrap belongs to the replay harness and is counted separately from the seeded solve; the focused tiny executable requires no bootstrap.

No further GPU work or solver tuning is contained in this bundle. `sha256.json` indexes every file except itself.
'''
(destination/'README.md').write_text(readme)
index={path.relative_to(destination).as_posix():{'bytes':path.stat().st_size,'sha256':digest(path)} for path in sorted(destination.rglob('*')) if path.is_file()}
(destination/'sha256.json').write_text(json.dumps(index,indent=2))
print(json.dumps({'directory':str(destination),'files':len(index),'bytes':sum(x['bytes'] for x in index.values()),'index_sha256':digest(destination/'sha256.json')}))
