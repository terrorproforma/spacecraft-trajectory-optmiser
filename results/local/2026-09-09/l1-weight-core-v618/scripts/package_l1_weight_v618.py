"""Package completed v618 evidence only; no GPU, Git, network or source edits."""
from pathlib import Path
import gzip,hashlib,json,shutil,tarfile
root=Path(__file__).resolve().parents[2]
destination=root/'results/local/2026-09-09/l1-weight-core-v618'
destination.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
excluded=[]
def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
def copy_tree(source,target):
    for p in sorted(source.rglob('*')):
        if not p.is_file():continue
        with p.open('rb') as f:magic=f.read(4)
        if '__pycache__' in p.parts or p.suffix in ('.so','.o','.a','.dll','.exe','.pyc','.pdb') or magic==b'\x7fELF' or magic[:2]==b'MZ':
            excluded.append({'source':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p),'reason':'compiled artifact or cache excluded'})
            continue
        copy_file(p,target/p.relative_to(source))
for version in ('a','b'):
    source=root/f'build/performance/l1-weight-v618{version}'
    copy_tree(source,destination/'builds'/version)
    m=json.loads((source/'manifest.json').read_text())
    with tarfile.open(source/'source.tar.gz','r:gz') as archive:
        assert {x.name for x in archive.getmembers() if x.isfile()}==set(m['source_sha256'])
        for member in archive.getmembers():
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            data=archive.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest()==m['source_sha256'][member.name]
            assert data[:4]!=b'\x7fELF' and data[:2]!=b'MZ'
        tree=''.join(k+':'+v+'\n' for k,v in m['source_sha256'].items())
        assert hashlib.sha256(tree.encode()).hexdigest()==m['source_tree_sha256']
manifest=json.loads((destination/'builds/b/manifest.json').read_text())
assert manifest['complete'] and manifest['frozen_commit'] is None
with tarfile.open(destination/'builds/b/source.tar.gz','r:gz') as archive:
    for name in manifest['owned_paths']:
        data=archive.extractfile(name).read();assert hashlib.sha256(data).hexdigest()==manifest['source_sha256'][name]==sha(root/name)
        path=destination/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
copy_tree(root/'build/performance/l1-weight-tiny-v618',destination/'tiny')
for name in ('l1-weight-cpu-v618','l1-weight-cpu-v618b','l1-weight-runner-cpu-v618'):
    copy_tree(root/'build/performance'/name,destination/'preparation'/name)
copy_tree(root/'build/performance/l1-weight-review-v618',destination/'analysis')
real=root/'build/performance/l1-weight-real-v618'
report=json.loads((real/'run/report.json').read_text());assert report['complete'] and len(report['cases'])==6
fresh=json.loads((real/'audit/findings.json').read_text());assert fresh['complete'] and fresh['report_sha256']==sha(real/'run/report.json')
for name in ('manifest.json','run.py','independent_auditor.py','tiny-report.json'):copy_file(real/name,destination/'real'/name)
copy_tree(real/'inputs',destination/'real/inputs');copy_tree(real/'audit',destination/'real/audit')
copy_file(real/'run/report.json',destination/'real/report.json')
members={}
with (destination/'real/raw-logs.tar.gz').open('xb') as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as archive:
            for case in report['cases']:
                name=case['name']+'.log';assert Path(name).name==name
                p=real/'run'/name;assert sha(p)==case['log_sha256']
                members[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
                info=tarfile.TarInfo(name);info.size=p.stat().st_size;info.mode=0o644
                with p.open('rb') as file:archive.addfile(info,file)
(destination/'real/raw-members.json').write_text(json.dumps(members,indent=2))
for name in ('build_l1_weight_v618a.py','build_l1_weight_v618b.py','prepare_l1_weight_b_v618.py','resume_l1_weight_v618b.py',
    'check_l1_weight_cpu_v618.py','check_l1_weight_cpu_v618b.py','run_l1_weight_tiny_v618.py',
    'prepare_l1_weight_real_v618.py','run_l1_weight_real_v618.py','check_l1_weight_runners_v618.py',
    'audit_l1_weight_real_v618.py','package_l1_weight_v618.py'):
    copy_file(root/'build/performance'/name,destination/'scripts'/name)
copy_file(root/'build/performance/recheck_l1_weight_package_v618.py',destination/'scripts/recheck_saved_outputs.py')
(destination/'excluded-artifacts.json').write_text(json.dumps(excluded,indent=2))
table='\n'.join('| '+c['capture'].capitalize()+' | '+c['mode']+' | '+f"{c['final']['solve_seconds']:.7f}"+' | '+f"{c['audit']['gap']:.10g}"+' | '+('Yes' if c['audit']['qualified'] else 'No')+' |' for c in report['cases'] if not c['seeded'])
assert fresh['qualified_seed_cases']==2 and fresh['qualified_cold_cases']==0
notes=f'''# Reciprocal L1 weight: no qualified cold solution

Both original supplied points remain qualified at zero optimization iterations.
All four cold runs reach 100,000 iterations and fail the unchanged original
accuracy gate. This experiment remains default off. It establishes no qualified
solution speedup, fleet-score change, SOTA claim or production GTOC12 integration.

| Capture | Weight policy | Native solve event seconds | Relative gap | Qualified |
|---|---|---:|---:|---|
{table}

These are four single fixed-budget measurements on one RTX 5090, using the same
frozen binary/core, explicit 128-block grid and original common-KKT gates. All
finish inside their 30-second cold deadlines. There is no retry, recovery,
cancellation or numerical failure in this real batch. Solve-event timing
excludes scaling, importer setup, diagnostic downloads and independent CPU
audit; those scopes are separately recorded. A smaller gap alone is not full
qualification. The conditioning gap improves and the difficult gap worsens.

The [tiny report](tiny/report.json) records two processes, 22 solve APIs and
22 optimization updates. Unit and fixed .25/4/cancel-global scalar/SOC/prox
oracles, exact mode-switch versus fresh-control behavior, untouched weak seeds,
cancellation and nonfinite/effective-step failure checks pass. These vector
oracles execute inside the tiny test; its raw logs export metrics, not all
vectors, so they are not independent raw-vector replay evidence.

The [real report](real/report.json) records six executions/eight solve APIs,
400,000 cold updates and two separately counted one-step seed bootstraps.
After each bootstrap, a full reset, explicit original primal/dual seed and
residual-only check precede the zero-step accepted solve. Supplied original
x/y/z FP64 bits are preserved. Across tiny and real: 30 solve APIs and 400,024
actual updates. None of the failed preparation attempts below ran GPU solvers.

The explicit fixed mode uses a caller-provided positive omega. The separate
`cancel-global` mode chooses omega=O/B on-device after reduced scaling, once per
solve. It reads coefficients only, with no reference solution, pilot solve,
adaptive update or parameter sweep. Native-scaled base steps are eta/omega and
eta*omega; original diagonal factors then give mathematical eta/D² and eta/R²
for cancel-global. Lambda uses the actual original-coordinate primal step.
The same reduced D/R/B/O and eta heuristic are retained. The mathematical step
product is unchanged; the 20-power-step norm estimate is still only a heuristic.

Original coefficients, objective, dual completion and common gates are intact:
relative 1e-9 for primal equations, stationarity, objective gap and maximum
scalar/SOC block complementarity; absolute 1e-8 for cone membership. Natural
residual telemetry remains a separate legacy unweighted diagnostic. Original
seeds are checked before any completion. Weight-record validity does not imply
KKT qualification. Invalid weights/steps fail without silently changing them;
cancellation wins before acceptance. Each fresh L1 enable restores unit policy.

The exact L1 maps remain 1470/1631 pairs, logical dimensions 3791/7811 and 4208/8666
variables/rows. Full original 5261/10751 and 5839/11928 layouts remain allocated.
The experiment does not compact memory or change the production trajectory
backend. The [coefficient-only derivation](analysis/MATH.md) and its reproducible
checks document the policy and its mathematical limits.

Existing default/common cooperative resources remain 80/96 registers and zero
stack; single-block variants remain 148/204 registers and 40-byte stack. Halpern
remains 96 registers. Unit and nonunit L1 templates now use 94 registers and zero
stack, versus the prior L1 kernel's 198 registers and 40-byte stack. The initializer
remains 58 registers. This compiler/resource change prevents a claim of byte or
runtime parity with the old build; this comparison uses the same new core in
both arms. The native setter checks occupancy without silently reducing 128 blocks.

The tested source is an uncommitted frozen tree, not a synthetic Git commit:
`{manifest['source_tree_sha256']}`. The assembly base is v615c commit
`{manifest['base_commit']}`; workspace parent was 9c8f2e96. Later fleet changes
are separate work. Every archived source member and all ten [owned source
files](source/cpp/cuda/) match the retained manifest. Replay metadata reports
`source_commit=uncommitted`, explicit source scope, tree SHA and base identity.
Core SHA: `{manifest['library_sha256']}`.
Manifest SHA: `{sha(real/'manifest.json')}`.

The first manual CPU compilation failed from a local test variable-name collision
and was corrected. Build a compiles and passes parser checks but CMake's source
Git lookup fails for its Git-free archive. Its `git_operations:0` field intended
zero mutations; read-only CMake identity queries did occur. Build b adds explicit
frozen-tree source provenance without bypassing pinned third-party checks and
reuses the exact a core. A b wrapper assertion then mistook expected parser
rejections for failures; its pre-continuation manifest and exact continuation
are preserved. These are preparation failures, not GPU solver attempts.

All six complete raw outputs, original x/y/z/slack vectors and metadata are
losslessly stored in [raw-logs.tar.gz](real/raw-logs.tar.gz), with per-member
hashes in [raw-members.json](real/raw-members.json). Inputs, exact runner and
pinned auditor are retained. The fresh [original-equation re-audit](real/audit/findings.json)
agrees with every runtime result. No compiled binaries, caches or keys are
included; excluded local CPU binaries are identified by hash.

After the independent Decimal65 review is copied, the package-local
`scripts/recheck_saved_outputs.py --output /absolute/path/new-audit.json` can
recheck the compressed outputs on CPU with NumPy/SciPy. No solver/CUDA loading
is performed. The final flat SHA256 index covers every retained file.
'''
(destination/'README.md').write_text(notes)
(destination/'.gitattributes').write_text('* -text\n')
print(json.dumps({'package':str(destination),'status':'prepared_pending_stable_Decimal_review_and_index'}))
