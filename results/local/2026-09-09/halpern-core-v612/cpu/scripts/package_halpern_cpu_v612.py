"""Package immutable source/CPU/compiler evidence only; no GPU calls or binaries."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tarfile

live=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
destination=live/'build/performance/halpern-v612-cpu'
destination.mkdir(exist_ok=False)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
attempts={}
for suffix in ('a','b','c','d'):
    root=Path('/home/angus/spacepdhcg-halpern-v612'+suffix)
    target=destination/('v612'+suffix);target.mkdir()
    for path in root.iterdir():
        if path.is_file():shutil.copy2(path,target/path.name)
    shutil.copytree(root/'fixtures',target/'fixtures')
    manifest=json.loads((target/'manifest.json').read_text())
    with tarfile.open(target/'source.tar.gz') as archive:
        for name,expected in manifest['source_sha256'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==expected,name
    attempts[suffix]={'source_commit':manifest['frozen_commit'],'manifest_sha256':digest(target/'manifest.json'),
        'complete_build_script':manifest['complete'],'source_archive_sha256':digest(target/'source.tar.gz'),
        'gpu_executions':0,'resource_isolation_passed':suffix=='d'}
    if suffix=='a':
        attempts[suffix]['limitation']='All compilation, CPU conversion/parser and resource stages completed; configure launch raised FileNotFoundError because cmake was absent from PATH. Original empty configure log retained. No GPU was launched.'
    elif suffix in ('b','c'):
        attempts[suffix]['limitation']='CPU/parser/configure checks completed; cooperative default/common kernel resource footprints regressed to158 registers/40 stack bytes. Superseded before any GPU run.'
final=json.loads((destination/'v612d/manifest.json').read_text())
for name in final['owned_paths']:
    assert digest(live/name)==final['source_sha256'][name],('live/frozen mismatch',name)
support=destination/'scripts';support.mkdir()
for name in ('run_halpern_tiny_v612.py','package_halpern_cpu_v612.py'):
    shutil.copy2(live/'build/performance'/name,support/name)
shutil.copy2(live/'build/performance/HALPERN_V612_DESIGN.md',destination/'DESIGN.md')
baseline=Path('/home/angus/spacepdhcg-common-kkt-v609d/build/cuda/libspacepdhcg_cuda.so')
resource=subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(baseline)],text=True)
(destination/'baseline-v609d-resource-usage.log').write_text(resource)
patterns={'cooperative_default':r'cooperative_solve_kernelILb0E','cooperative_common':r'cooperative_solve_kernelILb1E',
          'single_default':r'12solve_kernelILb0E','single_common':r'12solve_kernelILb1E',
          'cooperative_scaling':r'cooperative_initialise_kernel','single_scaling':r'initialise_control_kernel',
          'cooperative_residual':r'cooperative_residual_kernel','single_residual':r'15residual_kernel',
          'recovery':r'15recovery_kernel'}
def selected(text):
    result={};lines=text.splitlines()
    for name,pattern in patterns.items():
        matches=[lines[i+1].strip() for i,line in enumerate(lines[:-1]) if 'Function' in line and re.search(pattern,line)]
        assert len(matches)==1,(name,matches)
        result[name]={k:int(v) for k,v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',matches[0])}
    return result
before=selected(resource);after=selected((destination/'v612d/resource-usage.log').read_text());assert before==after
(destination/'resource-comparison.json').write_text(json.dumps({'baseline_core_sha256':digest(baseline),
    'candidate_core_sha256':final['library_sha256'],'baseline':before,'candidate':after,
    'all_listed_default_resources_match':True,'new_halpern':{'REG':94,'STACK':0,'SHARED':7168},
    'scope':'Compiler footprints only; no runtime or convergence inference.'},indent=2))
(destination/'attempts.json').write_text(json.dumps(attempts,indent=2))
(destination/'README.md').write_text('''# Optional reflected Halpern: source and CPU evidence

The final build is **v612d**, frozen commit `6fbff324b152b75e6c27e3d91b4dc0835c65ab93`. It implements a diagnostic-only zero-Q primal-first Halpern path and an adaptive restart/weight variant, retaining the existing common original-equation accuracy gates. See [DESIGN.md](DESIGN.md) for the equations, scope and spectral precondition.

This package contains **no GPU solve results**. It records CPU conversion/oracle tests, six actual-input GPU-hidden JSON parser cases, four unsupported-input rejections, CUDA compilation, both CMake test registrations, source archives and compiler resource records. No binaries are included. The linked library reuses unchanged non-persistent object files identified in each manifest; this is not a rebuild or validation of every production backend.

The final core SHA256 is `49d0eef5702a7dee319ef3b7747ded844bec0f39313300405f9d6f30fc61730e`; replay `872ef9d6fe84e85aee2431e2d92caa1ad43bf2293ef9fc91bd9ee3ba938717f2`; tiny test `6da1ba381a566a23923da26d21cc9e72e9e6eb8159399c800350f6a79e73bed7`. [v612d/manifest.json](v612d/manifest.json) binds all compiled source and reused objects.

[resource-comparison.json](resource-comparison.json) verifies that existing listed solve, scaling, residual and recovery kernels retain their prior register/stack/shared-memory footprints. Cooperative default/common use80/96 registers with zero stack; single-block default/common use148/204 with40-byte stack. The new Halpern kernel uses94 registers and zero stack. Matching resources is not evidence of runtime parity.

The superseded a–c attempts are preserved. a completed compilation and CPU/parser checks but its configure launch could not locate CMake; b/c completed CPU/configure checks. All three exposed an unwanted158-register/40-byte-stack footprint in the old cooperative kernels and were superseded before GPU execution. The final narrow forced-inline report helper restores those resources. See [attempts.json](attempts.json).

The reviewed [tiny runner](scripts/run_halpern_tiny_v612.py) is prepared for two executions under the shared GPU lock: plain/adaptive, seven solve API calls each, at most416 requested iterations total and410 expected actual optimization iterations. It has no bootstrap, no automatic reruns and no GPU launch in this package. The parent run will preserve its own fresh logs and report separately. Tiny success would validate the bounded fixtures, not cold convergence, throughput, fleet score or SOTA.

The flat [sha256.json](sha256.json) index binds every packaged file except itself. Source archives are checked against their per-file manifest hashes; all nine final owned live files matched the frozen archive at packaging.
''')
index={p.relative_to(destination).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(destination.rglob('*')) if p.is_file()}
(destination/'sha256.json').write_text(json.dumps(index,indent=2,sort_keys=True))
print(json.dumps({'destination':str(destination),'files':len(index),'bytes':sum(v['bytes'] for v in index.values()),'index_sha256':digest(destination/'sha256.json')}))
