"""Freeze only owned completion changes; CPU checks and CUDA compile, no GPU execution."""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

LIVE = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
ROOT = Path('/home/angus/spacepdhcg-completion-model-v622c')
OUT = LIVE / 'build/performance/completion-model-v622c'
BASE = 'fdf52ae31d259240ffebdcf79ed0965dce8b9298'
OWNED = ['cpp/cuda/include/spacepdhcg/cuda/gtoc12_completion_c_api.h',
         'cpp/cuda/src/gtoc12_completion.cu', 'cpp/cuda/src/gtoc12_completion_model.cuh',
         'src/spacepdhcg/gtoc12/gpu_completion.py', 'src/spacepdhcg/gtoc12/gpu_completion_model.py',
         'src/spacepdhcg/gtoc12/completion_capture.py', 'src/spacepdhcg/gtoc12/refinement_admission.py',
         'tests/test_gtoc12_gpu_completion_model.py']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

if '--cpu' in sys.argv:
    sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != '_editable_skbc_spacepdhcg']
    sys.path.insert(0, str(ROOT / 'source/src'))
    import spacepdhcg.gtoc12.gpu_completion_model as module
    assert Path(module.__file__).resolve() == ROOT / 'source/src/spacepdhcg/gtoc12/gpu_completion_model.py'
    import pytest
    raise SystemExit(pytest.main([str(ROOT/'source/tests/test_gtoc12_gpu_completion_model.py'),
        str(ROOT/'source/tests/test_gtoc12_gpu_completion.py'),
        str(ROOT/'source/tests/test_gtoc12_completion_costs.py'), '-q', '-p', 'no:cacheprovider']))

ROOT.mkdir(exist_ok=False)
OUT.mkdir(exist_ok=False)
source = ROOT / 'source'
source.mkdir()
archive = subprocess.check_output(['git', 'archive', BASE, 'src', 'tests', 'pyproject.toml'], cwd=LIVE)
with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
    for member in stream.getmembers():
        target = source / member.name
        assert target.resolve().is_relative_to(source.resolve())
        assert member.isfile() or member.isdir()
        if member.isdir(): target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stream.extractfile(member).read())
for name in OWNED:
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((LIVE / name).read_bytes())
shutil.copy2(__file__, ROOT / 'prepare.py')
report = {'base': BASE, 'complete': False, 'gpu_calls': 0,
          'owned_sources': {name:sha(source/name) for name in OWNED}, 'stages': []}
env = {k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'LD_LIBRARY_PATH'))}
env.update(CUDA_VISIBLE_DEVICES='', SPACEPDHCG_GTOC12_GPU_TESTS='0', PYTHONDONTWRITEBYTECODE='1')

def save():
    (ROOT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.json', '.log', '.py', '.gz'):
            shutil.copy2(p, OUT/p.name)

def run(name, command, timeout=180):
    start = time.perf_counter()
    with (ROOT/(name+'.log')).open('x') as log:
        result = subprocess.run(command, env=env, cwd=source, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    report['stages'].append({'name':name, 'command':command, 'exit_code':result.returncode,
                            'seconds':time.perf_counter()-start, 'log_sha256':sha(ROOT/(name+'.log'))})
    save()
    print(name, result.returncode, flush=True)
    if result.returncode:
        print((ROOT/(name+'.log')).read_text()[-7000:], flush=True)
        raise RuntimeError(name)

save()
python = '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
py = [str(source/x) for x in OWNED if x.endswith('.py')]
run('ruff', [python, '-m', 'ruff', 'check', '--no-cache', *py])
run('format', [python, '-m', 'ruff', 'format', '--check', '--no-cache', *py])
run('cpu-tests', [python, '-B', str(ROOT/'prepare.py'), '--cpu'])
nvcc = '/usr/local/cuda-12.8/bin/nvcc'
run('compiler-version', [nvcc, '--version'])
library = ROOT/'libspacepdhcg_completion.so'
run('build', [nvcc, '-std=c++20', '-O3', '--fmad=false', '-shared', '-Xcompiler=-fPIC',
             '-gencode=arch=compute_90,code=sm_90', '-gencode=arch=compute_120,code=sm_120',
             '-I'+str(source/'cpp/cuda/include'), str(source/'cpp/cuda/src/gtoc12_completion.cu'),
             '-o', str(library)])
run('resource-usage', ['/usr/local/cuda-12.8/bin/cuobjdump', '--dump-resource-usage', str(library)])
with tarfile.open(ROOT/'source.tar.gz', 'w:gz') as archive:
    for name in OWNED: archive.add(source/name, arcname=name, recursive=False)
report['library'] = {'path':str(library), 'sha256':sha(library), 'bytes':library.stat().st_size}
report['source_archive_sha256'] = sha(ROOT/'source.tar.gz')
report['complete'] = True
save()
print(json.dumps({'complete':True, 'report_sha256':sha(ROOT/'report.json')}))
