"""Freeze the owned Python changes and run CPU regressions without GPU loading."""
from pathlib import Path
import ctypes
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
BASE = '7eb8828f61bd35ec0abaf98c7651c5e82384ae27'
OUT = ROOT / 'build/performance/completion-adapter-cpu-v621a'
TESTS = ['test_gtoc12_completion_costs.py', 'test_gtoc12_gpu_completion.py',
         'test_gtoc12_chain.py', 'test_gtoc12_collectdp.py',
         'test_gtoc12_returns.py', 'test_gtoc12_harvestphase.py']
OWNED = ['src/spacepdhcg/gtoc12/search.py', 'src/spacepdhcg/gtoc12/gpu_completion.py',
         'src/spacepdhcg/gtoc12/gpu_lambert.py', 'tests/test_gtoc12_gpu_completion.py']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

if '--tests' in sys.argv:
    source = OUT / 'source'
    sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != '_editable_skbc_spacepdhcg']
    sys.path.insert(0, str(source / 'src'))
    os.environ['SPACEPDHCG_GTOC12_DATA'] = json.loads((ROOT / 'build/performance/incumbent-admission-v619/inputs/v616-profile.json').read_text())['data']
    os.environ['SPACEPDHCG_TEST_GTOC12_COMPLETION_BATCH'] = '0'
    os.environ['SPACEPDHCG_GTOC12_GPU_TESTS'] = '0'
    original = ctypes.CDLL
    def guarded(name, *args, **kwargs):
        if name != 'libc.so.6':
            raise AssertionError('CPU regressions forbid solver/GPU loading: ' + str(name))
        return original(name, *args, **kwargs)
    with patch.object(ctypes, 'CDLL', side_effect=guarded):
        import pytest
        import spacepdhcg.gtoc12.search as search
        assert Path(search.__file__).resolve() == source / 'src/spacepdhcg/gtoc12/search.py'
        raise SystemExit(pytest.main([*[str(source / 'tests' / t) for t in TESTS], '-q', '-p', 'no:cacheprovider']))

OUT.mkdir(exist_ok=False)
(OUT / 'source').mkdir()
(OUT / 'run.py').write_bytes(Path(__file__).read_bytes())
archived = subprocess.check_output(['git', 'archive', BASE, 'src', 'tests', 'pyproject.toml'], cwd=ROOT)
with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
    for member in archive.getmembers():
        target = OUT / 'source' / member.name
        assert target.resolve().is_relative_to((OUT / 'source').resolve())
        assert member.isfile() or member.isdir()
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
for name in OWNED:
    target = OUT / 'source' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / name).read_bytes())
report = {'complete': False, 'base_commit': BASE, 'gpu_calls': 0, 'owned_sources': {name: sha(ROOT/name) for name in OWNED},
          'source_sha256': {p.relative_to(OUT/'source').as_posix():sha(p) for p in sorted((OUT/'source').rglob('*')) if p.is_file()},
          'tests': TESTS, 'stages': []}
def save(): (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
save()
for name, command in [
    ('ruff', [sys.executable, '-B', '-m', 'ruff', 'check', '--no-cache', *[str(OUT/'source'/p) for p in OWNED]]),
    ('format', [sys.executable, '-B', '-m', 'ruff', 'format', '--check', '--no-cache', *[str(OUT/'source'/p) for p in OWNED]]),
    ('pytest', [sys.executable, '-B', str(__file__), '--tests']),
]:
    result = subprocess.run(command, cwd=OUT/'source', capture_output=True, text=True, timeout=180)
    path = OUT / (name + '.log')
    path.write_text(result.stdout + result.stderr)
    report['stages'].append({'name':name, 'command':command, 'exit_code':result.returncode, 'log_sha256':sha(path)})
    save()
    print(name, result.returncode, result.stdout[-4000:], result.stderr[-1000:], flush=True)
report['complete'] = all(row['exit_code'] == 0 for row in report['stages'])
save()
assert report['complete'], str(OUT/'report.json')
