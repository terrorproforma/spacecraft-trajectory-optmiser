"""Pin the four-case adapter runner and run collection-only checks with CUDA hidden."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / 'build/performance/completion-adapter-gpu-v621'
HOST = ROOT / 'build/performance/completion-adapter-cpu-v621b'
NATIVE = ROOT / 'build/performance/completion-fullcore-v621'
CONTROLS = ROOT / 'build/performance/completion-native-controls-v621'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
assert sha(HOST / 'report.json') == '2e7175e8f818d11987fc240b5511c54b498f5485798501703c03e0906359bc8a'
assert sha(NATIVE / 'manifest.json') == '43c3c2974af7a9d16ba060724855a56b5122acf9c53ff29d68243df612d82969'
assert sha(CONTROLS / 'gpu-output/report.json') == '3b0dddee118b28203dbcb136b731c5ab896cf2b6a7dd11378683ada0cff30e99'
host, native, prior = read(HOST / 'report.json'), read(NATIVE / 'manifest.json'), read(CONTROLS / 'gpu-profile.json')
assert host['complete'] and native['complete']
for name, digest in host['source_sha256'].items():
    assert sha(HOST / 'source' / name) == digest
tree = ''.join(k + ':' + v + '\n' for k, v in sorted(host['source_sha256'].items()))
profile = {
    'host_report_path': str(HOST / 'report.json'), 'host_report_sha256': sha(HOST / 'report.json'),
    'host_source_root': str(HOST / 'source'),
    'host_source_tree_sha256': hashlib.sha256(tree.encode()).hexdigest(),
    'host_source_scope': 'Frozen7eb8828f Python base plus four owned adapter/test changes from CPU attemptb; separate from native source.',
    'native_manifest_path': '/home/angus/spacepdhcg-completion-fullcore-v621/manifest.json',
    'native_manifest_sha256': sha(NATIVE / 'manifest.json'),
    'native_source_root': '/home/angus/spacepdhcg-completion-fullcore-v621/repo',
    'library': native['library'], 'python': sys.executable, 'python_sha256': sha(sys.executable),
    'native_controls_report_path': str(CONTROLS / 'gpu-output/report.json'),
    'native_controls_report_sha256': sha(CONTROLS / 'gpu-output/report.json'),
    'gpu_uuid': prior['gpu_uuid'], 'gpu_uuid_provenance': prior['gpu_uuid_provenance'],
    'gpu_lock': prior['gpu_lock'], 'nvidia_smi': prior['nvidia_smi'],
    'ld_library_path': prior['ld_library_path'],
    'test_suffixes': ['test_native_completion_matches_scalar_for_ragged_batches_and_workspace_reuse[' + m + ']'
                      for m in ('fit', 'flat', 'ratio', 'certified')],
    'budget': {'pytest_cases': 4, 'evaluation_calls': 8, 'candidates': 1048,
               'lambert_requests': 0, 'child_processes': 1, 'timeout_seconds': 180, 'retries': 0},
    'observation_scope': 'Wrap native evaluation boundary for exact call counts and raw FP64 readbacks. No numerical/model/gate changes. Instrumented timings are not benchmark data.',
}
with (KIT / 'profile.json').open('x') as stream:
    json.dump(profile, stream, indent=2, allow_nan=False)
    stream.write('\n')
(KIT / 'prepare.py').write_bytes(Path(__file__).read_bytes())
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'PYTEST_', 'LD_LIBRARY_PATH'))
       and k not in ('PYTHONPATH', 'PYTHONOPTIMIZE')}
env.update({'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONOPTIMIZE': '0',
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
            'SPACEPDHCG_GTOC12_CUDA_LIBRARY': native['library']['path'],
            'SPACEPDHCG_GTOC12_GPU_TESTS': '1'})
for name in ('run.py', 'pytest_child.py'):
    compile((KIT / name).read_text(), str(KIT / name), 'exec')
command = [sys.executable, '-B', str(KIT / 'pytest_child.py'), '--collect-only', '--output', str(KIT / 'cpu-collection')]
with (KIT / 'cpu-collection.log').open('x') as log:
    result = subprocess.run(command, env=env, cwd=HOST / 'source', stdout=log, stderr=subprocess.STDOUT, timeout=120)
assert result.returncode == 0, (KIT / 'cpu-collection.log').read_text()
validation = read(KIT / 'cpu-collection/report.json')
assert validation['passed'] and validation['collect_only'] and validation['actual_evaluation_calls'] == 0
assert len(validation['collected_tests']) == 4
files = ['run.py', 'pytest_child.py', 'profile.json', 'prepare.py', 'cpu-collection.log', 'cpu-collection/report.json']
ready = {'complete': True, 'cpu_collect_only_passed': True, 'gpu_calls': 0,
         'files': {p: sha(KIT / p) for p in files}, 'validation_command': command,
         'launch_command': [sys.executable, '-B', str(KIT / 'run.py')],
         'source_scope': profile['host_source_scope']}
with (KIT / 'ready.json').open('x') as stream:
    json.dump(ready, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'ready_sha256': sha(KIT / 'ready.json'), 'runner_sha256': sha(KIT / 'run.py'),
                  'child_sha256': sha(KIT / 'pytest_child.py'), 'host_source_tree_sha256': profile['host_source_tree_sha256']}))
