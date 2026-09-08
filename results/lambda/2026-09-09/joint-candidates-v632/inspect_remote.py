"""Read-only inventory of the existing Lambda joint selection evidence."""

from pathlib import Path
import hashlib
import json

root = Path('/home/ubuntu/spacepdhcg-joint-selection-v632')
print('ROOT_ENTRIES', json.dumps([
    {'name': p.name, 'directory': p.is_dir(), 'bytes': p.stat().st_size if p.is_file() else None}
    for p in sorted(root.iterdir())
]))
for relative in ('report.json', 'validation-v631/report.json',
                 'benchmark-v634/report.json', 'campaign-v636/report.json'):
    path = root / relative
    if path.is_file():
        value = json.loads(path.read_text())
        print('REPORT', relative, json.dumps(value))
for directory in ('validation-v631', 'benchmark-v634', 'campaign-v636'):
    base = root / directory
    if not base.is_dir():
        continue
    entries = []
    for path in sorted(base.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.log', '.txt', '.xml'):
            entries.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size})
    print('FILES', directory, json.dumps(entries))
for relative in ('src/spacepdhcg/gtoc12/gpu_joint.py',
                 'src/spacepdhcg/gtoc12/jointopt.py',
                 'cpp/cuda/src/gtoc12_joint.cu',
                 'cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h',
                 'build/performance/orphan-recovery-v595/run.py'):
    path = root / 'repo' / relative
    if path.is_file():
        print('SOURCE_HASH', relative, hashlib.sha256(path.read_bytes()).hexdigest())
