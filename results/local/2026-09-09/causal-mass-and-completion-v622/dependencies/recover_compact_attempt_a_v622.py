"""Recover exact source bytes from the preserved pre-build failure, CPU only."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
report_path = ROOT / 'build/performance/completion-model-v622a/report.json'
report = json.loads(report_path.read_text())
assert report['complete'] is False and report['gpu_calls'] == 0
frozen = Path('/home/angus/spacepdhcg-completion-model-v622a/source')
out = ROOT / 'build/performance/completion-model-v622a-source-recovery'
out.mkdir(exist_ok=False)
files = {}
for name, expected in report['owned_sources'].items():
    data = (frozen / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, name
    destination = out / 'source' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    files[name] = {'bytes': len(data), 'sha256': expected}
manifest = {'passed': True, 'GPU_calls': 0, 'native_loads': 0,
            'original_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
            'scope': 'Exact source bytes recovered from the preserved frozen a workspace after its pre-build failure; no source correction or rerun.',
            'files': files}
(out / 'report.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'passed': True, 'files': len(files), 'GPU_calls': 0}))
