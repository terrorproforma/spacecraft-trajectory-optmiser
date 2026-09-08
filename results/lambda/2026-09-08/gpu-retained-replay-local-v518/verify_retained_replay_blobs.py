from pathlib import Path
import hashlib
import io
import json
import subprocess
import tarfile

def blob(path):
    return subprocess.check_output(['git', 'show', ':' + path.as_posix()])

for tag in ['retained-replay-local-v518', 'retained-replay-v514', 'retained-replay-legs-v517']:
    root = Path('results/lambda/2026-09-08/gpu-' + tag)
    manifest = json.loads(blob(root / 'files-sha256.json'))
    for name, sha in manifest.items():
        assert hashlib.sha256(blob(root / name)).hexdigest() == sha, (tag, name)
    if (root / 'archive-manifests.json').exists():
        archives = json.loads(blob(root / 'archive-manifests.json'))
    else:
        archives = {'raw.tar.gz': json.loads(blob(root / 'archive-manifest.json'))}
    for name, members in archives.items():
        with tarfile.open(fileobj=io.BytesIO(blob(root / name))) as archive:
            actual = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                      for m in archive.getmembers() if m.isfile()}
        if name == 'raw.tar.gz':
            actual.pop('files-sha256.json')
        assert actual == members, (tag, name)
    print(tag, len(manifest), 'files and archive members verified')

report = json.loads(Path('build/performance/retained-replay-v515/report.json').read_text())
for name, sha in report['source_sha256'].items():
    data = Path(name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha, name
    assert blob(Path(name)).replace(b'\r\n', b'\n') == data.replace(b'\r\n', b'\n'), name
print('Tested source matches staged Git blobs')
