from pathlib import Path
import ast
import hashlib
import io
import json
import subprocess
import sys
import tarfile

ref = sys.argv[1] if len(sys.argv) > 1 else ''
tags = ['resident-options-local-v506', 'resident-options-v497',
        'resident-options-v500', 'lagrange-repeat-v501',
        'lagrange-precision-v503', 'resident-options-v507']
def blob(path):
    return subprocess.check_output(['git', 'show', ref + ':' + str(path)])

for tag in tags:
    root = Path('results/lambda/2026-09-08/gpu-' + tag)
    manifest = json.loads(blob(root / 'files-sha256.json'))
    for name, sha in manifest.items():
        data = blob(root / name)
        assert hashlib.sha256(data).hexdigest() == sha, (tag, name)
    archive_manifest_path = root / 'archive-manifests.json'
    if archive_manifest_path.exists():
        for name, members in json.loads(blob(archive_manifest_path)).items():
            with tarfile.open(fileobj=io.BytesIO(blob(root / name))) as archive:
                actual = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                          for m in archive.getmembers() if m.isfile()}
            assert actual == members, (tag, name)
    print(tag, len(manifest), 'Git blobs verified', flush=True)

report = json.loads(Path('build/performance/resident-options-v506/report.json').read_text())
for name, sha in report['source_sha256'].items():
    data = Path(name).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        assert name == 'src/spacepdhcg/gtoc12/lambert.py', name
        frozen = Path('build/performance/resident-options-v506/source') / name
        old = frozen.read_bytes()
        assert hashlib.sha256(old).hexdigest() == sha
        assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(data))
        print('Verified formatting-only lambert.py line wrap', flush=True)
    assert blob(name).replace(b'\r\n', b'\n') == data.replace(b'\r\n', b'\n'), name
print('Validated source matches staged/committed code', flush=True)
