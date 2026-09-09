"""Verify sealed display evidence bytes without executing archived code."""
from pathlib import Path
import hashlib
import json
import zipfile

if not __debug__:
    raise RuntimeError('Assertions must remain enabled')
root = Path(__file__).resolve().parent
payload = (root/'evidence.zip').read_bytes()
assert hashlib.sha256(payload).hexdigest() == 'c63d9c5ab2ee15e1cbdbb7a328e05c306d89731a81d98b94fbdafc51d4cb71c5'
with zipfile.ZipFile(root/'evidence.zip') as archive:
    raw = archive.read('index.json')
    assert hashlib.sha256(raw).hexdigest() == '769fdc01be4c118d61b06e97294996a135e25681df17c931ea44b9fbc4af23ea'
    index = json.loads(raw)
    assert set(archive.namelist()) == set(index['files']) | {'index.json'}
    assert len(archive.namelist()) == len(index['files']) + 1
    for name, item in index['files'].items():
        data = archive.read(name)
        assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], name
print(json.dumps({'passed': True, 'indexed_files': len(index['files']), 'archived_code_executed': False, 'GPU_calls': 0}))
