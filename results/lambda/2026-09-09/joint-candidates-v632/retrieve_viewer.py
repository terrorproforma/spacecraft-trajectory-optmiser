"""Retrieve only the already generated candidate0 viewer; no remote computation."""

from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile

destination = Path('build/performance/retrieved-joint-v632')
key = Path('/tmp/traj-key.pem')
if not key.exists():
    descriptor = os.open(str(key), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as output:
        output.write(Path('traj-key.pem').read_bytes())
remote = r'''
from pathlib import Path
import datetime, hashlib, io, json, sys, tarfile
root=Path('/home/ubuntu/spacepdhcg-joint-selection-v632')
names=('campaign-v636/candidate0/best/viewer/trajectories.json',
       'campaign-v636/candidate0/best/viewer/manifest.json')
metadata={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope':'Read existing viewer files only; no remote writes or trajectory computation.',
          'report_sha256':hashlib.sha256((root/'campaign-v636/candidate0/report.json').read_bytes()).hexdigest(),
          'files':{}}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for name in names:
        path=root/name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
        data=path.read_bytes()
        metadata['files'][name]={'source':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    data=(json.dumps(metadata,indent=2)+'\n').encode()
    info=tarfile.TarInfo('viewer-download-manifest.json');info.size=len(data)
    archive.addfile(info,io.BytesIO(data))
'''
archive_path=destination/'viewer-evidence.tar.gz'
if archive_path.exists():
    assert archive_path.stat().st_size == 0, 'refuse to replace an existing download'
with archive_path.open('wb' if archive_path.exists() else 'xb') as output:
    result=subprocess.run(['ssh','-i','/tmp/traj-key.pem','-o','BatchMode=yes','-o',
        'ConnectTimeout=15','ubuntu@192.222.55.229','python3 -'],input=remote.encode(),
        stdout=output,stderr=subprocess.PIPE,timeout=55)
assert result.returncode==0,result.stderr.decode()
evidence=destination/'evidence'
with tarfile.open(archive_path) as archive:
    for member in archive.getmembers():
        target=evidence/member.name
        assert member.isfile() and target.resolve().is_relative_to(evidence.resolve()) and not target.exists()
    archive.extractall(evidence,filter='data')
metadata=json.loads((evidence/'viewer-download-manifest.json').read_text())
for relative,item in metadata['files'].items():
    data=(evidence/relative).read_bytes()
    assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
report_path=evidence/'campaign-v636/candidate0/report.json'
assert hashlib.sha256(report_path.read_bytes()).hexdigest()==metadata['report_sha256']
report=json.loads(report_path.read_text())
viewer=evidence/'campaign-v636/candidate0/best/viewer'
manifest=json.loads((viewer/'manifest.json').read_text())
assert manifest==report['best']['viewer']
trajectory=(viewer/'trajectories.json').read_bytes()
assert len(trajectory)==manifest['files']['trajectories.json']['bytes']
assert hashlib.sha256(trajectory).hexdigest()==manifest['files']['trajectories.json']['sha256']
solution=(viewer.parent/'Result.txt').read_bytes()
assert len(solution)==manifest['source']['bytes']
assert hashlib.sha256(solution).hexdigest()==manifest['source']['sha256']
summary={'verified_files':len(metadata['files']),'snapshot_utc':metadata['snapshot_utc'],
    'archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    'archive_bytes':archive_path.stat().st_size,'report_unchanged':True,
    'manifest_equals_recorded_report':True,'viewer_source_matches_downloaded_result':True,
    'trajectories_sha256':hashlib.sha256(trajectory).hexdigest(),
    'scope':metadata['scope']}
(destination/'viewer-download-verified.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
