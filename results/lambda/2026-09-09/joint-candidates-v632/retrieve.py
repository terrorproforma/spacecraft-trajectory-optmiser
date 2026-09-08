"""Download only existing, allowlisted Lambda evidence; perform no remote writes."""

from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

destination = Path('build/performance/retrieved-joint-v632')
remote = r'''
from pathlib import Path
import datetime, hashlib, io, json, sys, tarfile
root = Path('/home/ubuntu/spacepdhcg-joint-selection-v632')
selected = set()
for relative in ('report.json', 'source-manifest.json', 'compatibility-source.json',
                 'campaign-fixtures.json', 'configure.log', 'build.log'):
    if (root/relative).is_file(): selected.add(relative)
for directory in ('validation-v631', 'compatibility-validation-v640', 'benchmark-v634'):
    base = root/directory
    if base.is_dir():
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in ('.json', '.log', '.xml'):
                selected.add(str(p.relative_to(root)))
selected.add('campaign-v636/report.json')
for name in ('baseline0', 'candidate0', 'candidate1', 'baseline1'):
    base = root/'campaign-v636'/name
    for p in base.rglob('*.json'):
        if 'viewer' not in p.relative_to(base).parts:
            selected.add(str(p.relative_to(root)))
    selected.add('campaign-v636/'+name+'.log')
selected.add('campaign-v636/candidate0/best/Result.txt')
provenance = {'source_hashes': {}, 'best_result_hashes': {}, 'runner_configuration_lines': {}}
for relative in ('src/spacepdhcg/gtoc12/gpu_joint.py',
                 'src/spacepdhcg/gtoc12/jointopt.py',
                 'cpp/cuda/src/gtoc12_joint.cu',
                 'cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h',
                 'cpp/cuda/CMakeLists.txt',
                 'build/performance/orphan-recovery-v595/run.py'):
    p = root/'repo'/relative
    provenance['source_hashes'][relative] = hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('baseline0', 'candidate0', 'candidate1', 'baseline1'):
    p = root/'campaign-v636'/name/'best/Result.txt'
    provenance['best_result_hashes'][name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
for relative in ('campaign-runner.py', 'benchmark-runner.py', 'check-compatibility.py'):
    p = root/relative
    if p.is_file():
        provenance['runner_configuration_lines'][relative] = [
            {'line': number, 'text': line}
            for number, line in enumerate(p.read_text().splitlines(), 1)
            if any(word in line for word in ('SPACEPDHCG_', 'QOCO_', "python=", "qoco=", 'command=', 'env=', 'env =', "'--", 'source='))]
final = root/'final-python-v640'
provenance['final_python_hashes'] = {}
for relative in ('src/spacepdhcg/gtoc12/gpu_joint.py', 'tests/test_gtoc12_gpu_joint_compatibility.py'):
    p=final/relative
    if p.is_file(): provenance['final_python_hashes'][relative]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'snapshot_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'remote_root': str(root),
            'scope': 'Existing v632 build, validation, component benchmarks, v636 campaign reports and candidate0 best Result; read-only SSH retrieval. No remote jobs, writes, source or test uploads.',
            'files': {}}
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    def add(name, data, origin):
        item=tarfile.TarInfo(name); item.size=len(data)
        archive.addfile(item, io.BytesIO(data))
        manifest['files'][name]={'source': origin, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    for relative in sorted(selected):
        p=root/relative
        assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()),relative
        add(relative,p.read_bytes(),str(p))
    add('remote-provenance.json',(json.dumps(provenance,indent=2)+'\n').encode(),'Read-only selected source/result hashes and configuration excerpts')
    data=(json.dumps(manifest,indent=2)+'\n').encode()
    item=tarfile.TarInfo('download-manifest.json');item.size=len(data)
    archive.addfile(item,io.BytesIO(data))
'''

archive_path = destination/'lambda-evidence.tar.gz'
with archive_path.open('xb') as output:
    result = subprocess.run(
        ['ssh', '-i', '/tmp/traj-key.pem', '-o', 'BatchMode=yes', '-o',
         'ConnectTimeout=15', 'ubuntu@192.222.55.229', 'python3 -'],
        input=remote.encode(), stdout=output, stderr=subprocess.PIPE, timeout=55)
assert result.returncode == 0, result.stderr.decode()
evidence = destination/'evidence'
evidence.mkdir(exist_ok=False)
with tarfile.open(archive_path) as archive:
    for member in archive.getmembers():
        assert member.isfile() and (evidence/member.name).resolve().is_relative_to(evidence.resolve())
    archive.extractall(evidence, filter='data')
manifest = json.loads((evidence/'download-manifest.json').read_text())
for name, metadata in manifest['files'].items():
    data=(evidence/name).read_bytes()
    assert len(data)==metadata['bytes']
    assert hashlib.sha256(data).hexdigest()==metadata['sha256'],name
summary={'verified_files':len(manifest['files']), 'snapshot_utc':manifest['snapshot_utc'],
         'archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
         'archive_bytes':archive_path.stat().st_size, 'scope':manifest['scope']}
(destination/'download-verified.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
