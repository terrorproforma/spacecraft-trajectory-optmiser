"""Independently verify completed evidence and assemble the captured-QP baseline."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import subprocess
import tarfile

workspace=Path(__file__).resolve().parents[2]
def sha(data): return hashlib.sha256(data).hexdigest()
def verify_index(root):
    index=json.loads((root/'sha256.json').read_text(encoding='utf-8-sig'))
    for name, expected in index.items():
        data=(root/name).read_bytes()
        assert sha(data)==expected['sha256'] and len(data)==expected['bytes'], name
    return len(index)
def members(path):
    result={}
    with tarfile.open(path) as archive:
        for entry in archive.getmembers():
            name=entry.name.removeprefix('./')
            assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
            if entry.isdir(): continue
            assert entry.isfile() and name not in result, name
            result[name]=sha(archive.extractfile(entry).read())
    return result

substitution=workspace/'results/local/2026-09-09/asteroid-substitution-v599'
sub_files=verify_index(substitution)
ready=json.loads((substitution/'execution/ready-manifest.json').read_text())['files']
sub_sources=members(substitution/'source.tar.gz')
for name,digest in sub_sources.items():
    assert name.startswith('execution/') and ready[name.removeprefix('execution/')]==digest, name
sub_outputs=members(substitution/'raw.tar.gz')
for name,digest in sub_outputs.items():
    assert name.startswith('output/')
    assert sha((workspace/'build/performance/asteroid-substitution-v599'/name).read_bytes())==digest, name

analytic=workspace/'build/performance/persistent-replay-v603'
analytic_files=verify_index(analytic)
manifest=json.loads((analytic/'manifest.json').read_text())
analytic_sources=members(analytic/'source.tar.gz')
for name,digest in manifest['source_sha256'].items():
    assert analytic_sources.get(name)==digest,name
extra_sources=set(analytic_sources)-set(manifest['source_sha256'])
for name in extra_sources:
    assert name.startswith('third_party/'),name
    committed=subprocess.run(['git','-C','/home/angus/spacepdhcg-persistent-replay-v603/repo',
                              'show',manifest['frozen_commit']+':'+name],
                             check=True,capture_output=True).stdout
    assert sha(committed)==analytic_sources[name],name

comparison=workspace/'build/performance/persistent-replay-20260909'
report=json.loads((comparison/'run/report.json').read_text())
assert report['complete'] and len(report['cases'])==4
for case in report['cases']:
    log=comparison/'run'/(case['label']+'-'+case['backend']+'.log')
    assert sha(log.read_bytes())==case['log_sha256']
    assert case['returncode']==0 and not case['timed_out']
    assert case['qualified']==sum(row['qualified'] for row in case['audits'])
    if case['backend']=='persistent': assert case['independent_native_audit_agrees']

output=workspace/'results/local/2026-09-09/persistent-snapshot-v603'
output.mkdir(exist_ok=False)
shutil.copytree(analytic,output/'analytic')
real=output/'real';real.mkdir()
for name in ('preparation.json','qoco-build.log','qoco-build-retry.log','qoco-build-retry.json'):
    shutil.copyfile(comparison/name,real/name)
shutil.copytree(comparison/'source',real/'source',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(comparison/'inputs',real/'inputs')
shutil.copyfile(comparison/'run/report.json',real/'report.json')
with tarfile.open(real/'raw.tar.gz','w:gz') as archive:
    for path in sorted((comparison/'run').iterdir()):
        assert path.is_file()
        archive.add(path,arcname='run/'+path.name)
shutil.copyfile(__file__,output/Path(__file__).name)
(output/'.gitattributes').write_bytes(b'* -text whitespace=cr-at-eol\n*.log -whitespace\n')
audit_record=dict(complete=True,substitution_index_files=sub_files,
    substitution_source_members=len(sub_sources),substitution_raw_members=len(sub_outputs),
    analytic_index_files=analytic_files,analytic_source_members=len(analytic_sources),
    cpp_source_members=len(manifest['source_sha256']),third_party_commit_members=len(extra_sources),
    real_cases_verified=len(report['cases']),real_qualified=sum(c['qualified'] for c in report['cases']))
(output/'publication-audit.json').write_text(json.dumps(audit_record,indent=2)+'\n')
print(json.dumps(audit_record))
