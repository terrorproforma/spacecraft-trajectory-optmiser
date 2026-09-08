"""Add an explicit strategy selector to replay, linking the unchanged reviewed core."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

live=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
base=Path('/home/angus/spacepdhcg-common-kkt-v609d');root=Path('/home/angus/spacepdhcg-common-kkt-v609e')
root.mkdir(exist_ok=False);repo=root/'repo';repo.mkdir()
for directory in ('cpp','third_party'):shutil.copytree(base/'repo'/directory,repo/directory)
owned='cpp/cuda/tests/persistent_snapshot_replay.cu';shutil.copy2(live/owned,repo/owned)
shutil.copy2(__file__,root/'build_common_kkt_replay_v609e.py')
previous=json.loads((base/'manifest.json').read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'complete':False,'base_manifest_sha256':digest(base/'manifest.json'),'base_frozen_source':str(base/'repo'),
          'owned_paths':previous['owned_paths'],'adapter_delta_paths':[owned],'library_sha256':previous['library_sha256'],'stages':[]}
def save():(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_','QOCO_','PDHCG_','LD_LIBRARY_PATH'))};env['CUDA_VISIBLE_DEVICES']=''
def call(name,command,cwd=None,expected=0):
    start=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:r=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    manifest['stages'].append({'name':name,'command':command,'returncode':r.returncode,'seconds':time.perf_counter()-start});save()
    assert r.returncode==expected,(name,(root/(name+'.log')).read_text()[-6000:])
for name,command in [('git-init',['git','init']),('git-add',['git','add','-f','.']),('git-freeze',['git','-c','user.name=Replay validation','-c','user.email=replay-validation@localhost','commit','-m','Freeze explicit replay execution-block selection'])]:call(name,command,repo)
manifest['frozen_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
manifest['source_sha256']={p.relative_to(repo).as_posix():digest(p) for p in (repo/'cpp').rglob('*') if p.is_file()}
for path in previous['owned_paths']:
    if path!=owned:assert manifest['source_sha256'][path]==previous['source_sha256'][path]
parts=['tests/persistent_snapshot.hpp','tests/persistent_snapshot_replay.cu','tests/cuda_test_support.hpp','include/spacepdhcg/cuda/persistent_pdhcg_c_api.h']
manifest['compiled_snapshot_source_sha256']=hashlib.sha256(''.join(p+':'+digest(repo/'cpp/cuda'/p)+'\n' for p in parts).encode()).hexdigest()
(root/'build/cuda').mkdir(parents=True);(root/'build/cuda-tests').mkdir()
shutil.copy2(base/'build/cuda/libspacepdhcg_cuda.so',root/'build/cuda/libspacepdhcg_cuda.so')
assert digest(root/'build/cuda/libspacepdhcg_cuda.so')==manifest['library_sha256']
command=next(x['command'] for x in previous['stages'] if x['name']=='persistent_snapshot_replay-build')
command=[x.replace(str(base),str(root)).replace(previous['frozen_commit'],manifest['frozen_commit']).replace(previous['compiled_snapshot_source_sha256'],manifest['compiled_snapshot_source_sha256']) for x in command]
call('replay-build',command)
binary=root/'build/cuda-tests/persistent_snapshot_replay';manifest['persistent_snapshot_replay_sha256']=digest(binary)
validation=[]
for fixture in ('mixed','conditioning','difficult'):
    for blocks in (0,2):
        for common in (False,True):
            if fixture=='mixed':snapshot=base/'fixtures/mixed.txt';point=base/'fixtures/mixed-initial-original.txt'
            else:
                inputs=live/'build/performance/known-point-replay-v606/inputs';snapshot=inputs/(fixture+'.txt');point=inputs/(fixture+'-initial.txt')
            name=f'validate-{fixture}-{blocks}-{int(common)}'
            command=[str(binary),str(snapshot),'--validate-only','--execution-blocks',str(blocks),'--initial-point',str(point)]
            if common:command.append('--common-kkt-stop')
            call(name,command)
            records={line.split(' ',1)[0]:json.loads(line.split(' ',1)[1]) for line in (root/(name+'.log')).read_text().splitlines()}
            assert records['PERSISTENT_REPLAY_META']['requested_execution_blocks']==blocks
            assert records['PERSISTENT_REPLAY_META']['common_kkt_initial_check']==common
            assert records['PERSISTENT_REPLAY_INITIAL_POINT']['supplied_qualified']
            validation.append({'name':name,'all_json_records_parsed':True})
call('reject-negative-blocks',[str(binary),str(base/'fixtures/mixed.txt'),'--validate-only','--execution-blocks','-1'],expected=1)
(root/'cpu-validation.json').write_text(json.dumps({'complete':True,'gpu_hidden':True,'cases':validation,'negative_blocks_rejected':True},indent=2))
subprocess.run(['git','archive','--format=tar.gz','-o',str(root/'source.tar.gz'),'HEAD'],cwd=repo,check=True)
manifest['complete']=True;save()
destination=live/'build/performance/common-kkt-v609e';destination.mkdir(exist_ok=False)
for path in root.iterdir():
    if path.is_file():shutil.copy2(path,destination/path.name)
print(json.dumps({k:manifest[k] for k in ('frozen_commit','library_sha256','persistent_snapshot_replay_sha256')}))
