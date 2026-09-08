"""Freeze the bounded Halpern diagnostic; compile and run CPU checks with GPU hidden."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time

live=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
base=Path('/home/angus/spacepdhcg-common-kkt-v609e')
core=Path('/home/angus/spacepdhcg-persistent-replay-v603')
root=Path('/home/angus/spacepdhcg-halpern-v612c')
root.mkdir(exist_ok=False);repo=root/'repo';repo.mkdir()
owned=['cpp/cuda/include/spacepdhcg/cuda/persistent_pdhcg_c_api.h',
       'cpp/cuda/include/spacepdhcg/cuda/halpern_arithmetic.hpp',
       'cpp/cuda/src/persistent_pdhcg.cu','cpp/cuda/src/cooperative_pdhg.cuh',
       'cpp/cuda/src/persistent_halpern.cuh','cpp/cuda/tests/persistent_snapshot_replay.cu',
       'cpp/cuda/tests/persistent_snapshot_conversion_test.cpp','cpp/cuda/tests/persistent_halpern_test.cu',
       'cpp/cuda/CMakeLists.txt']
for directory in ('cpp','third_party'):shutil.copytree(base/'repo'/directory,repo/directory)
for path in owned:shutil.copy2(live/path,repo/path)
# CMake's already-published upstream reference target must have its original
# source available during configure; this file is not part of the new core.
upstream='cpp/cuda/tests/upstream_snapshot_replay.cu'
if not (repo/upstream).exists():
    (repo/upstream).write_bytes(subprocess.check_output(['git','show','HEAD:'+upstream],cwd=live))
shutil.copy2(__file__,root/Path(__file__).name)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_','QOCO_','PDHCG_','LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES']=''
manifest={'complete':False,'base_frozen_source':str(base/'repo'),'base_core':str(core/'build/cuda/libspacepdhcg_cuda.so'),
          'base_core_sha256':digest(core/'build/cuda/libspacepdhcg_cuda.so'),'owned_paths':owned,'stages':[]}
assert manifest['base_core_sha256']=='d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633'
def save():(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
def call(name,command,cwd=None,expected=0):
    start=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    manifest['stages'].append({'name':name,'command':command,'returncode':result.returncode,'seconds':time.perf_counter()-start});save()
    if result.returncode!=expected:raise RuntimeError(name+' failed: '+(root/(name+'.log')).read_text()[-8000:])
for name,command in [('git-init',['git','init']),('git-add',['git','add','-f','.']),
    ('git-freeze',['git','-c','user.name=Replay validation','-c','user.email=replay-validation@localhost','commit','-m','Freeze optional reflected Halpern diagnostic'])]:call(name,command,repo)
manifest['frozen_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
manifest['source_sha256']={p.relative_to(repo).as_posix():digest(p) for p in (repo/'cpp').rglob('*') if p.is_file()}
parts=['tests/persistent_snapshot.hpp','tests/persistent_snapshot_replay.cu','tests/cuda_test_support.hpp','include/spacepdhcg/cuda/persistent_pdhcg_c_api.h']
manifest['compiled_snapshot_source_sha256']=hashlib.sha256(''.join(p+':'+digest(repo/'cpp/cuda'/p)+'\n' for p in parts).encode()).hexdigest()
subprocess.run(['git','archive','--format=tar.gz','-o',str(root/'source.tar.gz'),'HEAD'],cwd=repo,check=True)
build=root/'build';(build/'cuda-tests').mkdir(parents=True)
objdir=Path('cuda/CMakeFiles/spacepdhcg_cuda.dir/src');(build/objdir).mkdir(parents=True)
manifest['reused_object_sha256']={}
for path in (core/'build'/objdir).glob('*.o'):
    if path.name=='persistent_pdhcg.cu.o':continue
    shutil.copy2(path,build/objdir/path.name);manifest['reused_object_sha256'][str(objdir/path.name)]=digest(path)
call('cpu-build',['/usr/bin/c++','-std=c++20','-O2','-Wall','-Wextra','-Wpedantic','-Werror','-I'+str(repo/'cpp/cuda/include'),'-I'+str(repo/'cpp/include'),str(repo/'cpp/cuda/tests/persistent_snapshot_conversion_test.cpp'),'-o',str(build/'cuda-tests/persistent_snapshot_conversion_test')])
call('cpu-test',[str(build/'cuda-tests/persistent_snapshot_conversion_test'),'--write-fixtures',str(root/'fixtures')])
commands=subprocess.check_output(['ninja','-t','commands','cuda/libspacepdhcg_cuda.so'],cwd=core/'build',text=True).splitlines()
for name,line in zip(('persistent-compile','core-device-link','core-link'),commands[-3:]):
    args=shlex.split(line.replace(str(core),str(root)))
    if args[:2]==[':','&&']:args=args[2:]
    if args[-2:]==['&&',':']:args=args[:-2]
    call(name,args,build)
manifest['library_sha256']=digest(build/'cuda/libspacepdhcg_cuda.so')
nvcc='/usr/local/cuda-12.8/bin/nvcc'
for name in ('persistent_snapshot_replay','persistent_halpern_test'):
    command=[nvcc,'-std=c++20','-O3','-arch=sm_120','-I'+str(repo/'cpp/cuda/include'),'-I'+str(repo/'cpp/include'),
             '-I'+str(repo/'cpp/cuda/tests'),'-Xcompiler=-Wall,-Wextra,-Werror',
             '-DSPACEPDHCG_SOURCE_COMMIT="'+manifest['frozen_commit']+'"',
             '-DSPACEPDHCG_SNAPSHOT_SOURCE_SHA256="'+manifest['compiled_snapshot_source_sha256']+'"',
             str(repo/'cpp/cuda/tests'/f'{name}.cu'),'-L'+str(build/'cuda'),'-lspacepdhcg_cuda',
             '-Xlinker=-rpath,'+str(build/'cuda'),'-ldl','-o',str(build/'cuda-tests'/name)]
    call(name+'-build',command);manifest[name+'_sha256']=digest(build/'cuda-tests'/name)
binary=build/'cuda-tests/persistent_snapshot_replay';validation=[]
actual=live/'build/performance/known-point-replay-v606/inputs'
for name in ('conditioning','difficult'):
    snapshot=actual/(name+'.txt');point=actual/(name+'-initial.txt')
    for mode in ('off','plain','adaptive'):
        test_name='validate-'+name+'-'+mode
        command=[str(binary),str(snapshot),'--validate-only','--common-kkt-stop','--initial-point',str(point),'--execution-blocks','2','--halpern',mode]
        call(test_name,command)
        records={line.split(' ',1)[0]:json.loads(line.split(' ',1)[1]) for line in (root/(test_name+'.log')).read_text().splitlines()}
        assert records['PERSISTENT_REPLAY_META']['halpern_mode']==mode
        assert records['PERSISTENT_REPLAY_INITIAL_POINT']['supplied_qualified']
        validation.append({'case':test_name,'all_json_records_parsed':True,'snapshot_sha256':digest(snapshot),'point_sha256':digest(point)})
for name,snapshot,extra in [('shifted',root/'fixtures/mixed-shifted.txt',[]),('nonzero-q',root/'fixtures/mixed.txt',[]),
    ('folded',actual/'conditioning.txt',['--fold-singleton-bounds']),('blocks-zero',actual/'conditioning.txt',['--execution-blocks','0'])]:
    block=[] if name=='blocks-zero' else ['--execution-blocks','2']
    call('reject-'+name,[str(binary),str(snapshot),'--validate-only','--common-kkt-stop','--halpern','plain']+block+extra,expected=1)
(root/'cpu-validation.json').write_text(json.dumps({'complete':True,'cuda_visible_devices':'','accepted':validation,'unsupported_rejected':['shifted','nonzero-q','folded','blocks-zero']},indent=2))
call('resource-usage',['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(build/'cuda/libspacepdhcg_cuda.so')])
# Configure only; enumerate tests without running any CUDA program.
cache=dict(line.split('=',1) for line in (core/'build/CMakeCache.txt').read_text().splitlines() if '=' in line and not line.startswith(('#','//')))
pinned=next(v for k,v in cache.items() if k.startswith('SPACEPDHCG_PDHCG_SOURCE_ROOT:'))
call('cmake-configure',[cache['CMAKE_COMMAND:INTERNAL'],'-S',str(repo/'cpp'),'-B',str(root/'cmake-check'),'-G','Ninja','-DSPACEPDHCG_BUILD_CUDA=ON',
    '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER='+nvcc,'-DCMAKE_CUDA_ARCHITECTURES=120','-DSPACEPDHCG_PDHCG_SOURCE_ROOT='+pinned])
call('ctest-enumerate',[cache['CMAKE_CTEST_COMMAND:INTERNAL'],'--test-dir',str(root/'cmake-check'),'-N','-V','-R','persistent_halpern'])
enum=(root/'ctest-enumerate.log').read_text();assert 'persistent_halpern_adaptive_test' in enum and 'plain' in enum and 'adaptive' in enum
manifest['complete']=True;save()
destination=live/'build/performance/halpern-v612c';destination.mkdir(exist_ok=False)
for path in root.iterdir():
    if path.is_file():shutil.copy2(path,destination/path.name)
shutil.copytree(root/'fixtures',destination/'fixtures')
print(json.dumps({k:manifest[k] for k in ('frozen_commit','library_sha256','persistent_snapshot_replay_sha256','persistent_halpern_test_sha256')}))
