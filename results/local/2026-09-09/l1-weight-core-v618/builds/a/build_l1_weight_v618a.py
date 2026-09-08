"""Freeze and compile v618 without Git operations; only CPU checks run."""
from pathlib import Path
import hashlib,json,os,shlex,shutil,subprocess,tarfile,time
live=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
base=Path('/home/angus/spacepdhcg-l1-v615c')
core=Path('/home/angus/spacepdhcg-persistent-replay-v603')
root=Path('/home/angus/spacepdhcg-l1-weight-v618a')
root.mkdir(exist_ok=False);repo=root/'repo';repo.mkdir()
owned=['cpp/cuda/include/spacepdhcg/cuda/persistent_pdhcg_c_api.h',
       'cpp/cuda/include/spacepdhcg/cuda/l1_epigraph_arithmetic.hpp',
       'cpp/cuda/src/persistent_pdhcg.cu','cpp/cuda/src/persistent_l1.cuh','cpp/cuda/src/persistent_l1_host.cuh',
       'cpp/cuda/tests/persistent_l1_snapshot.hpp','cpp/cuda/tests/persistent_snapshot_replay.cu',
       'cpp/cuda/tests/persistent_snapshot_conversion_test.cpp','cpp/cuda/tests/persistent_l1_test.cu',
       'cpp/cuda/CMakeLists.txt']
for directory in ('cpp','third_party'):shutil.copytree(base/'repo'/directory,repo/directory)
for path in owned:shutil.copy2(live/path,repo/path)
shutil.copy2(__file__,root/Path(__file__).name)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_','QOCO_','PDHCG_','LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES']=''
manifest={'complete':False,'base_frozen_source':str(base/'repo'),
    'base_commit':json.loads((base/'manifest.json').read_text())['frozen_commit'],
    'base_manifest_sha256':digest(base/'manifest.json'),
    'workspace_parent_commit':'9c8f2e9673c577324f6fcfdfc007346766b0129d',
    'base_core':str(core/'build/cuda/libspacepdhcg_cuda.so'),
    'base_core_sha256':digest(core/'build/cuda/libspacepdhcg_cuda.so'),
    'owned_paths':owned,'stages':[],'git_operations':0,'gpu_calls':0}
assert manifest['base_core_sha256']=='d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633'
assert digest(base/'build/cuda/libspacepdhcg_cuda.so')=='83487574646fe67fce156c9a2055f448341c99f189ac2c66d856bfcf1b280047'
manifest['source_sha256']={p.relative_to(repo).as_posix():digest(p) for p in sorted(repo.rglob('*')) if p.is_file()}
tree=''.join(k+':'+v+'\n' for k,v in manifest['source_sha256'].items())
manifest['source_tree_sha256']=hashlib.sha256(tree.encode()).hexdigest()
manifest['source_identity_kind']='sha256_tree_not_git_commit'
manifest['frozen_commit']=None
manifest['compiled_source_commit']='uncommitted'
parts=['tests/persistent_snapshot.hpp','tests/persistent_snapshot_replay.cu','tests/cuda_test_support.hpp',
       'tests/persistent_l1_snapshot.hpp','include/spacepdhcg/cuda/persistent_pdhcg_c_api.h']
manifest['compiled_snapshot_source_sha256']=hashlib.sha256(''.join(p+':'+digest(repo/'cpp/cuda'/p)+'\n' for p in parts).encode()).hexdigest()
with tarfile.open(root/'source.tar.gz','w:gz') as tar:
    for name in manifest['source_sha256']:tar.add(repo/name,arcname=name,recursive=False)
def save():(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
def call(name,command,cwd=None,expected=0):
    start=time.perf_counter()
    with (root/(name+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
    manifest['stages'].append({'name':name,'command':command,'returncode':result.returncode,'seconds':time.perf_counter()-start});save()
    if result.returncode!=expected:raise RuntimeError(name+' failed: '+(root/(name+'.log')).read_text()[-8000:])
build=root/'build';(build/'cuda-tests').mkdir(parents=True)
objdir=Path('cuda/CMakeFiles/spacepdhcg_cuda.dir/src');(build/objdir).mkdir(parents=True)
manifest['reused_object_sha256']={}
for path in (core/'build'/objdir).glob('*.o'):
    if path.name=='persistent_pdhcg.cu.o':continue
    shutil.copy2(path,build/objdir/path.name);manifest['reused_object_sha256'][str(objdir/path.name)]=digest(path)
call('cpu-build',['/usr/bin/c++','-std=c++20','-O2','-Wall','-Wextra','-Wpedantic','-Werror',
    '-I'+str(repo/'cpp/cuda/include'),'-I'+str(repo/'cpp/include'),
    str(repo/'cpp/cuda/tests/persistent_snapshot_conversion_test.cpp'),'-o',str(build/'cuda-tests/persistent_snapshot_conversion_test')])
call('cpu-test',[str(build/'cuda-tests/persistent_snapshot_conversion_test'),'--write-fixtures',str(root/'fixtures')])
commands=subprocess.check_output(['ninja','-t','commands','cuda/libspacepdhcg_cuda.so'],cwd=core/'build',text=True).splitlines()
for name,line in zip(('persistent-compile','core-device-link','core-link'),commands[-3:]):
    args=shlex.split(line.replace(str(core),str(root)))
    if args[:2]==[':','&&']:args=args[2:]
    if args[-2:]==['&&',':']:args=args[:-2]
    call(name,args,build)
manifest['library_sha256']=digest(build/'cuda/libspacepdhcg_cuda.so')
nvcc='/usr/local/cuda-12.8/bin/nvcc'
for name in ('persistent_snapshot_replay','persistent_l1_test'):
    command=[nvcc,'-std=c++20','-O3','-arch=sm_120','-I'+str(repo/'cpp/cuda/include'),'-I'+str(repo/'cpp/include'),
        '-I'+str(repo/'cpp/cuda/tests'),'-Xcompiler=-Wall,-Wextra,-Werror',
        '-DSPACEPDHCG_SOURCE_COMMIT="'+manifest['compiled_source_commit']+'"',
        '-DSPACEPDHCG_SOURCE_TREE_SHA256="'+manifest['source_tree_sha256']+'"',
        '-DSPACEPDHCG_SOURCE_BASE_COMMIT="'+manifest['base_commit']+'"',
        '-DSPACEPDHCG_SNAPSHOT_SOURCE_SHA256="'+manifest['compiled_snapshot_source_sha256']+'"',
        str(repo/'cpp/cuda/tests'/f'{name}.cu'),'-L'+str(build/'cuda'),'-lspacepdhcg_cuda',
        '-Xlinker=-rpath,'+str(build/'cuda'),'-ldl','-o',str(build/'cuda-tests'/name)]
    call(name+'-build',command);manifest[name+'_sha256']=digest(build/'cuda-tests'/name)
binary=build/'cuda-tests/persistent_snapshot_replay';validation=[]
actual=live/'build/performance/known-point-replay-v606/inputs'
def strict_json(s):
    def bad(value):raise ValueError('nonfinite JSON '+value)
    return json.loads(s,parse_constant=bad)
for name in ('conditioning','difficult'):
    snapshot=actual/(name+'.txt');point=actual/(name+'-initial.txt')
    call('inspect-'+name,[str(build/'cuda-tests/persistent_snapshot_conversion_test'),'--inspect-l1',str(snapshot)])
    for mode in ('default','unit','fixed','cancel','fixed-seeded','cancel-seeded'):
        test_name='validate-'+name+'-'+mode
        command=[str(binary),str(snapshot),'--validate-only','--common-kkt-stop','--execution-blocks','2']
        if mode!='default':command+=['--l1-prox']
        if mode.startswith('fixed'):command+=['--l1-weight','.25']
        if mode.startswith('cancel'):command+=['--l1-weight','cancel-global']
        if mode.endswith('seeded'):command+=['--initial-point',str(point)]
        call(test_name,command)
        records={line.split(' ',1)[0]:strict_json(line.split(' ',1)[1]) for line in (root/(test_name+'.log')).read_text().splitlines()}
        meta=records['PERSISTENT_REPLAY_META'];assert meta['l1_prox']==(mode!='default')
        assert meta['source_tree_sha256']==manifest['source_tree_sha256'] and meta['source_commit']=='uncommitted'
        assert meta['source_commit_scope']=='uncommitted_frozen_source_tree' and meta['source_dirty'] is True
        assert meta['base_commit']==manifest['base_commit']
        expected='cancel_global_normalization' if mode.startswith('cancel') else 'fixed_user_input' if mode.startswith('fixed') else 'unit_default'
        assert meta['l1_weight_policy']==expected
        if mode.endswith('seeded'):assert records['PERSISTENT_REPLAY_INITIAL_POINT']['supplied_qualified']
        validation.append({'case':test_name,'all_json_records_parsed':True,'snapshot_sha256':digest(snapshot),'point_sha256':digest(point)})
rejected=[]
for name,extras in [('no-l1',['--l1-weight','4']),('zero',['--l1-prox','--l1-weight','0']),
    ('negative',['--l1-prox','--l1-weight','-1']),('nan',['--l1-prox','--l1-weight','nan']),
    ('infinity',['--l1-prox','--l1-weight','inf']),('reciprocal-overflow',['--l1-prox','--l1-weight','5e-324']),
    ('unknown-policy',['--l1-prox','--l1-weight','adaptive']),
    ('duplicate',['--l1-prox','--l1-weight','4','--l1-weight','cancel-global'])]:
    call('reject-'+name,[str(binary),str(actual/'conditioning.txt'),'--validate-only','--common-kkt-stop','--execution-blocks','2']+extras,expected=1)
    rejected.append(name)
(root/'cpu-validation.json').write_text(json.dumps({'complete':True,'cuda_visible_devices':'','accepted':validation,'rejected':rejected},indent=2))
call('resource-usage',['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(build/'cuda/libspacepdhcg_cuda.so')])
cache=dict(line.split('=',1) for line in (core/'build/CMakeCache.txt').read_text().splitlines() if '=' in line and not line.startswith(('#','//')))
pinned=next(v for k,v in cache.items() if k.startswith('SPACEPDHCG_PDHCG_SOURCE_ROOT:'))
call('cmake-configure',[cache['CMAKE_COMMAND:INTERNAL'],'-S',str(repo/'cpp'),'-B',str(root/'cmake-check'),'-G','Ninja','-DSPACEPDHCG_BUILD_CUDA=ON',
    '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER='+nvcc,'-DCMAKE_CUDA_ARCHITECTURES=120','-DSPACEPDHCG_PDHCG_SOURCE_ROOT='+pinned])
call('ctest-enumerate',[cache['CMAKE_CTEST_COMMAND:INTERNAL'],'--test-dir',str(root/'cmake-check'),'-N','-V','-R','persistent_(halpern|l1)'])
assert 'persistent_l1_weight_test' in (root/'ctest-enumerate.log').read_text()
manifest['complete']=True;save()
destination=live/'build/performance/l1-weight-v618a';destination.mkdir(exist_ok=False)
for path in root.iterdir():
    if path.is_file():shutil.copy2(path,destination/path.name)
shutil.copytree(root/'fixtures',destination/'fixtures')
print(json.dumps({k:manifest[k] for k in ('frozen_commit','library_sha256','persistent_snapshot_replay_sha256','persistent_l1_test_sha256')}))
