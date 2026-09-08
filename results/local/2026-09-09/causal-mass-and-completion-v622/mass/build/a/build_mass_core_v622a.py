"""Freeze then build only the reviewed v622 persistent overlay. Never run GPU tests."""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

LIVE = Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
ROOT = Path('/home/angus/spacepdhcg-mass-v622a')
EVIDENCE = LIVE / 'build/performance/mass-core-v622a'
BASE = 'fdf52ae31d259240ffebdcf79ed0965dce8b9298'
OWNED = [
    'cpp/cuda/include/spacepdhcg/cuda/persistent_pdhcg_c_api.h',
    'cpp/cuda/include/spacepdhcg/cuda/mass_causal_arithmetic.hpp',
    'cpp/cuda/src/persistent_pdhcg.cu', 'cpp/cuda/src/persistent_l1_host.cuh',
    'cpp/cuda/src/persistent_mass.cuh', 'cpp/cuda/src/persistent_mass_host.cuh',
    'cpp/cuda/tests/persistent_snapshot_replay.cu',
    'cpp/cuda/tests/persistent_mass_snapshot.hpp', 'cpp/cuda/tests/persistent_mass_fixture.hpp',
    'cpp/cuda/tests/persistent_mass_test.cu', 'cpp/cuda/CMakeLists.txt',
]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
env = {k: v for k, v in os.environ.items() if not k.startswith(('SPACEPDHCG_', 'QOCO_', 'PDHCG_', 'LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES'] = ''
env['PYTHONDONTWRITEBYTECODE'] = '1'


def save(manifest):
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in ('.json', '.log', '.py', '.gz'):
            shutil.copy2(path, EVIDENCE / path.name)


def freeze():
    ROOT.mkdir(exist_ok=False);EVIDENCE.mkdir(exist_ok=False);repo=ROOT/'repo';repo.mkdir()
    git=['/usr/bin/git','-c','core.autocrlf=false','-c','safe.directory='+str(LIVE),'-C',str(LIVE)]
    assert subprocess.check_output(git+['rev-parse',BASE],text=True,env=env).strip()==BASE
    tree=subprocess.check_output(git+['rev-parse',BASE+'^{tree}'],text=True,env=env).strip()
    raw=subprocess.check_output(git+['archive','--format=tar',BASE,'cpp','third_party'],env=env)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
        for member in archive:
            path=Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts and (member.isfile() or member.isdir())
            if member.isdir():(repo/path).mkdir(parents=True,exist_ok=True)
            else:
                (repo/path).parent.mkdir(parents=True,exist_ok=True)
                (repo/path).write_bytes(archive.extractfile(member).read())
    base_sha={p.relative_to(repo).as_posix():sha(p) for p in sorted(repo.rglob('*')) if p.is_file()}
    # The only CMake change allowed in this overlay is the detector identity.
    cmake='cpp/cuda/CMakeLists.txt'
    expected=(repo/cmake).read_text().replace('tests/cuda_test_support.hpp tests/persistent_l1_snapshot.hpp',
                                            'tests/cuda_test_support.hpp tests/persistent_l1_snapshot.hpp tests/persistent_mass_snapshot.hpp')
    assert (LIVE/cmake).read_text()==expected, 'Foreign CMake change must not enter this freeze'
    for name in OWNED:
        (repo/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(LIVE/name,repo/name)
    sources={p.relative_to(repo).as_posix():sha(p) for p in sorted(repo.rglob('*')) if p.is_file()}
    assert all(sources[p]==digest for p,digest in base_sha.items() if p not in OWNED)
    assert set(sources)-set(base_sha)==set(OWNED)-set(base_sha)
    tree_sha=hashlib.sha256(''.join(p+':'+h+'\n' for p,h in sources.items()).encode()).hexdigest()
    with tarfile.open(ROOT/'source.tar.gz','w:gz') as archive:
        for name in sources:archive.add(repo/name,arcname=name,recursive=False)
    shutil.copy2(__file__,ROOT/Path(__file__).name)
    manifest={'complete':False,'source_frozen':True,'gpu_calls':0,'git_mutations':0,
        'base_commit':BASE,'base_git_tree':tree,'base_archive_sha256':hashlib.sha256(raw).hexdigest(),
        'source_identity_kind':'sha256_tree_not_git_commit','frozen_commit':None,'compiled_source_commit':'uncommitted',
        'source_dirty':True,'source_identity_scope':'C++/CUDA/CMake/third-party build sources only',
        'owned_paths':OWNED,'owned_sha256':{p:sources[p] for p in OWNED},'base_source_sha256':base_sha,
        'source_sha256':sources,'source_tree_sha256':tree_sha,'source_archive_sha256':sha(ROOT/'source.tar.gz'),
        'foreign_overlays':'Excluded: compact completion, search, fleet and other working-tree changes.',
        'reused_compiled_objects':[],'cuda_architectures':[120],'stages':[],
        'fixture_sha256':'7312df3f38291175e76df66459da86c68fdb71cb8c1895a67b68d72790291169',
        'mode':'default-off exact causal mass + L1; original-coordinate direct diagonal metric; no GPU qualification result'}
    save(manifest)
    print(json.dumps({'frozen':True,'manifest_sha256':sha(ROOT/'manifest.json'),'source_tree_sha256':tree_sha,'owned_files':len(OWNED)}))


def build():
    manifest=json.loads((ROOT/'manifest.json').read_text());repo=ROOT/'repo';assert manifest['source_frozen'] and not manifest['stages']
    assert not (ROOT/'build-marker.json').exists()
    for name,digest in manifest['source_sha256'].items():assert sha(repo/name)==digest,name
    assert sha(ROOT/'source.tar.gz')==manifest['source_archive_sha256']
    (ROOT/'build-marker.json').write_text(json.dumps({'single_build_attempt':True,'source_tree_sha256':manifest['source_tree_sha256']})+'\n')
    def call(name,command,expected=0):
        start=time.perf_counter()
        with (ROOT/(name+'.log')).open('x') as log:
            result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        manifest['stages'].append({'name':name,'command':command,'returncode':result.returncode,'expected':expected,'seconds':time.perf_counter()-start})
        save(manifest)
        if result.returncode!=expected:raise RuntimeError(name+': '+(ROOT/(name+'.log')).read_text()[-8000:])
    cache_path=Path('/home/angus/spacepdhcg-persistent-replay-v603/build/CMakeCache.txt')
    cache=dict(line.split('=',1) for line in cache_path.read_text().splitlines() if '=' in line and not line.startswith(('#','//')))
    cmake=cache['CMAKE_COMMAND:INTERNAL'];ctest=cache['CMAKE_CTEST_COMMAND:INTERNAL']
    pinned=next(v for k,v in cache.items() if k.startswith('SPACEPDHCG_PDHCG_SOURCE_ROOT:'))
    manifest['cmake_cache_source']={'path':str(cache_path),'sha256':sha(cache_path)};manifest['pinned_upstream_checkout']=pinned
    build_dir=ROOT/'build'
    call('compiler-version',['/usr/local/cuda-12.8/bin/nvcc','--version'])
    call('configure',[cmake,'-S',str(repo/'cpp'),'-B',str(build_dir),'-G','Ninja','-DSPACEPDHCG_BUILD_CUDA=ON',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc','-DCMAKE_CUDA_ARCHITECTURES=120',
        '-DSPACEPDHCG_PDHCG_SOURCE_ROOT='+pinned,'-DSPACEPDHCG_FROZEN_SOURCE_TREE_SHA256='+manifest['source_tree_sha256'],
        '-DSPACEPDHCG_FROZEN_SOURCE_BASE_COMMIT='+BASE,'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
    call('compile',[cmake,'--build',str(build_dir),'--target','spacepdhcg_cuda','persistent_mass_test','persistent_snapshot_replay','persistent_snapshot_conversion_test','-j','4'])
    lib=build_dir/'cuda/libspacepdhcg_cuda.so';test=build_dir/'cuda-tests/persistent_mass_test';replay=build_dir/'cuda-tests/persistent_snapshot_replay'
    conversion=build_dir/'cuda/persistent_snapshot_conversion_test'
    for name,path in [('library',lib),('test',test),('replay',replay),('conversion_test',conversion)]:
        manifest[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
    save(manifest)
    call('cpu-mass',[str(test),'--cpu-only']);call('cpu-snapshot',[str(conversion)])
    inputs=LIVE/'build/performance/known-point-replay-v606/inputs'
    for case in ('conditioning','difficult'):
        for mode,flags in [('unit',[]),('mass',['--mass-eliminate']),('mass-seed',['--mass-eliminate','--initial-point',str(inputs/(case+'-initial.txt'))])]:
            name='validate-'+case+'-'+mode
            call(name,[str(replay),str(inputs/(case+'.txt')),'--validate-only','--common-kkt-stop','--l1-prox','--execution-blocks','2',*flags])
            records=[]
            for line in (ROOT/(name+'.log')).read_text().splitlines():
                if line.startswith('PERSISTENT_'):records.append(json.loads(line.split(' ',1)[1],parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))))
            assert records and records[0]['source_tree_sha256']==manifest['source_tree_sha256']
            if mode.startswith('mass'):assert len(records[0]['mass_map'])==(211 if case=='conditioning' else 234)
    for label,flags in [('without-l1',['--mass-eliminate']),('weighted',['--l1-prox','--mass-eliminate','--l1-weight','4'])]:
        call('reject-'+label,[str(replay),str(inputs/'conditioning.txt'),'--validate-only','--common-kkt-stop','--execution-blocks','2',*flags],expected=1)
    call('ctest-enumerate',[ctest,'--test-dir',str(build_dir),'-N','-V','-R','persistent_mass_test'])
    call('resources',['/usr/local/cuda-12.8/bin/cuobjdump','--dump-resource-usage',str(lib)])
    call('symbols',['/usr/bin/nm','-D','--defined-only',str(lib)])
    for name in ('spacepdhcg_cuda_workspace_set_mass_options','spacepdhcg_cuda_workspace_mass_diagnostics'):
        assert name in (ROOT/'symbols.log').read_text()
    shutil.copy2(build_dir/'compile_commands.json',ROOT/'compile-commands.json')
    manifest['compile_commands_sha256']=sha(ROOT/'compile-commands.json')
    manifest['object_sha256']={p.relative_to(build_dir).as_posix():sha(p) for p in sorted(build_dir.rglob('*.o'))}
    manifest['complete']=True;save(manifest)
    print(json.dumps({'complete':True,'manifest_sha256':sha(ROOT/'manifest.json'),'library_sha256':manifest['library']['sha256']}))


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('freeze','build')
    freeze() if sys.argv[1]=='freeze' else build()
