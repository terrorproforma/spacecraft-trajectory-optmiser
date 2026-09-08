"""Resume b after a harness assertion mistakenly rejected expected exit-1 parser tests."""
from pathlib import Path
import hashlib,json,os,shlex,shutil,subprocess,tarfile,time
live=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
base=Path('/home/angus/spacepdhcg-l1-v615c')
core=Path('/home/angus/spacepdhcg-persistent-replay-v603')
root=Path('/home/angus/spacepdhcg-l1-weight-v618b');repo=root/'repo';build=root/'build'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'manifest.json').read_text());assert not manifest['complete']
assert [s['name'] for s in manifest['stages']]==['cpu-build','cpu-test']
shutil.copy2(root/'manifest.json',root/'manifest-before-continuation.json')
shutil.copy2(__file__,root/Path(__file__).name)
manifest['continuation']={'script':Path(__file__).name,'sha256':digest(Path(__file__)),
    'reason':'reuse guard initially treated expected exit-1 parser rejections as failures; no core/compiler/GPU failure'}
owned=manifest['owned_paths']
for name,sha in manifest['source_sha256'].items():assert digest(repo/name)==sha
env={k:v for k,v in os.environ.items() if not k.startswith(('SPACEPDHCG_','QOCO_','PDHCG_','LD_LIBRARY_PATH'))}
env['CUDA_VISIBLE_DEVICES']=''
source=(live/'build/performance/build_l1_weight_v618b.py').read_text()
exec(compile(source[source.index('def save():'):source.index("build=root/'build'")],str(__file__), 'exec'))
tail=source[source.index("reuse=Path('/home/angus/spacepdhcg-l1-weight-v618a')"):]
tail=tail.replace("if s['name']!='cmake-configure'","if s['name']!='cmake-configure' and not s['name'].startswith('reject-')")
exec(compile(tail,str(__file__),'exec'))
