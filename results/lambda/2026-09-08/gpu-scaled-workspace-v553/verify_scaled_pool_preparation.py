from pathlib import Path
import hashlib,json,shutil,sys,tempfile
sys.path.insert(0,str(Path('scripts/gpu').resolve()))
from prepare_qoco_preserve_objective import prepare
old=Path('/home/angus/build-qoco-soc-step-v358/source')
new=Path('/home/angus/build-qoco-scaled-pool-v540/source')
paths=['src/equilibration.c','algebra/cuda/qoco_device_update.cuh']
with tempfile.TemporaryDirectory(prefix='qoco-preserve-provenance-') as directory:
 root=Path(directory)
 for name in paths:
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/name,dest)
 result=prepare(root)
 for name in paths:assert (root/name).read_bytes()==(new/name).read_bytes(),name
 # Reapplying must reject the unexpected layout without partially changing it.
 before={name:(root/name).read_bytes() for name in paths}
 try:prepare(root)
 except RuntimeError:pass
 else:raise AssertionError('reapplication unexpectedly accepted')
 assert all((root/name).read_bytes()==before[name] for name in paths)
result['final_preparer_sha256']=hashlib.sha256(Path('scripts/gpu/prepare_qoco_preserve_objective.py').read_bytes()).hexdigest()
result['compiled_outputs_match']=True
Path('build/performance/scaled-pool-preparation-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
