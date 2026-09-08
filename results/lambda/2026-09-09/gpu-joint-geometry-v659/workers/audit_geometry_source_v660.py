from pathlib import Path
import hashlib,json,re,subprocess
frozen=Path('/home/angus/spacepdhcg-joint-geometry-v651/repo');dest=Path('results/lambda/2026-09-09/gpu-joint-geometry-v659')
names=['cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h','cpp/cuda/internal/gtoc12_joint_geometry.h','cpp/cuda/src/gtoc12_joint.cu','cpp/cuda/src/orbitweaver_gpu.cu','src/spacepdhcg/gtoc12/gpu_joint.py','tests/test_gtoc12_gpu_joint_geometry.py']
records={}
for name in names:
 before=(frozen/name).read_bytes();after=Path(name).read_bytes()
 exact=before==after
 if not exact:
  assert name==names[0]
  strip=lambda b:re.sub(rb'/\*.*?\*/',b'',b,flags=re.S).replace(b'\r\n',b'\n')
  assert strip(before)==strip(after)
 records[name]=dict(frozen_sha256=hashlib.sha256(before).hexdigest(),published_sha256=hashlib.sha256(after).hexdigest(),byte_identical=exact,comment_only=not exact)
report=dict(source_files=records,note='After measurements, only the public header buffer-ownership comment changed. Runtime source and final test are otherwise byte-identical to the frozen tested source.',parent_at_publication=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
(dest/'published-source.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
