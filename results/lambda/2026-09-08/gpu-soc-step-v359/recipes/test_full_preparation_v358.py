from pathlib import Path
import json,subprocess,re,hashlib
root=Path('build/performance/soc-step-header-v358')
recipe=Path('scripts/gpu/prepare_qoco_gpu.py')
base=json.loads(Path('/home/angus/build-qoco-nonfinite-ir-v336/source/spacepdhcg-provenance.json').read_text())
options=set(re.findall(r'"(--[a-z-]+)"',recipe.read_text()))
dest=Path('/home/angus/qoco-soc-step-preparation-v358')
cmd=['python3',str(recipe),'--source',base['source'],'--destination',str(dest)]
cmd.extend('--'+k.replace('_','-') for k,v in base.items() if v is True and '--'+k.replace('_','-') in options)
r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(root/'full-preparation.log').write_text(r.stdout+r.stderr);r.check_returncode()
report=json.loads((dest/'spacepdhcg-provenance.json').read_text())
assert report['compensated_soc_steps']
assert report['prepared_files_sha256']['src/qoco_soc_step.cuh']==hashlib.sha256(Path('cpp/cuda/patches/qoco_soc_step.cuh').read_bytes()).hexdigest()
assert '#include "qoco_soc_step.cuh"' in (dest/'src/cone.cu').read_text()
(root/'full-preparation.json').write_text(json.dumps(dict(command=cmd,provenance=report),indent=2));print('Full preparation and emitted header provenance passed')
