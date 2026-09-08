from pathlib import Path
import hashlib,json,tarfile
home=Path.home();paths={}
assert json.loads((home/'spacepdhcg-fleet-v726/report.json').read_text())['success']
for version in ('v725','v726'):
    root=home/('spacepdhcg-fleet-'+version)
    for path in root.iterdir():
        if path.is_file() and path.suffix in ('.json','.log'):paths[version+'/'+path.name]=path
owned=['cpp/cuda/include/spacepdhcg/cuda/gtoc12_fleet_c_api.h','cpp/cuda/src/gtoc12_fleet.cuh','cpp/cuda/src/orbitweaver_gpu.cu','src/spacepdhcg/gtoc12/gpu_fleet.py','src/spacepdhcg/gtoc12/cooperative.py','src/spacepdhcg/gtoc12/cli.py','tests/test_gtoc12_gpu_fleet.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_collectdp.py']
for name in owned:paths['final-source/'+name]=home/'spacepdhcg-fleet-v726/repo'/name
manifest={name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in sorted(paths.items())}
out=home/'fleet-followup-v726.tar.gz'
with tarfile.open(out,'w:gz') as tar:
    for name,p in sorted(paths.items()):tar.add(p,arcname=name)
(home/'fleet-followup-v726-files.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(archive=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),members=len(manifest))))
