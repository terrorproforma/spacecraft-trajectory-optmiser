"""Saved-byte audit. This does not rerun numerical propagation or CUDA solvers."""
from pathlib import Path
import collections,hashlib,json,math,tarfile
root=Path(__file__).resolve().parent
if not (root/'h100.tar.gz').exists():root=Path('results/lambda/2026-09-09/gpu-regeneration-v799')
output={};plans={}
for side in ('local','h100'):
    archive=root/(side+'.tar.gz');receipt=json.loads((root/(side+'-receipt.json')).read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[x.name for x in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,r in manifest.items():
            data=tar.extractfile(name).read();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'],name
        def read(name):return json.load(tar.extractfile(name))
        source=read('runtime/source-manifest.json')
        for name,digest in source['files'].items():assert manifest['runtime/repo/'+name]['sha256']==digest
        for version in ('v792','v793','v794','v795','v796','v798'):
            r=read(version+'/report.json')
            assert r['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
            if 'qoco_sha256' in r:assert r['qoco_sha256']==manifest['runtime/libqoco.so']['sha256']
        assert read('v792/report.json')['bonus_sha256']==manifest['runtime/bonus_coefficients.txt']['sha256']
        bonus=[float(line.split()[0]) for line in tar.extractfile('runtime/bonus_coefficients.txt').read().decode().splitlines() if line.strip()]
        assert len(bonus)==60000
        results={}
        for version in ('v793','v795','v796','v798'):
            r=read(version+'/report.json');assert r['complete'] and r['success']
            path=version+'/fleet/Result.txt';assert manifest[path]['sha256']==r['solution_sha256']
            assert r['qualified']==(r['independent']['ok'] and r['official']['ok'])
            assert r['qualified']==(version!='v793')
            pending=None;mass=[];weighted=[];ships=set();asteroids=set()
            for line in tar.extractfile(path).read().decode().splitlines():
                fields=line.split()
                if not fields:continue
                ship,event=map(int,fields[:2]);ships.add(ship)
                if event==-1:continue
                value=float(fields[-1])
                if pending is None:pending=(ship,event,value)
                else:
                    assert pending[:2]==(ship,event)
                    if event>0 and value>pending[2]:
                        cargo=value-pending[2];mass.append(cargo);weighted.append(cargo*bonus[event-1]);asteroids.add(event)
                    pending=None
            assert pending is None
            assert abs(math.fsum(weighted)-r['score_kg'])<1e-7
            assert abs(math.fsum(mass)-r['independent']['total_mass_kg'])<1e-7
            limit=min(100.,2*math.exp(.004*math.fsum(mass)/len(ships)))
            assert (len(ships)<=limit+1e-9)==r['qualified']
            if version!='v793':assert all(c['certified'] for c in r['master']['selected'])
            results[version]=dict(qualified=r['qualified'],weighted_kg=math.fsum(weighted),raw_kg=math.fsum(mass),ships=len(ships),asteroids=len(asteroids),ship_limit=limit)
        for version in ('v793','v795'):
            r=read(version+'/report.json');solves=[json.loads(x) for x in tar.extractfile(version+'/solves.jsonl').read().decode().splitlines()]
            assert len(solves)==r['native_solves']
            results[version]['native_status_counts']=dict(collections.Counter(s.get('status','error') for s in solves))
        plans[side]={name:read(name) for name in manifest if name.endswith('/plans.json')}
        output[side]=dict(verified_members=len(manifest),results=results)
proxy_differences=[]
def compare(a,b,path):
    if a==b:return
    if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
        for k in a:compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    else:
        # Only approximate Lambert/propellant fields may vary. Exact schedules,
        # cargo, eligibility, candidate ordering and body IDs must match.
        assert path.split('/')[-1] in ('dv_proxy_km_s','inflation','propellant_proxy_kg','final_mass_proxy_kg'),path
        assert isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b),path
        assert abs(a-b)<1e-9,path
        proxy_differences.append(abs(a-b))
compare(plans['local'],plans['h100'],'plans')
output['candidate_comparison']=dict(files=len(plans['local']),exact_schedules_and_cargo=True,proxy_differences=len(proxy_differences),maximum_proxy_absolute_difference=max(proxy_differences,default=0.))
print(json.dumps(output,indent=2))
