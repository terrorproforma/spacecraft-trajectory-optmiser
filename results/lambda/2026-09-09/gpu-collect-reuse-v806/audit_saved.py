"""Verify saved bytes and bookkeeping; numerical propagation is recorded separately."""
from pathlib import Path
import collections,hashlib,json,math,statistics,tarfile
root=Path(__file__).resolve().parent
if not (root/'h100.tar.gz').exists():root=Path('results/lambda/2026-09-09/gpu-collect-reuse-v806')
output={};plans={}
for side in ('local','h100'):
    archive=root/(side+'.tar.gz');receipt=json.loads((root/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[m.name for m in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,row in manifest.items():
            b=tar.extractfile(name).read();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],name
        def read(name):return json.load(tar.extractfile(name))
        for name,digest in read('runtime/source-manifest.json')['files'].items():assert manifest['runtime/repo/'+name]['sha256']==digest
        for version in ('v800','v801','v802','v803','v804','v805'):
            r=read(version+'/report.json');assert r['complete'] and r['success']
            assert r['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
        assert read('v803/report.json')['qoco_sha256']==manifest['runtime/libqoco.so']['sha256']
        assert read('v802/report.json')['bonus_sha256']==manifest['runtime/bonus_coefficients.txt']['sha256']
        assert manifest['v802/fleet.txt']['sha256']=='97d1f351bf6ad4907ddce887aa47d1fa974f587ab270374f4bb46a5898491d48'
        for stage in read('v805/report.json')['stages']:assert stage['code']==0
        bonus=[float(line.split()[0]) for line in tar.extractfile('runtime/bonus_coefficients.txt').read().decode().splitlines() if line.strip()];assert len(bonus)==60000
        results={}
        for version in ('v803','v804'):
            r=read(version+'/report.json');path=version+'/fleet/Result.txt'
            assert r['qualified'] and r['independent']['ok'] and r['official']['ok']
            assert manifest[path]['sha256']==r['solution_sha256']
            assert all(c['certified'] for c in r['master']['selected'])
            pending=None;mass=[];weighted=[];ships=set();asteroids=set()
            for line in tar.extractfile(path).read().decode().splitlines():
                f=line.split()
                if not f:continue
                ship,event=map(int,f[:2]);ships.add(ship)
                if event==-1:continue
                value=float(f[-1])
                if pending is None:pending=(ship,event,value)
                else:
                    assert pending[:2]==(ship,event)
                    if event>0 and value>pending[2]:
                        cargo=value-pending[2];mass.append(cargo);weighted.append(cargo*bonus[event-1]);asteroids.add(event)
                    pending=None
            assert pending is None
            assert abs(math.fsum(weighted)-r['score_kg'])<1e-7
            assert abs(math.fsum(mass)-r['independent']['total_mass_kg'])<1e-7
            limit=min(100.,2*math.exp(.004*math.fsum(mass)/len(ships)));assert len(ships)<=limit+1e-9
            assert r['score_kg']>=12992.035662031367-1e-7
            results[version]=dict(weighted_kg=math.fsum(weighted),raw_kg=math.fsum(mass),ships=len(ships),asteroids=len(asteroids),ship_limit=limit)
        solves=[json.loads(s) for s in tar.extractfile('v803/solves.jsonl').read().decode().splitlines()]
        assert len(solves)==read('v803/report.json')['native_solves']
        bench=read('v801/report.json');assert len(bench['runs'])==8
        for ship in (10,21):
            hashes=set()
            for run in bench['runs']:
                row=next(x for x in run['routes'] if x['ship']==ship)
                name=f"v801/{run['name']}/ship-{ship:02d}/plans.json"
                assert manifest[name]['sha256']==row['plans_sha256'];hashes.add(row['plans_sha256'])
                assert row['telemetry']['collect_dp_allocations']==({10:26,21:25}[ship] if run['reuse'] else {10:1152,21:1032}[ship])
                assert row['telemetry'].get('collect_dp_rebinds',0)==({10:1126,21:1007}[ship] if run['reuse'] else 0)
            assert len(hashes)==1
            for reuse,key in ((False,'fresh_seconds'),(True,'reuse_seconds')):
                values=[x['seconds'] for run in bench['runs'] if run['measured'] and run['reuse']==reuse for x in run['routes'] if x['ship']==ship]
                assert len(values)==3 and statistics.median(values)==bench['medians'][str(ship)][key]
        plans[side]={name:read(name) for name in manifest if name.startswith('v802/') and name.endswith('/plans.json')}
        output[side]=dict(verified_members=len(manifest),results=results,native_status_counts=dict(collections.Counter(s.get('status','error') for s in solves)),medians=bench['medians'])
differences=[]
def compare(a,b,path):
    if a==b:return
    if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
        for k in a:compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    else:
        assert path.split('/')[-1] in ('dv_proxy_km_s','inflation','propellant_proxy_kg','final_mass_proxy_kg'),path
        assert isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b)
        assert abs(a-b)<1e-9,path
        differences.append(abs(a-b))
compare(plans['local'],plans['h100'],'plans')
output['candidate_comparison']=dict(files=len(plans['local']),exact_schedules_and_cargo=True,proxy_differences=len(differences),maximum_proxy_absolute_difference=max(differences,default=0.))
print(json.dumps(output,indent=2))
