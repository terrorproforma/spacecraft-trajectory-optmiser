"""Audit saved runtime identities, repeated searches and emitted fleet scores."""
from pathlib import Path
import collections,hashlib,json,math,statistics,tarfile
root=Path(__file__).resolve().parent
if not (root/'h100.tar.gz').exists():root=Path('results/lambda/2026-09-09/gpu-catalogue-cache-v822')
output={}
for side in ('local','h100'):
    archive=root/(side+'.tar.gz');receipt=json.loads((root/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[m.name for m in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,row in manifest.items():
            b=tar.extractfile(name).read();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],name
        def read(name):return json.load(tar.extractfile(name))
        final=read('runtime/source-manifest.json')['files']
        for name,digest in final.items():assert manifest['runtime/repo/'+name]['sha256']==digest,name
        for name,digest in read('v808/source-manifest.json')['files'].items():
            path=('runtime/repo/' if final.get(name)==digest else 'v808/original/')+name
            assert manifest[path]['sha256']==digest,name
            if name.startswith('cpp/'):assert final[name]==digest
        for version in ('v814','v815','v816','v817','v819','v820'):
            r=read(version+'/report.json');assert r['complete'] and r['success']
            assert r['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
        for version in ('v814','v819'):
            r=read(version+'/report.json');assert all(s['code']==0 for s in r['stages'])
            assert r['controller_sha256']==manifest['runtime/gtoc12_scvx_test']['sha256']
        assert read('v819/report.json')['source_manifest_sha256']==manifest['runtime/source-manifest.json']['sha256']
        assert read('v815/report.json')['qoco_sha256']==manifest['runtime/libqoco.so']['sha256']
        for version in ('v809','v810','v812','v813'):
            r=read(version+'/report.json');assert r['complete'] and not r['success']
            assert not r.get('runs',[]) and r.get('native_solves',0)==0
        assert manifest['v817/incumbent.txt']['sha256']=='765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da'
        comparisons={}
        for version in ('v816','v820'):
            bench=read(version+'/report.json');assert len(bench['runs'])==8
            medians={}
            for ship,count in ((10,650),(21,652)):
                hashes=set();packing={};counts={}
                for run in bench['runs']:
                    row=next(x for x in run['routes'] if x['ship']==ship)
                    digest=manifest[f"{version}/{run['name']}/ship-{ship:02d}/plans.json"]['sha256']
                    assert digest==row['plans_sha256'];hashes.add(digest)
                    t=row['telemetry'];assert t['completion_catalogue_hashes']+t['completion_catalogue_hash_reuses']==t['completion_batches']==count
                    assert row['candidates']==(517 if ship==10 else 472)
                assert len(hashes)==1
                for enabled,key in ((False,'fresh_seconds'),(True,'reuse_seconds')):
                    rows=[row for run in bench['runs'] if run['measured'] and run['reuse']==enabled for row in run['routes'] if row['ship']==ship]
                    assert len(rows)==3 and statistics.median(x['seconds'] for x in rows)==bench['medians'][str(ship)][key]
                    packing[key]=statistics.median(x['telemetry']['completion_pack_seconds'] for x in rows)
                    counts[key]=sorted({(x['telemetry']['completion_catalogue_hashes'],x['telemetry']['completion_catalogue_hash_reuses']) for x in rows})
                medians[ship]=dict(**bench['medians'][str(ship)],packing=packing,hash_counts=counts,plans_sha256=next(iter(hashes)))
            comparisons[version]=medians
        for ship in (10,21):assert comparisons['v816'][ship]['plans_sha256']==comparisons['v820'][ship]['plans_sha256']
        bonus=[float(line.split()[0]) for line in tar.extractfile('runtime/bonus_coefficients.txt').read().decode().splitlines() if line.strip()];assert len(bonus)==60000
        results={}
        for version in ('v815','v817'):
            r=read(version+'/report.json');path=version+'/fleet/Result.txt'
            assert r['qualified'] and r['independent']['ok'] and r['official']['ok'] and manifest[path]['sha256']==r['solution_sha256']
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
            assert pending is None and abs(math.fsum(weighted)-r['score_kg'])<1e-7 and abs(math.fsum(mass)-r['independent']['total_mass_kg'])<1e-7
            limit=min(100.,2*math.exp(.004*math.fsum(mass)/len(ships)));assert len(ships)<=limit+1e-9
            results[version]=dict(weighted_kg=math.fsum(weighted),raw_kg=math.fsum(mass),ships=len(ships),asteroids=len(asteroids),ship_limit=limit)
        assert results['v817']['weighted_kg']>=13023.704900978004-1e-7
        solves=[json.loads(line) for line in tar.extractfile('v815/solves.jsonl').read().decode().splitlines()];assert len(solves)==read('v815/report.json')['native_solves']==147
        output[side]=dict(verified_members=len(manifest),comparisons=comparisons,results=results,native_status_counts=dict(collections.Counter(x.get('status','error') for x in solves)))
print(json.dumps(output,indent=2))
