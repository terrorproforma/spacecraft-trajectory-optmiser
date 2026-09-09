"""Audit exact runtime bytes, candidate equivalence and paired search timings."""
from pathlib import Path
import hashlib,json,math,statistics,tarfile
root=Path(__file__).resolve().parent
if not (root/'h100.tar.gz').exists():root=Path('results/lambda/2026-09-09/gpu-beam-expansion-v835')
def compare(a,b):
    if isinstance(a,float) and isinstance(b,(float,int)):
        assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-9,(a,b)
        return abs(a-b)
    assert type(a)==type(b),(type(a),type(b))
    if isinstance(a,dict):
        assert a.keys()==b.keys();return max((compare(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        assert len(a)==len(b);return max((compare(x,y) for x,y in zip(a,b)),default=0.)
    assert a==b,(a,b);return 0.
output={}
for side in ('local','h100'):
    archive=root/(side+'.tar.gz');receipt=json.loads((root/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('FILES.json'));names=[m.name for m in tar.getmembers()]
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,row in manifest.items():
            data=tar.extractfile(name).read();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
        def read(name):return json.load(tar.extractfile(name))
        final=read('runtime/source-manifest.json')['files']
        for name,digest in final.items():assert manifest['runtime/repo/'+name]['sha256']==digest,name
        for version in ('v830','v833'):
            for name,digest in read(version+'/source-manifest.json')['files'].items():
                prefix='runtime/repo/' if final[name]==digest else version+'/original/'
                assert manifest[prefix+name]['sha256']==digest,name
                if name.startswith('cpp/'):assert final[name]==digest
        initial=read('v830/report.json');assert initial['complete'] and not initial['success']
        for version in ('v833','v834'):
            tests=read(version+'/report.json');assert tests['complete'] and tests['success']
            assert all(s['code']==0 for s in tests['stages'])
            assert tests['core_sha256']==initial['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
        assert '183 passed' in tar.extractfile('v834/pytest.log').read().decode()
        for mode in ('memcheck','racecheck','synccheck'):
            log=tar.extractfile('v834/'+mode+'.log').read().decode();assert '45 passed' in log
            assert ('0 hazards' if mode=='racecheck' else 'ERROR SUMMARY: 0 errors') in log
        leak=read('v836/report.json');assert leak['complete'] and leak['success'] and leak['returncode']==0
        assert leak['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
        assert leak['source_manifest_sha256']==manifest['runtime/source-manifest.json']['sha256']
        log=tar.extractfile('v836/leakcheck.log').read().decode()
        assert '7 passed' in log and 'ERROR SUMMARY: 0 errors' in log
        assert 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in log
        bench=read('v831/report.json');assert bench['complete'] and bench['success'] and len(bench['runs'])==8
        assert bench['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
        assert bench['source_manifest_sha256']==manifest['v833/source-manifest.json']['sha256']
        comparison={}
        for ship,candidates,valid,materialized in ((10,517,621850,10123),(21,472,630066,95632)):
            reference=read(f'v831/warm-fresh/ship-{ship:02d}/plans.json')
            hashes={False:set(),True:set()};maximum=0.
            for run in bench['runs']:
                row=next(x for x in run['routes'] if x['ship']==ship)
                name=f"v831/{run['name']}/ship-{ship:02d}/plans.json"
                digest=manifest[name]['sha256'];assert row['plans_sha256']==digest;hashes[run['reuse']].add(digest)
                difference=compare(reference,read(name));assert difference==row['max_numeric_difference'];maximum=max(maximum,difference)
                t=row['telemetry'];assert row['candidates']==candidates and row['expansions']==513
                if run['reuse']:
                    assert t['expansion_depths']==9 and t['expansion_input_slots']==1969920
                    assert t['expansion_valid_children']==valid and t['expansion_materialized_children']==materialized
                else:assert t.get('expansion_depths',0)==0
            assert all(len(values)==1 for values in hashes.values())
            assert next(iter(hashes[False]))==manifest[f'v829/ship-{ship:02d}/plans.json']['sha256']
            ranges={};rates={}
            for enabled,key in ((False,'fresh_seconds'),(True,'reuse_seconds')):
                rows=[row for run in bench['runs'] if run['measured'] and run['reuse']==enabled for row in run['routes'] if row['ship']==ship]
                assert len(rows)==3 and statistics.median(x['seconds'] for x in rows)==bench['medians'][str(ship)][key]
                ranges[key]=[min(x['seconds'] for x in rows),max(x['seconds'] for x in rows)]
                rates[key]=candidates/bench['medians'][str(ship)][key]
            comparison[ship]=dict(**bench['medians'][str(ship)],seconds_range=ranges,candidates_per_second=rates,
                maximum_numeric_difference=maximum,valid_children=valid,materialized_children=materialized,
                plans_sha256={str(k):next(iter(v)) for k,v in hashes.items()})
        wide=read('wide/report.json');assert wide['complete'] and wide['success']
        assert wide['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
        assert wide['source_manifest_sha256']==manifest['runtime/source-manifest.json']['sha256']
        wider=[]
        for ship,count in ((10,1020),(21,940)):
            r=read(f'wide/ship-{ship:02d}/report.json');plans=read(f'wide/ship-{ship:02d}/plans.json')
            assert r['settings']['beam_width']==r['settings']['max_per_first']==128
            assert r['candidates']==len(plans)==count and r['expansions']==1025
            before={tuple(p['asteroids']) for p in read(f'v831/warm-reuse/ship-{ship:02d}/plans.json')}
            after={tuple(p['asteroids']) for p in plans}
            wider.append(dict(ship=ship,candidates=count,seconds=r['search_seconds'],branches=r['lambert_evaluations'],
                best_weighted_proxy_kg=r['best_weighted_kg'],distinct_orders=len(after),new_orders=len(after-before)))
        output[side]=dict(verified_members=len(manifest),comparison=comparison,wide=wider)
print(json.dumps(output,indent=2))
