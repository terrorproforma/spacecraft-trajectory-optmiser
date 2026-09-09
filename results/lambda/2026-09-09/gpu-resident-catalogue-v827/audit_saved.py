"""Validate every saved byte and recompute paired medians/counters."""
from pathlib import Path
import hashlib,json,statistics,tarfile
root=Path(__file__).resolve().parent
if not (root/'h100.tar.gz').exists():root=Path('results/lambda/2026-09-09/gpu-resident-catalogue-v827')
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
        for name,digest in read('v823/source-manifest.json')['files'].items():
            prefix='runtime/repo/' if final[name]==digest else 'v823/original/'
            assert manifest[prefix+name]['sha256']==digest,name
            if name.startswith('cpp/'):assert final[name]==digest
        initial=read('v823/report.json');assert initial['complete'] and not initial['success']
        tests=read('v826/report.json');assert tests['complete'] and tests['success']
        assert all(s['code']==0 for s in tests['stages'])
        assert tests['core_sha256']==initial['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
        assert '165 passed' in tar.extractfile('v826/pytest.log').read().decode()
        for mode in ('memcheck','racecheck','synccheck'):
            log=tar.extractfile('v826/'+mode+'.log').read().decode();assert '36 passed' in log
            if mode=='racecheck':assert '0 hazards' in log
            else:assert 'ERROR SUMMARY: 0 errors' in log
        bench=read('v824/report.json');assert bench['complete'] and bench['success'] and len(bench['runs'])==8
        leak=read('v828/report.json');assert leak['complete'] and leak['success'] and leak['returncode']==0
        assert leak['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
        assert leak['source_manifest_sha256']==manifest['runtime/source-manifest.json']['sha256']
        leaklog=tar.extractfile('v828/leakcheck.log').read().decode()
        assert '4 passed' in leaklog and 'ERROR SUMMARY: 0 errors' in leaklog
        assert 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in leaklog
        assert bench['core_sha256']==manifest['runtime/libspacepdhcg_cuda.so']['sha256']
        assert bench['source_manifest_sha256']==manifest['runtime/source-manifest.json']['sha256']
        comparison={}
        for ship,rebuilds,candidates in ((10,147,517),(21,209,472)):
            hashes=set();packing={};counts={};rates={};ranges={}
            for run in bench['runs']:
                row=next(x for x in run['routes'] if x['ship']==ship)
                digest=manifest[f"v824/{run['name']}/ship-{ship:02d}/plans.json"]['sha256'];assert row['plans_sha256']==digest;hashes.add(digest)
                t=row['telemetry'];assert row['candidates']==candidates and t['completion_model_rebuilds']==rebuilds
                uploads=1 if run['reuse'] else rebuilds
                assert t['completion_catalogue_uploads']==uploads
                assert t['completion_resident_catalogue_reuses']==rebuilds-uploads
                assert t['completion_catalogue_upload_bytes']==uploads*60000*40
                assert t['completion_catalogue_hashes']==2
            assert len(hashes)==1
            for enabled,key in ((False,'fresh_seconds'),(True,'reuse_seconds')):
                rows=[row for run in bench['runs'] if run['measured'] and run['reuse']==enabled for row in run['routes'] if row['ship']==ship]
                assert len(rows)==3 and statistics.median(x['seconds'] for x in rows)==bench['medians'][str(ship)][key]
                packing[key]=statistics.median(x['telemetry']['completion_pack_seconds'] for x in rows)
                counts[key]={k:rows[0]['telemetry'][k] for k in ('completion_model_rebuilds','completion_catalogue_uploads','completion_resident_catalogue_reuses','completion_catalogue_upload_bytes')}
                rates[key]=candidates/statistics.median(x['seconds'] for x in rows)
                ranges[key]=[min(x['seconds'] for x in rows),max(x['seconds'] for x in rows)]
            comparison[ship]=dict(**bench['medians'][str(ship)],packing=packing,counts=counts,candidates_per_second=rates,seconds_range=ranges,plans_sha256=next(iter(hashes)))
        output[side]=dict(verified_members=len(manifest),comparison=comparison)
print(json.dumps(output,indent=2))
