"""Portable byte, source, work-count and paired-timing audit; no GPU execution."""
from pathlib import Path
import hashlib,json,marshal,re,statistics,tarfile

ROOT=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
reference=json.loads((ROOT/'reference.json').read_text())
result={};sources=[]
for side in ('local','h100'):
    manifest=json.loads((ROOT/(side+'-manifest.json')).read_text())
    assert sha((ROOT/(side+'.tar.gz')).read_bytes())==manifest['archive_sha256']
    payload={}
    with tarfile.open(ROOT/(side+'.tar.gz'),'r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in manifest['files'] and member.name not in payload
            raw=tar.extractfile(member).read();item=manifest['files'][member.name]
            assert len(raw)==item['bytes'] and sha(raw)==item['sha256'],member.name
            payload[member.name]=raw
    assert set(payload)==set(manifest['files'])
    def read(name):return json.loads(payload[name])
    runtime=read('runtime/report.json');bench=read('benchmark/report.json')
    source=read('runtime/source-manifest.json');sources.append(source)
    assert runtime['complete'] and runtime['success'] and bench['complete'] and bench['success']
    assert all(s['code']==0 for s in runtime['stages'])
    for path,digest in source['files'].items():assert sha(payload['runtime/repo/'+path])==digest,path
    assert sha(payload['runtime/source-manifest.json'])==bench['source_manifest_sha256']
    assert sha(payload['runtime/libspacepdhcg_cuda.so'])==runtime['core_sha256']==bench['core_sha256']
    assert re.search(r'\b236 passed\b',payload['runtime/pytest.log'].decode())
    for mode in ('memcheck','racecheck','synccheck'):
        log=payload['runtime/'+mode+'.log'].decode()
        assert re.search(r'\b98 passed\b',log)
        assert ('0 errors' in log or '0 hazards displayed (0 errors, 0 warnings)' in log)
        assert '0 errors' in payload['runtime/'+mode+'-controller.log'].decode()
    leaks=payload['runtime/leaks.log'].decode()
    assert '31 passed' in leaks and '0 bytes leaked in 0 allocations' in leaks and '0 errors' in leaks
    assert read('terminal.json')['all_owned_workers_exited']
    assert len(bench['runs'])==8 and sum(r['measured'] for r in bench['runs'])==6
    comparison={}
    for ship in (10,21):
        by_mode={False:[],True:[]};hashes=set()
        for run in bench['runs']:
            directory=f"benchmark/{run['name']}/ship-{ship:02d}/"
            raw=payload[directory+'plans.json'];hashes.add(sha(raw))
            row=read(directory+'report.json')
            indexed=next(x for x in run['routes'] if x['ship']==ship)
            assert sha(raw)==indexed['plans_sha256']
            assert len(json.loads(raw))==row['candidates']==indexed['candidates']
            assert row['search_seconds']==indexed['seconds'] and indexed['max_numeric_difference']==0
            assert row['gpu_telemetry']==indexed['telemetry']
            if run['measured']:by_mode[run['reuse']].append(indexed)
        assert len(hashes)==1 and hashes=={reference['candidate_sha256'][side][str(ship)]}
        modes={}
        for enabled,rows in by_mode.items():
            assert len(rows)==3
            seconds=[x['seconds'] for x in rows]
            stable=['completed_collection_dp_passes','collect_dp_allocations','collect_dp_rebinds']
            if enabled:stable+=['native_collection_plans','collection_plan_metadata_upload_bytes','collection_plan_result_download_bytes']
            counters={key:rows[0]['telemetry'][key] for key in stable}
            for row in rows:assert all(row['telemetry'][key]==value for key,value in counters.items())
            if enabled:assert counters['collection_plan_result_download_bytes']==counters['native_collection_plans']*656
            modes['native' if enabled else 'host']=dict(seconds=statistics.median(seconds),range=[min(seconds),max(seconds)],
                candidates_per_second=rows[0]['candidates']/statistics.median(seconds),counters=counters)
        assert modes['host']['counters']['completed_collection_dp_passes']==modes['native']['counters']['completed_collection_dp_passes']
        expected=bench['medians'][str(ship)]
        speedup=modes['host']['seconds']/modes['native']['seconds']
        assert expected['fresh_seconds']==modes['host']['seconds'] and expected['reuse_seconds']==modes['native']['seconds'] and expected['speedup']==speedup
        modes['host']['result_download_bytes']=modes['host']['counters']['completed_collection_dp_passes']*632
        comparison[ship]=dict(**modes,speedup=speedup,plans_sha256=next(iter(hashes)))
    saved=dict(verified_members=len(payload),archive_sha256=manifest['archive_sha256'],core_sha256=runtime['core_sha256'],
        source_manifest_sha256=bench['source_manifest_sha256'],comparison=comparison,
        tests=236,sanitizer_tests_per_tool=98,leak_tests=31,gpu_context=bench.get('gpu_context'))
    if side=='h100':
        profiles={}
        for ship in (10,21):
            stats=marshal.loads(payload[f'profile/ship-{ship:02d}.pstats'])
            assert not any(k[2]=='_solve_collect_dp' for k in stats)
            rows=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in sorted(stats.items(),key=lambda x:-x[1][3])[:40]]
            assert rows==read('profiles.json')[str(ship)]
            profiles[ship]=dict(top_cumulative=rows[:12],total_calls=sum(v[1] for v in stats.values()),total_self_seconds=sum(v[2] for v in stats.values()))
            assert sha(payload[f'profile/ship-{ship:02d}/plans.json'])==comparison[ship]['plans_sha256']
        saved['profiles']=profiles
    result[side]=saved
assert sources[0]==sources[1]
result['passed']=True
(ROOT/'saved-audit.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
print(json.dumps({side:{'verified_members':result[side]['verified_members'],'comparison':result[side]['comparison']} for side in ('local','h100')},indent=2))
