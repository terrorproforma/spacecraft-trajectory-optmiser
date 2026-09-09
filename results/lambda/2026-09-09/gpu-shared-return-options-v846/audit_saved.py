"""Portable byte, source, candidate and performance audit; no GPU required."""
from pathlib import Path
import hashlib,io,json,marshal,statistics,tarfile
root=Path(__file__).resolve().parent
if not (root/'h100.tar.gz').exists():root=Path('results/lambda/2026-09-09/gpu-shared-return-options-v846')
output={}
for side in ('local','h100'):
    archive=root/(side+'.tar.gz');receipt=json.loads((root/(side+'-receipt.json')).read_text())
    assert archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive,'r|gz') as tar:
        names=[];payload={}
        for member in tar:
            assert member.isfile();names.append(member.name);payload[member.name]=tar.extractfile(member).read()
        class Payload:
            def extractfile(self,name):return io.BytesIO(payload[name])
        tar=Payload()
        manifest=json.load(tar.extractfile('FILES.json'))
        assert len(names)==len(set(names)) and set(names)==set(manifest)|{'FILES.json'}
        for name,row in manifest.items():
            data=tar.extractfile(name).read();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],name
        def read(name):return json.load(tar.extractfile(name))
        source=read('runtime/source-manifest.json')
        for name,digest in source['files'].items():assert manifest['runtime/repo/'+name]['sha256']==digest,name
        tests=read('validation/report.json');assert tests['complete'] and tests['success']
        assert all(s['code']==0 for s in tests['stages'])
        core=manifest['runtime/libspacepdhcg_cuda.so']['sha256'];assert core==tests['core_sha256']
        assert '205 passed' in tar.extractfile('validation/pytest.log').read().decode()
        for mode in ('memcheck','racecheck','synccheck'):
            log=tar.extractfile('validation/'+mode+'.log').read().decode();assert '67 passed' in log
            assert ('0 hazards' if mode=='racecheck' else 'ERROR SUMMARY: 0 errors') in log
        leak=read('leaks/report.json');assert leak['complete'] and leak['success'] and leak['returncode']==0
        assert leak['core_sha256']==core and leak['source_manifest_sha256']==manifest['runtime/source-manifest.json']['sha256']
        log=tar.extractfile('leaks/leakcheck.log').read().decode()
        assert '29 passed' in log and '0 bytes leaked in 0 allocations' in log and 'ERROR SUMMARY: 0 errors' in log
        bench=read('benchmark/report.json');assert bench['complete'] and bench['success'] and len(bench['runs'])==8
        assert bench['core_sha256']==core and bench['source_manifest_sha256']==manifest['runtime/source-manifest.json']['sha256']
        prior=read('prior-v840-report.json');prior_hashes={r['ship']:r['plans_sha256'] for run in prior['runs'] if run['name']=='warm-reuse' for r in run['routes']}
        all_hashes={10:set(),21:set()};by_ship={}
        for run in bench['runs']:
            run_report=read('benchmark/'+run['name']+'/report.json');assert run_report['complete'] and run_report['success']
            script=tar.extractfile('benchmark/'+run['name']+'/run.py').read().decode();assert 'spacepdhcg-return-cache-v844' in script
            for row in run['routes']:
                ship=row['ship'];prefix=f"benchmark/{run['name']}/ship-{ship:02d}/"
                detail=read(prefix+'report.json');plans=read(prefix+'plans.json')
                assert manifest[prefix+'plans.json']['sha256']==row['plans_sha256']
                assert row['plans_sha256']==prior_hashes[ship]
                all_hashes[ship].add(row['plans_sha256'])
                assert row['candidates']==len(plans)==(517 if ship==10 else 472) and row['expansions']==513
                assert row['seconds']==detail['search_seconds'] and row['telemetry']==detail['gpu_telemetry']
                stats=row['telemetry'];assert stats['expansion_depths']==9 and stats['expansion_input_slots']==1969920
                assert stats['expansion_valid_children']==(621850 if ship==10 else 630066)
                assert stats['admission_depths']==9 and stats['admission_input_children']==stats['expansion_valid_children']
                assert stats['expansion_materialized_children']==stats['admission_selected']==(576 if ship==10 else 516)
                assert stats['resident_return_queries']>0
                if run['reuse']:
                    assert stats['resident_return_cache_hits']+stats['resident_return_cache_misses']==stats['resident_return_queries']
                    assert stats['resident_return_cache_hits']==stats['resident_option_shared_handles']>0
                    assert stats['cached_return_branches']>0
                else:assert 'resident_return_cache_hits' not in stats and 'cached_return_branches' not in stats
        for ship in (10,21):
            assert len(all_hashes[ship])==1
            result={}
            for flag,label in ((False,'fresh'),(True,'shared')):
                rows=[r for run in bench['runs'] if run['measured'] and run['reuse']==flag for r in run['routes'] if r['ship']==ship]
                assert len(rows)==3
                times=[r['seconds'] for r in rows];stats=rows[0]['telemetry']
                result[label]=dict(seconds=statistics.median(times),range=[min(times),max(times)],candidates_per_second=rows[0]['candidates']/statistics.median(times),return_queries=stats['resident_return_queries'],return_hits=stats.get('resident_return_cache_hits',0),return_misses=stats.get('resident_return_cache_misses',0),cached_branches=stats.get('cached_return_branches',0),completed_branches=stats['completed_branch_requests'],resident_option_builds=stats['resident_option_builds'],option_download_bytes=stats['compact_option_download_bytes'])
            result['speedup']=result['fresh']['seconds']/result['shared']['seconds']
            expected=bench['medians'][str(ship)]
            assert result['fresh']['seconds']==expected['fresh_seconds'] and result['shared']['seconds']==expected['reuse_seconds'] and result['speedup']==expected['speedup']
            a,b=result['fresh'],result['shared']
            assert a['return_queries']==b['return_queries']
            assert a['completed_branches']-b['completed_branches']==b['cached_branches']
            assert a['resident_option_builds']-b['resident_option_builds']==b['return_hits']
            assert a['option_download_bytes']-b['option_download_bytes']==4*b['return_hits']
            baseline=read(f"benchmark/warm-fresh/ship-{ship:02d}/report.json")
            for run in bench['runs']:
                row=next(r for r in run['routes'] if r['ship']==ship)
                detail=read(f"benchmark/{run['name']}/ship-{ship:02d}/report.json")
                assert baseline['lambert_evaluations']-detail['lambert_evaluations']==row['telemetry'].get('cached_return_branches',0)
            result['plans_sha256']=next(iter(all_hashes[ship]));by_ship[ship]=result
        profile=read('profile/report.json');assert profile['complete'] and profile['success']
        assert profile['core_sha256']==core and profile['source_manifest_sha256']==manifest['runtime/source-manifest.json']['sha256']
        summaries=read('profile/summary.json')
        for ship in (10,21):
            assert manifest[f'profile/ship-{ship:02d}/plans.json']['sha256']==next(iter(all_hashes[ship]))
            stats=marshal.loads(payload[f'profile/ship-{ship:02d}.pstats'])
            expected=dict(total_calls=sum(v[1] for v in stats.values()),total_seconds=sum(v[2] for v in stats.values()),top_cumulative=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],total_calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in sorted(stats.items(),key=lambda kv:kv[1][3],reverse=True)[:35]])
            assert expected==summaries[str(ship)]
        output[side]=dict(verified_members=len(manifest),core_sha256=core,source_manifest_sha256=manifest['runtime/source-manifest.json']['sha256'],comparison=by_ship,profile=summaries)
print(json.dumps(output,indent=2))
