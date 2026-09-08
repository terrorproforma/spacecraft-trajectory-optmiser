from pathlib import Path
import hashlib,json,shutil,statistics,tarfile
p=Path('build/performance');remote=p/'retrieved-fast-root-v589'
dest=Path('results/lambda/2026-09-08/gpu-fast-lambert-v589')
read=lambda path:json.loads(path.read_text())
summary=dict(scope='One full-catalogue ship; ABBA two runs per mode, identical binary per GPU. Complete process includes search, CUDA refinement and official/independent verification.',campaigns={},microbenchmarks={},final_validation={},incumbent_weighted_kg=12805.194102488575)
for gpu,root,micro,final,validation in [
 ('RTX5090',p/'fast-root-campaign-v583',p/'fast-root-v585',p/'fast-root-campaign-v587',p/'fast-root-v587'),
 ('H100',remote/'campaign-experiment',remote/'microbenchmark',remote/'campaign-final',remote/'validation-final')]:
 r=read(root/'report.json');assert r['complete'] and not r.get('error')
 times={mode:[c['process_seconds'] for c in r['campaigns'] if c['candidate']==flag] for mode,flag in [('baseline',False),('candidate',True)]}
 medians={mode:statistics.median(v) for mode,v in times.items()}
 cases=[]
 for c in r['campaigns']:
  result=read(root/c['name']/'output/run_report.json')
  assert result['best']['accepted'] and result['best']['official']['ok'] and result['best']['independent']['ok']
  assert c['screening']['completed_branch_requests']==45188558
  assert c['screening']['completed_collection_options']==2782091
  assert abs(c['score']-548.2546201232)<1e-6
  cases.append(dict(name=c['name'],process_seconds=c['process_seconds'],cli_seconds=c['cli_seconds'],weighted_score_kg=c['score']))
 summary['campaigns'][gpu]=dict(cases=cases,seconds=times,medians=medians,less_time_percent=100*(1-medians['candidate']/medians['baseline']),speedup=medians['baseline']/medians['candidate'],logical_branches=45188558,collection_options=2782091,core_sha256=r['core_sha256'])
 b=read(micro/'benchmark.json');assert b['complete'];rows=[]
 for count in sorted(set(x['count'] for x in b['rows'])):
  selected=[x for x in b['rows'] if x['count']==count]
  a=statistics.median(x['median_seconds'] for x in selected if not x['candidate'])
  c=statistics.median(x['median_seconds'] for x in selected if x['candidate'])
  rows.append(dict(count=count,baseline_seconds=a,candidate_seconds=c,speedup=a/c,hops_per_second=count/c))
 summary['microbenchmarks'][gpu]=dict(scope=b['timing_scope'],rows=rows)
 v=read(validation/'report.json');assert v['complete'] and not v.get('error')
 assert '46 passed' in (validation/'pytest.log').read_text()
 f=read(final/'report.json');assert f['complete'] and not f.get('error')
 result=read(final/'default/output/run_report.json');assert result['best']['accepted'] and result['best']['official']['ok'] and result['best']['independent']['ok']
 summary['final_validation'][gpu]=dict(tests=46,lambert_sanitizers=['memcheck','synccheck','racecheck'],core_sha256=v['core_sha256'],source_sha256=v['source_sha256'],campaign=f['campaigns'][0])
 for name in ['cpp/cuda/src/orbitweaver_gpu.cu','tests/test_gtoc12_gpu_fast_lambert_root.py']:
  assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==v['source_sha256'][name],name
files={}
for group in ['host-profile-v579','host-profile-v580','fast-root-v581','fast-root-v582','fast-root-campaign-v583','fast-root-v585','fast-root-v587','fast-root-campaign-v587']:
 for path in sorted((p/group).rglob('*')):
  if path.is_file() and '__pycache__' not in path.parts:files[group+'/'+path.relative_to(p/group).as_posix()]=path
for version in [581,587]:files[f'binaries/core-v{version}.so']=Path(f'/home/angus/build-spacepdhcg-fast-root-v{version}/final/libspacepdhcg_cuda.so')
files['binaries/libqoco.so']=Path('/home/angus/build-qoco-scaled-pool-v540/final/libqoco.so')
for pattern in ['*fast_root*.py','*host_profile_v579.py','*host_profile_v580.py','profile_campaign_host*.py','solver_phase_details.py']:
 for path in p.glob(pattern):files['workers/'+path.name]=path
manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
(dest/'local-archive-manifest.json').write_text(json.dumps(manifest,indent=2))
archive=dest/'local-raw.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for name,path in files.items():t.add(path,arcname=name,recursive=False)
with tarfile.open(archive,'r:gz') as t:
 for m in t.getmembers():assert m.isfile() and hashlib.sha256(t.extractfile(m).read()).hexdigest()==manifest[m.name]
summary['local_archive']=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),verified_members=len(manifest))
summary['profiling']='v579 cProfile attribution overcounts elapsed time around HiGHS callbacks and is invalid. v580 disables profiling around separately timed linprog calls; both clocks pass the elapsed-time consistency check. Those instrumented runs locate work and are not speed benchmarks.'
summary['limitations']='No new best fleet or leaderboard submission. CPU orchestration and symbolic setup remain. Sanitizer passes cover Lambert only; previously recorded full-QOCO/cuDSS sanitizer failures remain unresolved. H100 campaign timing ranges overlap.'
(dest/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['archive_fast_root_v589.py','collect_fast_root_v589.py','publish_fast_root_v589.py','benchmark_fast_root.py']:
 shutil.copy2(p/name,dest/name)
(dest/'.gitattributes').write_text('* -text whitespace=cr-at-eol,-blank-at-eof\n')
print(json.dumps({gpu:summary['campaigns'][gpu] for gpu in summary['campaigns']},indent=2))
