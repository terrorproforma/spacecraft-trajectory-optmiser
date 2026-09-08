from pathlib import Path
import hashlib,json,shutil,statistics
local=Path('/home/angus/spacepdhcg-joint-selection-v630');remote=Path('build/performance/retrieved-joint-v642');dest=Path('results/lambda/2026-09-09/gpu-joint-itinerary-v642')
read=lambda p:json.loads(p.read_text())
summary=dict(scope='Batched joint search and compact GPU winner selection. Component timings are search-surrogate timings. Full-fleet replays include native SCvx, CPU physics verification and IO.',base_commit='bd944af935e5b71e50ef7ff4d34ea7674602832a',gpus={})
for gpu,root in [('RTX5090',local),('H100',remote)]:
    r=read(root/'campaign-v636/report.json');assert r['complete']
    runs=r['runs']
    if gpu=='RTX5090':
        assert len(runs)==3 and "('baseline1', 1)" in r['exception']
        retry=read(root/'campaign-v639/report.json');assert retry['success'];runs=runs+retry['runs']
    else:assert r['success']
    assert len(runs)==4
    times={str(mode):[v['process_seconds'] for v in runs if v['selection']==mode] for mode in (0,1)}
    medians={mode:statistics.median(values) for mode,values in times.items()}
    for run in runs:
        assert run['best']['official']['ok'] and run['best']['independent']['ok']
        assert abs(run['best']['score_kg']-12810.135953048577)<1e-6
        assert run['screening']['completed_branch_requests']==106024898 and run['screening']['completed_joint_evaluations']==23142
        assert run['native_solves']==36
        expected=88576 if run['selection'] else 15359152
        assert run['screening']['joint_result_download_bytes']==expected
    check=read(root/'compatibility-validation-v640/report.json');assert check['returncode']==0
    assert '62 passed' in (root/'compatibility-validation-v640/pytest.log').read_text()
    bench=read(root/'benchmark-v634/scalar-vs-batched.json');assert bench['complete'] and bench['correctness_passed']
    selection=read(root/'benchmark-v634/selection.json');assert selection['complete'] and all(v['exact_winner_parity'] for v in selection['cases'])
    entry=dict(core_sha256=r['core_sha256'],qoco_sha256=r['qoco_sha256'],final_wrapper_sha256=check['python_sha256'],tests=62,campaign_runs=runs,process_seconds=times,median_process_seconds=medians,less_time_percent=100*(1-medians['1']/medians['0']),component_cases=[])
    for ship in bench['ships']:
        for case in ship['cases']:
            entry['component_cases'].append(dict(ship=ship['ship'],cache=case['cache_condition'],candidates=case['candidate_count'],speedup=case['speedup_A_over_B'],scalar_seconds=case['timings']['A']['median_seconds'],gpu_seconds=case['timings']['B']['median_seconds'],gpu_candidates_per_second=case['candidate_count']/case['timings']['B']['median_seconds']))
    entry['selection_cases']=[{k:case[k] for k in ('ship','cache','candidates','visits','median_seconds','speedup','exact_winner_parity')} for case in selection['cases']]
    summary['gpus'][gpu]=entry
    out=dest/gpu;out.mkdir(exist_ok=False)
    for path in ('report.json','benchmark-v634/report.json','benchmark-v634/scalar-vs-batched.json','benchmark-v634/selection.json','compatibility-validation-v640/report.json','compatibility-validation-v640/pytest.log','compatibility.patch','campaign-v636/report.json'):
        target=out/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/path,target)
    print(gpu,times,entry['less_time_percent'])
summary['limitations']=['Joint search remains opt-in. Python metadata, epoch construction and cache handling remain.','Final wrapper includes a subsequently tested optional-ABI compatibility guard; archived timings pin the earlier wrapper.','Only two full processes per mode; RTX last baseline was retried after a GPU lock conflict. No reliable overall speedup is established.','Sanitizer successes apply to the joint native probes, not all QOCO/cuDSS code.','Replays reproduce the v595 score improvement; differences below 1e-6 kg are not new records.']
summary['fleet']=dict(ships=23,mined_asteroids=195,deployed_asteroids=196,raw_kg=14051.854893908598,weighted_kg=12810.135953048577)
summary['archives']={gpu:read(dest/(gpu+'-raw.tar.record.json')) for gpu in ('local','h100')}
(dest/'summary.json').write_text(json.dumps(summary,indent=2))
workers=dest/'workers';workers.mkdir(exist_ok=False)
for path in Path('build/performance').glob('*joint*v6*.py'):shutil.copy2(path,workers/path.name)
shutil.copy2('build/performance/prepare_joint_viewer_v642.mjs',workers/'prepare_joint_viewer_v642.mjs')
viewer=Path('results/lambda/2026-09-06/visualiser/data/gtoc12-joint-v642')
shutil.copytree(viewer,dest/'viewer-dataset')
manifest={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dest.rglob('*') if p.is_file() and p.name!='sha256.json'}
(dest/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('Published files:',len(manifest))
