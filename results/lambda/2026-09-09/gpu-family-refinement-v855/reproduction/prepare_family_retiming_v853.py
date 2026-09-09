from pathlib import Path
root=Path(__file__).resolve().parent
script=(root/'run_family_refinement_v851.py').read_text()
script=script.replace('fixed_requests=3','fixed_requests=2')
old="protocol='Ranks 0, 1, 2 from ship-18 unit arm; duplicate bonus requests excluded. Fixed cargo and epochs. 40+4 outer updates with existing conditioning recovery. Exact boundary/settings cache only; every route retains independent flight certification. No promotion without both full-fleet checkers. 600-second campaign budget before each new leg.'"
new="protocol='Two distinct rank-0 schedules: third flight +15 or +30 days; shift intervening deployments until the first camp, shortening that camp. Reduce five affected asteroid cargo prescriptions by actual lost mining time. Original 10000 penalty and physics gates. Replay only the two exact saved successful prefix solves; fresh certification and native solves for changed legs. 600-second campaign budget; both full-fleet checkers mandatory.'"
assert old in script;script=script.replace(old,new)
start=script.index("    unit=json.loads(")
end=script.index("    write(ROOT/'requests.json'",start)
script=script[:start]+'''    original=json.loads((PRIOR/'ship-18-unit/plans.json').read_text())[0]
    unit=[];jobs=[]
    old_refine=HOME/'spacepdhcg-family-refinement-v851'
    old_report=json.loads((old_refine/'report.json').read_text())
    assert old_report['complete'] and old_report['success']
    assert dataclasses.asdict(settings)==old_report['settings']
    for rank,delay in enumerate((15.,30.)):
        summary=copy.deepcopy(original);changed=[];flights=0
        for leg in summary['legs']:
            if leg['role']=='camp':
                if changed:
                    leg['t0']+=delay
                    assert leg['t0']<leg['tf']
                    break
                continue
            flights+=1
            if flights<3:continue
            if flights>3:leg['t0']+=delay
            leg['tf']+=delay
            body=str(leg['to'])
            assert body in summary['deploy_epochs']
            summary['deploy_epochs'][body]+=delay
            summary['collected_mass_kg'][body]-=delay*p.C.MINING_RATE_KG_PER_YEAR/p.C.YEAR_DAYS
            changed.append(body)
        assert len(changed)==5
        summary['total_collected_kg']=sum(summary['collected_mass_kg'].values())
        gain=summary['total_collected_kg']-594.7980835045999
        assert gain>0.1
        unit.append(summary)
        cargo={int(k):v for k,v in summary['collected_mass_kg'].items()}
        request=FixedCargoRequest.from_summary(summary,cargo)
        jobs.append((rank,RoutePlan.from_summary(summary),request.sha256))
    assert len({j[2] for j in jobs})==2
    report['timing_policy']=dict(delays_days=[15.,30.],shortened_camp_asteroid=1122,
        changed_deployments=[8846,49900,37385,8123,1122],proxy_costs_recomputed=False,
        proxy_cost_fields='Inherited annotations only; acceptance uses fresh low-thrust refinement with exact new cargo.')
''' + script[end:]
old='    cache={}\n'
new='''    cache={}
    for number in (0,1):
        base=old_refine/f'solve-{number:03d}'
        boundary=json.loads((base/'boundary.json').read_text())
        for k in ('departure_position','departure_velocity','arrival_position','arrival_velocity'):boundary[k]=np.asarray(boundary[k])
        boundary=p.LegBoundary(**boundary)
        fields=json.loads((base/'solution.json').read_text())
        assert fields['status'] in ('converged','iteration_limit') and fields['max_defect']<=settings.defect_tolerance
        z=np.load(base/'raw.npz')
        fields.update(boundary=boundary,node_epochs_mjd=z['epochs'],thrust_n=z['thrust'],states_scaled=z['states'],
            departure_vinf_km_s=z['departure_vinf'],arrival_vinf_km_s=z['arrival_vinf'])
        sol=p.LegSolution(**fields)
        encoded=json.dumps(dataclasses.asdict(boundary),sort_keys=True,default=lambda v:v.tolist(),allow_nan=False).encode()
        key=sha(encoded+json.dumps(dataclasses.asdict(settings),sort_keys=True).encode())
        cache[key]=(f'v851/solve-{number:03d}',sol)
'''
assert old in script;script=script.replace(old,new)
compile(script,'run.py','exec')
(root/'run_family_retiming_v853.py').write_text(script)
launch=(root/'launch_family_refinement_v851.py').read_text().replace('run_family_refinement_v851.py','run_family_retiming_v853.py')
launch=launch.replace("root=home/'spacepdhcg-family-refinement-v851'", "root=home/'spacepdhcg-family-retiming-v853'")
launch=launch.replace('family-refinement-launch-v851.json','family-retiming-launch-v853.json')
(root/'launch_family_retiming_v853.py').write_text(launch)
