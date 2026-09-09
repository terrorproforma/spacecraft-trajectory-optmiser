from pathlib import Path
root=Path(__file__).resolve().parent
script=(root/'run_family_retiming_v854.py').read_text()
script=script.replace('fixed_requests=2','fixed_requests=3')
start=script.index("protocol='");end=script.index("')\ndef save",start)
script=script[:start]+"protocol='Use the +30-day deployment schedule from v854 rank 1 and its 16 certified flights. Test original Earth return +60, +120, +180 days; cold CUDA initialization for each changed return. Cargo fixed. Exact saved boundary/settings prefix reuse, fresh flight certificates. Stop at first gain passing both full-fleet checkers; no changed physics tolerance.'"+script[end+1:]
start=script.index("    original=json.loads(");end=script.index("    write(ROOT/'requests.json'",start)
script=script[:start]+'''    old_refine=HOME/'spacepdhcg-family-retiming-v854'
    old_report=json.loads((old_refine/'report.json').read_text())
    assert old_report['complete'] and old_report['success']
    assert old_report['attempts'][1]['certified_legs']==16 and old_report['attempts'][1]['legs']==17
    assert dataclasses.asdict(settings)==old_report['settings']
    original=json.loads((old_refine/'requests.json').read_text())[1]['plan']
    unit=[];jobs=[]
    for rank,delay in enumerate((60.,120.,180.)):
        summary=copy.deepcopy(original)
        summary['legs'][-1]['tf']+=delay
        summary['earth_return_epoch']+=delay
        assert summary['legs'][-1]['to']==0 and summary['earth_return_epoch']<=p.C.MISSION_END_MJD
        unit.append(summary)
        cargo={int(k):v for k,v in summary['collected_mass_kg'].items()}
        request=FixedCargoRequest.from_summary(summary,cargo)
        jobs.append((rank,RoutePlan.from_summary(summary),request.sha256))
    assert len({j[2] for j in jobs})==3
    report['return_policy']=dict(arrivals_mjd=[u['earth_return_epoch'] for u in unit],
        prefix_source='v854 rank 1, 16 certified flights',cargo_unchanged=True,proxy_costs_recomputed=False)
''' + script[end:]
start=script.index('    cache={}\n');end=script.index("    with (HOME/'.spacepdhcg-gpu.lock')",start)
script=script[:start]+'''    cache={}
    prefix_dirs=[HOME/'spacepdhcg-family-refinement-v851'/f'solve-{i:03d}' for i in (0,1)]
    # v854 rank 0 performs solves 0..14; rank 1 performs 15..29.
    # Rank 1 flights 2..15 therefore map to native solves 15..28.
    prefix_dirs.extend(old_refine/f'solve-{i:03d}' for i in range(15,29))
    for number,base in enumerate(prefix_dirs):
        boundary=json.loads((base/'boundary.json').read_text())
        for k in ('departure_position','departure_velocity','arrival_position','arrival_velocity'):boundary[k]=np.asarray(boundary[k])
        boundary=p.LegBoundary(**boundary)
        fields=json.loads((base/'solution.json').read_text())
        assert fields['status'] in ('converged','iteration_limit') and fields['max_defect']<=settings.defect_tolerance
        record=json.loads((old_refine/f'rank-001/leg-{number:02d}.json').read_text())
        assert record['certified'] and record['departure']==boundary.departure_epoch and record['arrival']==boundary.arrival_epoch
        assert record['initial_mass_kg']==boundary.initial_mass
        z=np.load(base/'raw.npz')
        fields.update(boundary=boundary,node_epochs_mjd=z['epochs'],thrust_n=z['thrust'],states_scaled=z['states'],
            departure_vinf_km_s=z['departure_vinf'],arrival_vinf_km_s=z['arrival_vinf'])
        sol=p.LegSolution(**fields)
        encoded=json.dumps(dataclasses.asdict(boundary),sort_keys=True,default=lambda v:v.tolist(),allow_nan=False).encode()
        key=sha(encoded+json.dumps(dataclasses.asdict(settings),sort_keys=True).encode())
        cache[key]=(str(base),sol)
    assert len(cache)==16
''' + script[end:]
script=script.replace("report['attempts'].append(entry);save()", "report['attempts'].append(entry);save()\n                if entry.get('verified_gain'):break")
compile(script,'run.py','exec');(root/'run_family_returns_v855.py').write_text(script)
launch=(root/'launch_family_refinement_v851.py').read_text().replace('run_family_refinement_v851.py','run_family_returns_v855.py').replace('spacepdhcg-family-refinement-v851','spacepdhcg-family-returns-v855').replace('family-refinement-launch-v851.json','family-returns-launch-v855.json')
(root/'launch_family_returns_v855.py').write_text(launch)
