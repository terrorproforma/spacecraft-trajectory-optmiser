"""Isolate return-inflation mass bias on 23 saved incumbent controls; no solver calls."""
from pathlib import Path
import ast
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/performance/return-mass-bias-v619'
OUT.mkdir(exist_ok=False)
(OUT / 'inputs').mkdir()
sha = lambda data: hashlib.sha256(data).hexdigest()

def literals(path, names):
    data = path.read_bytes()
    found = {}
    for node in ast.parse(data).body:
        if isinstance(node, ast.Assign):
            keys = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            keys = [node.target.id]
        else:
            continue
        for key in keys:
            if key in names:
                found[key] = ast.literal_eval(node.value)
    assert set(found) == set(names)
    return found, sha(data)

constants, constants_sha = literals(ROOT / 'src/spacepdhcg/gtoc12/constants.py',
    {'G0_M_S2', 'ISP_S', 'THRUST_MAX_N', 'DAY_S', 'MINER_MASS_KG'})
model, model_sha = literals(ROOT / 'src/spacepdhcg/gtoc12/screening.py',
    {'RETURN_INFLATION_TOF_DAYS', 'RETURN_INFLATION_P65',
     'RETURN_INFLATION_RATIO_SLOPE', 'RETURN_INFLATION_RATIO_CENTRE'})

def inflation(dv, mass, tof):
    days, values = model['RETURN_INFLATION_TOF_DAYS'], model['RETURN_INFLATION_P65']
    base = values[-1]
    if tof <= days[0]:
        base = values[0]
    else:
        for i in range(1, len(days)):
            if tof <= days[i]:
                base = values[i-1] + (values[i]-values[i-1]) / (days[i]-days[i-1]) * (tof-days[i-1])
                break
    authority = constants['THRUST_MAX_N'] / mass * 1e-3 * tof * constants['DAY_S']
    ratio = dv / max(authority, 1e-12)
    correction = 1 + model['RETURN_INFLATION_RATIO_SLOPE'] * (ratio-model['RETURN_INFLATION_RATIO_CENTRE'])
    return max(base * min(max(correction, .85), 1.2), .85)

def fuel(mass, dv):
    exhaust = constants['ISP_S'] * constants['G0_M_S2'] * 1e-3
    return mass * (1-math.exp(-dv/exhaust))

inventory_path = ROOT / 'build/performance/depth-diverse-generation-v616/inputs/incumbent-inventory.json'
inventory_bytes = inventory_path.read_bytes()
(OUT / 'inputs/incumbent-inventory.json').write_bytes(inventory_bytes)
inventory = json.loads(inventory_bytes)
assert len(inventory['routes']) == 23
rows = []
for entry in inventory['routes']:
    path = Path(entry['path'].replace('/mnt/c/', 'C:/', 1))
    data = path.read_bytes()
    assert sha(data) == entry['sha256']
    route = json.loads(data)
    (OUT / f'inputs/ship-{entry["ship"]:02}.json').write_bytes(data)
    plan = route['plan']
    deploy = max(((int(k), float(t)) for k,t in plan['deploy_epochs'].items()), key=lambda x:x[1])
    deploy_leg = [leg for leg in route['legs'] if leg['to'] == deploy[0] and leg['tf'] == deploy[1]]
    returns = [leg for leg in route['legs'] if leg['to'] == 0]
    planned_returns = [leg for leg in plan['legs'] if leg['role'] == 'earth_return']
    assert len(deploy_leg) == len(returns) == len(planned_returns) == 1
    burn, planned = returns[0], planned_returns[0]
    assert (burn['from'],burn['to'],burn['t0'],burn['tf']) == (planned['from'],planned['to'],planned['t0'],planned['tf'])
    assert burn['certified'] and deploy_leg[0]['certified'] and entry['cargo_matches_retained_Result']
    payload = sum(float(v) for v in plan['collected_mass_kg'].values())
    prefix_mass = deploy_leg[0]['mass_after'] - constants['MINER_MASS_KG']
    guessed_mass = prefix_mass + payload
    actual_mass = burn['mass_before']
    dv, tof = planned['dv_proxy_km_s'], planned['tf']-planned['t0']
    assert dv > 0 and tof > 0 and guessed_mass >= actual_mass > 0
    actual_model = inflation(dv, actual_mass, tof)
    guessed_model = inflation(dv, guessed_mass, tof)
    at_actual = fuel(actual_mass, dv*actual_model)
    at_guessed = fuel(actual_mass, dv*guessed_model)
    measured = burn['propellant_kg']
    rows.append({
        'ship': entry['ship'], 'input_sha256': sha(data),
        'independent_inventory': entry['independent_inventory'], 'archived_certification': True,
        'fixed_cargo_kg': payload, 'measured_post_deploy_mass_kg': prefix_mass,
        'builder_style_mass_using_measured_prefix_kg': guessed_mass,
        'measured_return_departure_mass_kg': actual_mass, 'mass_overestimate_kg': guessed_mass-actual_mass,
        'return_tof_days': tof, 'saved_lambert_dv_km_s': dv,
        'measured_delta_v_km_s': burn['delta_v_km_s'],
        'saved_planned_inflation': planned['inflation'],
        'measured_inflation': burn['delta_v_km_s']/dv,
        'model_at_guessed_mass': guessed_model, 'model_at_measured_mass': actual_model,
        'fuel_at_measured_mass_with_guessed_mass_inflation_kg': at_guessed,
        'fuel_at_measured_mass_with_measured_mass_inflation_kg': at_actual,
        'fuel_bias_from_mass_argument_alone_kg': at_guessed-at_actual,
        'archived_return_propellant_kg': measured,
        'model_error_even_at_measured_mass_kg': at_actual-measured,
        'saved_leg_fuel_error_kg': fuel(actual_mass,dv*planned['inflation'])-measured,
    })
report = {
    'complete': True, 'GPU_calls': 0, 'solver_calls': 0, 'fresh_lambert_evaluations': 0,
    'fleet_promotions': 0, 'fresh_physics_certifications': 0,
    'source_hashes': {'constants.py': constants_sha, 'screening.py': model_sha},
    'inventory_sha256': sha(inventory_bytes), 'constants': constants, 'model': model,
    'route_count': len(rows), 'rows': rows,
    'summary': {
        'median_mass_overestimate_kg': statistics.median(r['mass_overestimate_kg'] for r in rows),
        'median_return_fuel_bias_from_mass_argument_kg': statistics.median(r['fuel_bias_from_mass_argument_alone_kg'] for r in rows),
        'maximum_return_fuel_bias_from_mass_argument_kg': max(r['fuel_bias_from_mass_argument_alone_kg'] for r in rows),
        'median_model_error_at_measured_mass_kg': statistics.median(r['model_error_even_at_measured_mass_kg'] for r in rows),
        'positive_model_errors_at_measured_mass': sum(r['model_error_even_at_measured_mass_kg'] > 0 for r in rows),
    },
    'scope': 'Component diagnostic using archived measurements and stored Lambert values; both fuel calculations hold actual return mass fixed to isolate the inflation mass argument. This does not reproduce a full sequential proxy or newly certify any route.',
}
(OUT / 'run.py').write_bytes(Path(__file__).read_bytes())
(OUT / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'complete': True, 'summary': report['summary'], 'report_sha256': sha((OUT/'report.json').read_bytes())},indent=2))
