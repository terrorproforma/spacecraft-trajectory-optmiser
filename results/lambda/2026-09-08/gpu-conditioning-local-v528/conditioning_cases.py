"""Bounded conditioning pilot; no changes to accuracy or acceptance limits."""
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import json
import os
import sys
import time

import numpy as np
from spacepdhcg.gtoc12.low_thrust import LegBoundary, ScvxSettings, solve_leg, certify_leg
from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution

root = Path(sys.argv[1])
origin, ruiz = map(int, sys.argv[2:4])
root.mkdir(parents=True, exist_ok=False)
indices = [0, 12, 44, 60, 77, 87, 156, 204]
fixture = Path('build/performance/grid-cache-fleet-v403/scvx-calls.json')
rows = json.loads(fixture.read_text())
os.environ['SPACEPDHCG_TEST_GTOC12_STATE_ORIGIN'] = str(origin)
os.environ['SPACEPDHCG_TEST_GTOC12_QOCO_POOL'] = '0'
os.environ['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY'] = '0'
os.environ['SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH'] = '0'
results = []

def save():
    temp = root / 'results.tmp'
    temp.write_text(json.dumps(results, indent=2, default=lambda v: v.tolist()))
    temp.replace(root / 'results.json')

with using_gpu_execution(SimpleNamespace(gpu_execution='graph', outer_loop_backend='cuda', workers=1)):
    for index in indices:
        row = rows[index]
        kw = dict(row['boundary'])
        for key in ['departure_position', 'departure_velocity', 'arrival_position', 'arrival_velocity']:
            kw[key] = np.array(kw[key], dtype=float)
        boundary = LegBoundary(**kw)
        settings = ScvxSettings(**row['settings'])
        # The same explicit 30-second pilot budget applies to every mode.
        # All mathematical and certificate tolerances remain the fixture's.
        settings = dataclasses.replace(settings, qoco_ruiz_iterations=ruiz, time_limit_s=30)
        start = time.perf_counter()
        solution = solve_leg(boundary, settings)
        result = dict(index=index, origin=origin, ruiz=ruiz,
                      settings=dataclasses.asdict(settings), prior_status=row['status'],
                      seconds=time.perf_counter() - start, status=solution.status,
                      diagnostic=solution.diagnostic, iterations=solution.iterations,
                      accepted=solution.accepted_iterations, history=solution.history,
                      reports=solution.solver_reports)
        if solution.converged:
            certificate = certify_leg(solution)
            result.update(certified=certificate.within_tolerance,
                          certificate=dataclasses.asdict(certificate))
            np.savez_compressed(root / f'leg-{index:03d}.npz', states=solution.states_scaled,
                                thrust=solution.thrust_n, epochs=solution.node_epochs_mjd)
        results.append(result)
        save()
        print(json.dumps({k: result[k] for k in ['index', 'origin', 'ruiz', 'seconds', 'status', 'diagnostic']}), flush=True)
(root / 'summary.json').write_text(json.dumps(dict(complete=True, cases=len(results),
    solver_seconds=sum(r['seconds'] for r in results), certified=sum(r.get('certified', False) for r in results))))
