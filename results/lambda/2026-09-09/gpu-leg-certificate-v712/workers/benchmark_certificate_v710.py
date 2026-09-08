"""Alternating paired certificates on identical emitted real-route legs."""
from pathlib import Path
import dataclasses,json,statistics,sys,time
from types import SimpleNamespace
import numpy as np
from spacepdhcg.gtoc12.gpu_verifier import GpuVerifierSession,certify_legs_cuda,pack_solution
from spacepdhcg.gtoc12.low_thrust import DU_KM,certify_leg
from spacepdhcg.gtoc12.solution import Solution

root=Path(sys.argv[1]);output=Path(sys.argv[2])
cases=[]
for path in sorted((root/'campaign-cpu/candidates').glob('*/route/Result.txt')):
    parsed=Solution.read(path)
    *_,metadata=pack_solution(parsed)
    solutions=[]
    for sid,previous,target,burns in metadata:
        boundary=SimpleNamespace(departure_position=previous.after.position,
            departure_velocity=previous.after.velocity,initial_mass=previous.after.mass,
            departure_epoch=previous.after.epoch,arrival_epoch=target.epoch,
            arrival_position=target.before.position)
        thrust=np.concatenate([arc.interior_arrays()[1] for arc in burns]) if burns else np.zeros((1,3))
        solutions.append(SimpleNamespace(boundary=boundary,burn_arcs=lambda burns=burns:burns,
            departure_ship_velocity_km_s=lambda previous=previous:previous.after.velocity,
            arrival_ship_velocity_km_s=lambda target=target:target.before.velocity,
            states_scaled=np.array([np.r_[target.before.position/DU_KM,np.zeros(4)]]),thrust_n=thrust))
    samples=[]
    with GpuVerifierSession() as session:
        for repeat in range(5):
            records={}
            for backend in (('cpu','cuda') if repeat%2==0 else ('cuda','cpu')):
                started=time.perf_counter()
                # Match the production driver: one completed leg at a time.
                records[backend]=[certify_leg(s) if backend=='cpu' else certify_legs_cuda([s],workspace=session)[0] for s in solutions]
                samples.append(dict(repeat=repeat,backend=backend,seconds=time.perf_counter()-started,warmup=repeat==0))
            for a,b in zip(records['cpu'],records['cuda'],strict=True):
                assert a.within_tolerance==b.within_tolerance
                assert abs(a.position_error_km-b.position_error_km)<0.01
                assert abs(a.velocity_error_km_s-b.velocity_error_km_s)<1e-8
                assert abs(a.final_mass_kg-b.final_mass_kg)<1e-7
                assert abs(a.minimum_sun_distance_au-b.minimum_sun_distance_au)<1e-5
    medians={backend:statistics.median(s['seconds'] for s in samples if s['backend']==backend and not s['warmup']) for backend in ('cpu','cuda')}
    cases.append(dict(source=str(path),legs=len(solutions),samples=samples,medians=medians,
        ratio=medians['cpu']/medians['cuda'],cpu_certificates=[dataclasses.asdict(c) for c in records['cpu']],
        cuda_certificates=[dataclasses.asdict(c) for c in records['cuda']]))
output.write_text(json.dumps(dict(cases=cases),indent=2))
print(json.dumps([dict(legs=c['legs'],medians=c['medians'],ratio=c['ratio']) for c in cases]))
