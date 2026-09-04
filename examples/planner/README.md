# Planner examples

One runnable problem document per planning family for the `spacepdhcg plan` CLI and the
`spacepdhcg.planner.plan()` API. Every document follows planner schema 1.0.0
(`src/spacepdhcg/planner/schema/problem.schema.json`) and uses the vehicle, gravity, and
constraint constants of the frozen transcriptions/dynamics models unless it says otherwise.

| file | family | horizon | what it plans |
|---|---|---|---|
| `hcw_rendezvous.json` | `hcw` | 40 × 20 s | 100 m radial / 50 m along-track / 20 m cross-track approach to the origin with a 0.05 m/s² acceleration-norm bound |
| `powered_descent_3dof.json` | `powered_descent_3dof` | 40 × 0.5 s | 150 m soft landing from (30, −20) m horizontal offset and −10 m/s, 15 kN max thrust, 30° tilt, 60° glide slope, Isp 221.6 s |
| `powered_descent_6dof.json` | `powered_descent_6dof` | 20 × 0.2 s | braking from −1 m/s at 100 m to a 97 m hover point with attitude, angular-rate, torque, and tilt constraints |
| `low_thrust.json` | `low_thrust` | 40 × 60 s | 500 kg / 1 N tangential radius raise from a 7000 km circular orbit to a reachable target state (μ = 398 600.4418 km³/s²) |

## Prerequisites

- Python ≥ 3.11 with the project installed (`pip install -e .` or the wheel) so `spacepdhcg`
  is on `PATH`, or run `python -m spacepdhcg.planner.cli` with `PYTHONPATH=src`.
- The native library `libspacepdhcg` (packaged in the wheel, or point
  `SPACEPDHCG_NATIVE_LIBRARY` at a build of `cpp/src/c_api.cpp`). It provides the planner
  transcription ABI used by validation helpers and the CPU reference.
- For GPU backends: the CUDA build (`-DSPACEPDHCG_BUILD_CUDA=ON`) produces
  `cuda-tools/spacepdhcg_plan`; set `SPACEPDHCG_PLAN_EXECUTABLE` to it (or put it on `PATH`).
  The default `pure_qoco` backend also needs the pinned QOCO-GPU library through
  `SPACEPDHCG_QOCO_LIBRARY` (see `scripts/gpu/checkout_build_qoco_gpu.sh`).

## Exact commands

Validate and print the canonical (unit-normalised) document:

```bash
spacepdhcg validate examples/planner/powered_descent_3dof.json
```

Plan on the GPU with the default backend (`pure_qoco`, frozen adaptive + pure-QOCO preset):

```bash
export SPACEPDHCG_PLAN_EXECUTABLE=/path/to/build/cuda-tools/spacepdhcg_plan
export SPACEPDHCG_QOCO_LIBRARY=/path/to/libqoco.so
spacepdhcg plan examples/planner/hcw_rendezvous.json      --output out/hcw
spacepdhcg plan examples/planner/powered_descent_3dof.json --output out/pd3
spacepdhcg plan examples/planner/powered_descent_6dof.json --output out/pd6
spacepdhcg plan examples/planner/low_thrust.json           --output out/low_thrust
```

Choose another validated backend:

```bash
spacepdhcg plan examples/planner/hcw_rendezvous.json --backend pdhcg          --output out/hcw-pdhcg
spacepdhcg plan examples/planner/low_thrust.json     --backend pdhcg_recovery --output out/lt-recovery
```

Run the clearly labelled CPU reference (Clarabel SCvx over the same native transcription;
no GPU needed):

```bash
spacepdhcg plan examples/planner/powered_descent_3dof.json --backend cpu_reference --output out/pd3-cpu
```

Warm start from a previous result and export the trajectory for the WebGL viewer:

```bash
spacepdhcg plan examples/planner/powered_descent_3dof.json \
    --warm-start out/pd3/plan-result.json \
    --output out/pd3-warm --export-viewer out/pd3-viewer
cd out/pd3-viewer && npm run check && npm run serve   # open http://127.0.0.1:4173/
```

Python API:

```python
from spacepdhcg.planner import PlanOptions, plan

result = plan("examples/planner/powered_descent_3dof.json", PlanOptions(output_directory="out/pd3"))
print(result.status, result.certified, result.objective, result.terminal_position_error)
again = plan("examples/planner/powered_descent_3dof.json", PlanOptions(warm_start=result))
```

## Outputs

`--output DIR` writes `problem.json` (canonical document), `plan-result.json` (strict result
document), `states.csv`, `controls.csv`, `replay.csv` (dense independent RK4/ZOH replay),
`iterations.csv` (per-iteration trust/forcing/accept-reject telemetry), and `plan-summary.md`.
For GPU runs `native-request.json` and `native-result.json` are kept as well.

Exit codes: `0` certified, `2` plan produced but not certified (max iterations, trust region
exhausted, time limit, failed gate), `3` inner solver failure, `64` invalid problem, `65` I/O
error, `66` GPU unavailable / CUDA error, `70` internal error.

## Family state and control orders (canonical units)

| family | state | control | units |
|---|---|---|---|
| `hcw` | `x y z vx vy vz` | `ax ay az` | m, m/s, m/s², rad/s |
| `powered_descent_3dof` | `x y z vx vy vz mass` | `thrust_x thrust_y thrust_z sigma` | m, m/s, kg, N, rad |
| `powered_descent_6dof` | `x y z vx vy vz q0 q1 q2 q3 wx wy wz mass` | `thrust_x thrust_y thrust_z torque_x torque_y torque_z sigma` | m, m/s, kg, N, N·m, rad, rad/s |
| `low_thrust` | `x y z vx vy vz mass` | `thrust_x thrust_y thrust_z sigma` | km, km/s, kg, N |

`sigma` is the thrust-magnitude slack of the lossless convexification (`|T| ≤ sigma ≤ T_max`).
A `units` block converts user units (deg, kN, km, min, …) into these canonical units.

## Known limitations (schema 1.0.0)

- Fixed final time only; `horizon.free_final_time` must be `false`.
- Terminal free/fixed flags must match the frozen transcriptions: position and velocity (plus
  attitude and angular rate for 6-DoF) are fixed, terminal mass is free. Other patterns are
  rejected with an explicit message.
- `minimum_altitude` must be 0 (the powered-descent transcriptions bound `z ≥ 0`); a positive
  low-thrust `thrust.minimum` is unsupported; HCW has no path constraints beyond the
  acceleration bound; inertia is always read in kg·m².
- The device path certifies node-level residuals; the dense continuous-time violation is
  reported (and flagged) but not part of the certificate.
- Pure QOCO inner solves run a cuDSS refactorisation per interior-point iteration, so the
  `pure_qoco` backend is the accuracy reference on the device (seconds per outer iteration at
  N = 20–40), while `pdhcg` is the fast persistent path. `pdhcg_recovery` uses tight
  1,000,000-iteration PDHCG inner solves with the projected-KKT/CGLS recovery enabled and can
  be very slow on infeasible or badly scaled requests; keep `time_limit_seconds` set (CLI
  default 600 s).
