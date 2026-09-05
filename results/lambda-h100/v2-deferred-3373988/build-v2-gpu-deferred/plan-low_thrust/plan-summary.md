# Plan summary: Low-thrust transfer: 40-minute tangential radius raise from a 7000 km circular orbit (500 kg, 1 N)

- Family: `low_thrust` (low-thrust two-body transfer; central-body inertial Cartesian)
- Status: **certified** — converged and independently certified
- Certified: **yes**
- Execution: `native_cuda` via backend `pure_qoco` (policy `pure_qoco`)

## Problem

- Intervals: 40, final time 2400 s, step 60 s
- Initial state: [7000, 0, 0, 0, 7.546053290107541, 0, 500]
- Terminal target (free components shown as `free`): [-5952.149848617802, 3686.049123066764, 0, -3.973625734575774, -6.4145609503740335, 0, "free"]
- Terminal fixed flags: [true, true, true, true, true, true, false]
- Canonical units: {"position": "km", "velocity": "km/s", "mass": "kg", "thrust": "N", "gravitational_parameter": "km^3/s^2", "time": "s"}

## Result

- Objective: 0.149921716 (mean_k(sigma_k) / maximum_thrust (normalised fuel))
- Outer iterations: 2 (accepted 2, rejected 0), inner iterations 26
- Final trust radius: 1
- Propellant used: 0.0122336, final mass 499.988
- Terminal position error (independent replay): 3.183e-11 km; velocity error 4.663e-14 km/s

## Residuals

| quantity | solver | independent replay |
|---|---:|---:|
| canonical residual | 1.746e-10 | — |
| dynamics defect (scaled) | 0 | 2.22e-16 |
| path violation (normalised) | 0 | 0 |
| terminal residual (scaled) | 4.663e-14 | 4.663e-14 |
| virtual control | 0 | — |
| replay parity | — | 2.22e-16 |
| continuous-time violation (dense replay) | — | 0 |

## Certificate gates

| gate | passed | value | limit |
|---|---|---:|---:|
| solver_api_success | yes | 0 | 0 |
| converged | yes | 0 | 0 |
| canonical_residual | yes | 1.746e-10 | 1e-06 |
| device_dynamics_defect | yes | 0 | 1e-06 |
| device_path_violation | yes | 0 | 1e-06 |
| device_terminal_residual | yes | 4.663e-14 | 1e-06 |
| virtual_control | yes | 0 | 1e-06 |
| independent_replay_parity | yes | 2.22e-16 | 1e-09 |
| independent_dynamics_defect | yes | 2.22e-16 | 1e-06 |
| independent_path_violation | yes | 0 | 1e-06 |
| independent_terminal_residual | yes | 4.663e-14 | 1e-06 |
| no_hidden_cpu_fallback | yes | 0 | 0 |
| steady_state_residency | yes | 0 | 0 |
| coefficient_parity | yes | 1.688e-14 | 5e-12 |
| independent_replay_evaluated | yes | 0 | 0 |

## Timings (seconds)

| stage | seconds |
|---|---:|
| cuda_startup_seconds | 0.2017 |
| topology_seconds | 0.0038166 |
| coefficient_seconds | 0.0016894 |
| workspace_create_seconds | 0.010687 |
| solve_seconds | 0.38298 |
| recovery_seconds | 0 |
| replay_seconds | 0.0010258 |
| acceptance_seconds | 2.3769e-05 |
| scvx_total_seconds | 0.49625 |
| independent_replay_seconds | 6.0553e-05 |
| plan_wall_seconds | 0.51367 |
