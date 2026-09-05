# Plan summary: 3-DoF powered descent: 150 m soft landing with a 30 m / -20 m horizontal offset in 20 s

- Family: `powered_descent_3dof` (3-DoF powered descent; local-level inertial Cartesian (z up))
- Status: **certified** — converged and independently certified (CPU reference execution)
- Certified: **yes**
- Execution: `cpu_reference` via backend `cpu_reference` (policy `clarabel_scvx_reference`)

## Problem

- Intervals: 40, final time 20 s, step 0.5 s
- Initial state: [30, -20, 150, 0, 0, -10, 2000]
- Terminal target (free components shown as `free`): [0, 0, 0, 0, 0, 0, "free"]
- Terminal fixed flags: [true, true, true, true, true, true, false]
- Canonical units: {"position": "m", "velocity": "m/s", "mass": "kg", "thrust": "N", "angle": "rad", "time": "s"}

## Result

- Objective: 0.552213602 (mean_k(sigma_k) / maximum_thrust (normalised fuel))
- Outer iterations: 2 (accepted 2, rejected 0), inner iterations 82
- Final trust radius: 1
- Propellant used: 76.2321, final mass 1923.77
- Terminal position error (independent replay): 6.987e-06 m; velocity error 1.378e-07 m/s

## Residuals

| quantity | solver | independent replay |
|---|---:|---:|
| canonical residual | 1.193e-08 | — |
| dynamics defect (scaled) | 0 | 0 |
| path violation (normalised) | 1.417e-08 | 1.417e-08 |
| terminal residual (scaled) | 6.987e-09 | 6.987e-09 |
| virtual control | 2.683e-14 | — |
| replay parity | — | 0 |
| continuous-time violation (dense replay) | — | 1.417e-08 |

## Certificate gates

| gate | passed | value | limit |
|---|---|---:|---:|
| solver_api_success | yes | 0 | 0 |
| converged | yes | 0 | 0 |
| canonical_residual | yes | 1.193e-08 | 1e-06 |
| device_dynamics_defect | yes | 0 | 1e-06 |
| device_path_violation | yes | 1.417e-08 | 1e-06 |
| device_terminal_residual | yes | 6.987e-09 | 1e-06 |
| virtual_control | yes | 2.683e-14 | 1e-06 |
| independent_replay_parity | yes | 0 | 1e-09 |
| independent_dynamics_defect | yes | 0 | 1e-06 |
| independent_path_violation | yes | 1.417e-08 | 1e-06 |
| independent_terminal_residual | yes | 6.987e-09 | 1e-06 |
| no_hidden_cpu_fallback | yes | 0 | 0 |
| steady_state_residency | yes | 0 | 0 |
| coefficient_parity | yes | 0 | 5e-12 |
| independent_replay_evaluated | yes | 0 | 0 |

## Timings (seconds)

| stage | seconds |
|---|---:|
| cuda_startup_seconds | 0 |
| topology_seconds | 0.00013532 |
| coefficient_seconds | 0 |
| workspace_create_seconds | 0.33815 |
| solve_seconds | 0.060895 |
| recovery_seconds | 0 |
| replay_seconds | 0 |
| acceptance_seconds | 0 |
| scvx_total_seconds | 1.3736 |
| independent_replay_seconds | 0.00013651 |
| plan_wall_seconds | 1.3737 |
