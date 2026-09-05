# Plan summary: HCW rendezvous: 100 m radial / 50 m along-track approach to the origin in 800 s

- Family: `hcw` (Hill-Clohessy-Wiltshire rendezvous; Hill/LVLH relative Cartesian (x radial, y along-track, z cross-track))
- Status: **certified** — converged and independently certified (CPU reference execution)
- Certified: **yes**
- Execution: `cpu_reference` via backend `cpu_reference` (policy `clarabel_scvx_reference`)

## Problem

- Intervals: 40, final time 800 s, step 20 s
- Initial state: [100, -50, 20, 0.1, -0.2, 0.05]
- Terminal target (free components shown as `free`): [0, 0, 0, 0, 0, 0]
- Terminal fixed flags: [true, true, true, true, true, true]
- Canonical units: {"position": "m", "velocity": "m/s", "acceleration": "m/s^2", "angular_rate": "rad/s", "time": "s"}

## Result

- Objective: 0.000369945898 (0.5 * sum_k |a_k|^2 (m^2/s^4))
- Outer iterations: 2 (accepted 1, rejected 1), inner iterations 10
- Final trust radius: 0.9
- Propellant used: 1.18171, final mass 0
- Terminal position error (independent replay): 1.714e-12 m; velocity error 2.659e-15 m/s

## Residuals

| quantity | solver | independent replay |
|---|---:|---:|
| canonical residual | 6.178e-11 | — |
| dynamics defect (scaled) | 0 | 0 |
| path violation (normalised) | 0 | 0 |
| terminal residual (scaled) | 1.714e-12 | 1.714e-12 |
| virtual control | 0 | — |
| replay parity | — | 0 |
| continuous-time violation (dense replay) | — | 0 |

## Certificate gates

| gate | passed | value | limit |
|---|---|---:|---:|
| solver_api_success | yes | 0 | 0 |
| converged | yes | 0 | 0 |
| canonical_residual | yes | 6.178e-11 | 1e-06 |
| device_dynamics_defect | yes | 0 | 1e-06 |
| device_path_violation | yes | 0 | 1e-06 |
| device_terminal_residual | yes | 1.714e-12 | 1e-06 |
| virtual_control | yes | 0 | 1e-06 |
| independent_replay_parity | yes | 0 | 1e-09 |
| independent_dynamics_defect | yes | 0 | 1e-06 |
| independent_path_violation | yes | 0 | 1e-06 |
| independent_terminal_residual | yes | 1.714e-12 | 1e-06 |
| no_hidden_cpu_fallback | yes | 0 | 0 |
| steady_state_residency | yes | 0 | 0 |
| coefficient_parity | yes | 0 | 5e-12 |
| independent_replay_evaluated | yes | 0 | 0 |

## Timings (seconds)

| stage | seconds |
|---|---:|
| cuda_startup_seconds | 0 |
| topology_seconds | 0.00010746 |
| coefficient_seconds | 0 |
| workspace_create_seconds | 0.073029 |
| solve_seconds | 0.0029316 |
| recovery_seconds | 0 |
| replay_seconds | 0 |
| acceptance_seconds | 0 |
| scvx_total_seconds | 0.282 |
| independent_replay_seconds | 0.00010212 |
| plan_wall_seconds | 0.28211 |
