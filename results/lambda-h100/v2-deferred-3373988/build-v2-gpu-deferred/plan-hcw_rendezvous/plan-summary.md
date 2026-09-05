# Plan summary: HCW rendezvous: 100 m radial / 50 m along-track approach to the origin in 800 s

- Family: `hcw` (Hill-Clohessy-Wiltshire rendezvous; Hill/LVLH relative Cartesian (x radial, y along-track, z cross-track))
- Status: **converged_not_certified** — the solver reported convergence but independent replay failed the certificate gates
- Certified: **no** (failed gates: independent_replay_parity, independent_dynamics_defect)
- Execution: `native_cuda` via backend `pure_qoco` (policy `pure_qoco`)

## Problem

- Intervals: 40, final time 800 s, step 20 s
- Initial state: [100, -50, 20, 0.1, -0.2, 0.05]
- Terminal target (free components shown as `free`): [0, 0, 0, 0, 0, 0]
- Terminal fixed flags: [true, true, true, true, true, true]
- Canonical units: {"position": "m", "velocity": "m/s", "acceleration": "m/s^2", "angular_rate": "rad/s", "time": "s"}

## Result

- Objective: 0.000369945935 (0.5 * sum_k |a_k|^2 (m^2/s^4))
- Outer iterations: 10 (accepted 1, rejected 9), inner iterations 40
- Final trust radius: 0.00351563
- Propellant used: 1.18171, final mass 0
- Terminal position error (independent replay): 5.823e-13 m; velocity error 9.853e-16 m/s

## Residuals

| quantity | solver | independent replay |
|---|---:|---:|
| canonical residual | 3.971e-11 | — |
| dynamics defect (scaled) | 0 | 1.386e-06 |
| path violation (normalised) | 0 | 0 |
| terminal residual (scaled) | 9.768e-07 | 5.823e-13 |
| virtual control | 0 | — |
| replay parity | — | 1.386e-06 |
| continuous-time violation (dense replay) | — | 0 |

## Certificate gates

| gate | passed | value | limit |
|---|---|---:|---:|
| solver_api_success | yes | 0 | 0 |
| converged | yes | 0 | 0 |
| canonical_residual | yes | 3.971e-11 | 1e-06 |
| device_dynamics_defect | yes | 0 | 1e-06 |
| device_path_violation | yes | 0 | 1e-06 |
| device_terminal_residual | yes | 9.768e-07 | 1e-06 |
| virtual_control | yes | 0 | 1e-06 |
| independent_replay_parity | no | 1.386e-06 | 1e-09 |
| independent_dynamics_defect | no | 1.386e-06 | 1e-06 |
| independent_path_violation | yes | 0 | 1e-06 |
| independent_terminal_residual | yes | 5.823e-13 | 1e-06 |
| no_hidden_cpu_fallback | yes | 0 | 0 |
| steady_state_residency | yes | 0 | 0 |
| coefficient_parity | yes | 1.777e-16 | 5e-12 |
| independent_replay_evaluated | yes | 0 | 0 |

## Timings (seconds)

| stage | seconds |
|---|---:|
| cuda_startup_seconds | 0.20301 |
| topology_seconds | 0.0016088 |
| coefficient_seconds | 0.00026826 |
| workspace_create_seconds | 0.010621 |
| solve_seconds | 0.38342 |
| recovery_seconds | 0 |
| replay_seconds | 0.0011328 |
| acceptance_seconds | 0.00037576 |
| scvx_total_seconds | 0.49831 |
| independent_replay_seconds | 2.4929e-05 |
| plan_wall_seconds | 0.51194 |
