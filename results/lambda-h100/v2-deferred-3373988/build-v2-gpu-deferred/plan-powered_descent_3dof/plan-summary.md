# Plan summary: 3-DoF powered descent: 150 m soft landing with a 30 m / -20 m horizontal offset in 20 s

- Family: `powered_descent_3dof` (3-DoF powered descent; local-level inertial Cartesian (z up))
- Status: **converged_not_certified** — the solver reported convergence but independent replay failed the certificate gates
- Certified: **no** (failed gates: canonical_residual)
- Execution: `native_cuda` via backend `pure_qoco` (policy `pure_qoco`)

## Problem

- Intervals: 40, final time 20 s, step 0.5 s
- Initial state: [30, -20, 150, 0, 0, -10, 2000]
- Terminal target (free components shown as `free`): [0, 0, 0, 0, 0, 0, "free"]
- Terminal fixed flags: [true, true, true, true, true, true, false]
- Canonical units: {"position": "m", "velocity": "m/s", "mass": "kg", "thrust": "N", "angle": "rad", "time": "s"}

## Result

- Objective: 0.552219649 (mean_k(sigma_k) / maximum_thrust (normalised fuel))
- Outer iterations: 15 (accepted 1, rejected 14), inner iterations 39
- Final trust radius: 0.0001
- Propellant used: 76.2329, final mass 1923.77
- Terminal position error (independent replay): 0.000135 m; velocity error 5.402e-06 m/s

## Residuals

| quantity | solver | independent replay |
|---|---:|---:|
| canonical residual | 0.002525 | — |
| dynamics defect (scaled) | 0 | 0 |
| path violation (normalised) | 0 | 0 |
| terminal residual (scaled) | 1.35e-07 | 1.35e-07 |
| virtual control | 0 | — |
| replay parity | — | 0 |
| continuous-time violation (dense replay) | — | 0 |

## Certificate gates

| gate | passed | value | limit |
|---|---|---:|---:|
| solver_api_success | yes | 0 | 0 |
| converged | yes | 0 | 0 |
| canonical_residual | no | 0.002525 | 1e-06 |
| device_dynamics_defect | yes | 0 | 1e-06 |
| device_path_violation | yes | 0 | 1e-06 |
| device_terminal_residual | yes | 1.35e-07 | 1e-06 |
| virtual_control | yes | 0 | 1e-06 |
| independent_replay_parity | yes | 0 | 1e-09 |
| independent_dynamics_defect | yes | 0 | 1e-06 |
| independent_path_violation | yes | 0 | 1e-06 |
| independent_terminal_residual | yes | 1.35e-07 | 1e-06 |
| no_hidden_cpu_fallback | yes | 0 | 0 |
| steady_state_residency | yes | 0 | 0 |
| coefficient_parity | yes | 2.22e-16 | 5e-12 |
| independent_replay_evaluated | yes | 0 | 0 |

## Timings (seconds)

| stage | seconds |
|---|---:|
| cuda_startup_seconds | 0.20248 |
| topology_seconds | 0.0031952 |
| coefficient_seconds | 0.0016302 |
| workspace_create_seconds | 0.010565 |
| solve_seconds | 0.49757 |
| recovery_seconds | 0 |
| replay_seconds | 0.003922 |
| acceptance_seconds | 0.00055486 |
| scvx_total_seconds | 0.68354 |
| independent_replay_seconds | 5.0616e-05 |
| plan_wall_seconds | 0.70025 |
