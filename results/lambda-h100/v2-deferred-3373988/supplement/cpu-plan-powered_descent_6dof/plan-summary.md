# Plan summary: 6-DoF powered descent: braking from -1 m/s at 100 m to a 97 m hover point in 4 s

- Family: `powered_descent_6dof` (6-DoF powered descent; local-level inertial Cartesian (z up); body thrust/torque; scalar-first quaternion)
- Status: **certified** — converged and independently certified (CPU reference execution)
- Certified: **yes**
- Execution: `cpu_reference` via backend `cpu_reference` (policy `clarabel_scvx_reference`)

## Problem

- Intervals: 20, final time 4 s, step 0.2 s
- Initial state: [0, 0, 100, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 2000]
- Terminal target (free components shown as `free`): [0, 0, 97, 0, 0, -0.4, 1, 0, 0, 0, 0, 0, 0, "free"]
- Terminal fixed flags: [true, true, true, true, true, true, true, true, true, true, true, true, true, false]
- Canonical units: {"position": "m", "velocity": "m/s", "mass": "kg", "thrust": "N", "torque": "N*m", "angle": "rad", "angular_rate": "rad/s", "inertia": "kg*m^2", "time": "s"}

## Result

- Objective: 0.51297567 (mean_k(sigma_k) / maximum_thrust (normalised fuel))
- Outer iterations: 2 (accepted 2, rejected 0), inner iterations 51
- Final trust radius: 1
- Propellant used: 14.1581, final mass 1985.84
- Terminal position error (independent replay): 2.372e-08 m; velocity error 5.124e-12 m/s

## Residuals

| quantity | solver | independent replay |
|---|---:|---:|
| canonical residual | 1.929e-07 | — |
| dynamics defect (scaled) | 0 | 0 |
| path violation (normalised) | 2.942e-07 | 2.942e-07 |
| terminal residual (scaled) | 2.372e-11 | 2.372e-11 |
| virtual control | 1.718e-16 | — |
| replay parity | — | 0 |
| continuous-time violation (dense replay) | — | 2.942e-07 |

## Certificate gates

| gate | passed | value | limit |
|---|---|---:|---:|
| solver_api_success | yes | 0 | 0 |
| converged | yes | 0 | 0 |
| canonical_residual | yes | 1.929e-07 | 1e-06 |
| device_dynamics_defect | yes | 0 | 1e-06 |
| device_path_violation | yes | 2.942e-07 | 1e-06 |
| device_terminal_residual | yes | 2.372e-11 | 1e-06 |
| virtual_control | yes | 1.718e-16 | 1e-06 |
| independent_replay_parity | yes | 0 | 1e-09 |
| independent_dynamics_defect | yes | 0 | 1e-06 |
| independent_path_violation | yes | 2.942e-07 | 1e-06 |
| independent_terminal_residual | yes | 2.372e-11 | 1e-06 |
| no_hidden_cpu_fallback | yes | 0 | 0 |
| steady_state_residency | yes | 0 | 0 |
| coefficient_parity | yes | 0 | 5e-12 |
| independent_replay_evaluated | yes | 0 | 0 |

## Timings (seconds)

| stage | seconds |
|---|---:|
| cuda_startup_seconds | 0 |
| topology_seconds | 0.00014573 |
| coefficient_seconds | 0 |
| workspace_create_seconds | 0.32782 |
| solve_seconds | 0.046785 |
| recovery_seconds | 0 |
| replay_seconds | 0 |
| acceptance_seconds | 0 |
| scvx_total_seconds | 1.3237 |
| independent_replay_seconds | 0.0001473 |
| plan_wall_seconds | 1.3238 |
