# Plan summary: 6-DoF powered descent: braking from -1 m/s at 100 m to a 97 m hover point in 4 s

- Family: `powered_descent_6dof` (6-DoF powered descent; local-level inertial Cartesian (z up); body thrust/torque; scalar-first quaternion)
- Status: **certified** — converged and independently certified
- Certified: **yes**
- Execution: `native_cuda` via backend `pure_qoco` (policy `pure_qoco`)

## Problem

- Intervals: 20, final time 4 s, step 0.2 s
- Initial state: [0, 0, 100, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 2000]
- Terminal target (free components shown as `free`): [0, 0, 97, 0, 0, -0.4, 1, 0, 0, 0, 0, 0, 0, "free"]
- Terminal fixed flags: [true, true, true, true, true, true, true, true, true, true, true, true, true, false]
- Canonical units: {"position": "m", "velocity": "m/s", "mass": "kg", "thrust": "N", "torque": "N*m", "angle": "rad", "angular_rate": "rad/s", "inertia": "kg*m^2", "time": "s"}

## Result

- Objective: 0.512975691 (mean_k(sigma_k) / maximum_thrust (normalised fuel))
- Outer iterations: 3 (accepted 2, rejected 1), inner iterations 68
- Final trust radius: 0.5
- Propellant used: 14.1581, final mass 1985.84
- Terminal position error (independent replay): 1.563e-13 m; velocity error 7.639e-13 m/s

## Residuals

| quantity | solver | independent replay |
|---|---:|---:|
| canonical residual | 3.687e-09 | — |
| dynamics defect (scaled) | 0 | 2.465e-32 |
| path violation (normalised) | 0 | 0 |
| terminal residual (scaled) | 7.639e-15 | 7.639e-15 |
| virtual control | 0 | — |
| replay parity | — | 7.211e-31 |
| continuous-time violation (dense replay) | — | 0 |

## Certificate gates

| gate | passed | value | limit |
|---|---|---:|---:|
| solver_api_success | yes | 0 | 0 |
| converged | yes | 0 | 0 |
| canonical_residual | yes | 3.687e-09 | 1e-06 |
| device_dynamics_defect | yes | 0 | 1e-06 |
| device_path_violation | yes | 0 | 1e-06 |
| device_terminal_residual | yes | 7.639e-15 | 1e-06 |
| virtual_control | yes | 0 | 1e-06 |
| independent_replay_parity | yes | 7.211e-31 | 1e-09 |
| independent_dynamics_defect | yes | 2.465e-32 | 1e-06 |
| independent_path_violation | yes | 0 | 1e-06 |
| independent_terminal_residual | yes | 7.639e-15 | 1e-06 |
| no_hidden_cpu_fallback | yes | 0 | 0 |
| steady_state_residency | yes | 0 | 0 |
| coefficient_parity | yes | 5.291e-17 | 5e-12 |
| independent_replay_evaluated | yes | 0 | 0 |

## Timings (seconds)

| stage | seconds |
|---|---:|
| cuda_startup_seconds | 0.20375 |
| topology_seconds | 0.0058627 |
| coefficient_seconds | 0.0024665 |
| workspace_create_seconds | 0.010762 |
| solve_seconds | 0.6982 |
| recovery_seconds | 0 |
| replay_seconds | 0.0010953 |
| acceptance_seconds | 0.00010151 |
| scvx_total_seconds | 0.82353 |
| independent_replay_seconds | 5.8969e-05 |
| plan_wall_seconds | 0.84395 |
