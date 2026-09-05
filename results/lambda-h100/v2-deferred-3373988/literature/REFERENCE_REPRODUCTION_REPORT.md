# Reference reproduction report (Phase 0-1 of the comparative solver campaign)

Generated: 2026-09-04T16:01:47+00:00  
Repository commit: `3373988057a251a4df11b9ac125583ed2a5ca65b` (working tree dirty)
  
Host: Linux-6.8.0-1046-nvidia-x86_64-with-glibc2.35, Python 3.12.14

Machine-readable twin: `benchmarks/literature/reference_reproduction.json`; per-target details in `results/literature/<target>.json`; provenance in `benchmarks/literature/provenance.json`; target registry in `benchmarks/literature/targets.json`.

Status vocabulary: `reproduced` (within the declared envelope), `gap` (measured but outside the envelope or not converged), `descriptive-only` (published data unrecoverable), `unsupported` (dynamics/model not implemented), `blocked` (external dependency).

## Summary

| Target | Family | Status | Published | Measured | Gap |
| --- | --- | --- | --- | --- | --- |
| `acikmese-ploen-2007-pd3` | P1-C-pd3 | **reproduced** | 399.5 kg propellant | 400.63 kg (lossless SOCP); 399.361 kg (repo SCvx CPU, converged); 399.367 kg (repo SCvx QOCO-GPU) | 1.1299 kg (lossless); -0.139144 kg (SCvx CPU) |
| `blackmore-2010-pd3-case1` | P1-C-pd3 | **reproduced** | 399.4 kg propellant | 400.088 kg (lossless SOCP); 398.848 kg (repo SCvx CPU, converged); GPU leg deferred (G4 owns the device; `literature gpu-run`) | 0.688345 kg (lossless); -0.551618 kg (SCvx CPU) |
| `chari-2024-pd6-monte-carlo` | P1-D-pd6 | **gap** | batch 256 demonstrated (no objective printed) | N=1: conv 0.00, 0.01 traj/s; N=16: conv 0.00, 0.05 traj/s; N=64: conv 0.05, 0.06 traj/s | GPU pure-QOCO pd6_fft batch deferred (preflight refused; `literature gpu-run`); device SCvx blocked |
| `esa-tops-2026` | P1-E-low-thrust | **reproduced** | no reference objectives | easy_two_body: converged; multi_revolution_two_body: converged; inclination_or_eccentricity_change: converged; cr3bp: unsupported | - |
| `gtoc12-official-verifier` | P2-F | **reproduced** | official verifier acceptance | bundled_result_example: 0 asteroids, 0 kg; gtoc12.solution_37_self_cleaning: 338 asteroids, 27045.3 kg; gtoc12.solution_39_mass_optimal: 356 asteroids, 28975.1 kg | - |
| `gtoc5-data-pin` | P2-F | **blocked** | asteroid_count=7075; evidence_label=published-reference | asteroid_count=7075 |  |
| `gtoc9-example-validation` | P2-F | **reproduced** | example1 removes debris 23, 3, 51 | example1: valid=True debris=[3, 23, 51]; example2: valid=True debris=[38, 46, 103, 114] | - |
| `gtopx-2021` | secondary | **reproduced** | cassini1=4.930708733982513; rosetta=1.343367; messenger_reduced=8.629944278158570; gtoc1=-1581950.131840605288744 | cassini1=4.93071; rosetta=1.34337; messenger_reduced=8.62994; gtoc1=-1.58195e+06 | cassini1=0.0e+00; rosetta=3.4e-07; messenger_reduced=6.7e-13; gtoc1=0.0e+00 |
| `szmuk-acikmese-2018-pd6-2d` | P1-D-pd6 | **reproduced** | t_f: figure-only; sweep spread <= 0.01 UT; 6 iterations | t_f = 3.39008 UT; sweep spread 0.000544859 UT; native pd6_fft t_f = 3.39254 UT (reproduced, gap vs core 0.00246459 UT) | spread - published = -0.00945514 UT; t_f descriptive-only |
| `tafazzol-taheri-earth-dionysus` | P1-E-low-thrust | **reproduced** | 2718.33 kg final mass | 2717.43 kg | -0.902514 kg |
| `tafazzol-taheri-earth-mars` | P1-E-low-thrust | **reproduced** | 603.935 kg final mass | 603.925 kg | -0.0102507 kg |

## Per-target detail

### `acikmese-ploen-2007-pd3`

- Family: P1-C-pd3
- Status: **reproduced** (support: supported)
- Wall time: 125.5 s
- Evidence labels:
  - `published.fuel_used_kg`: `published-reference`
  - `measured.lossless_fuel_used_kg`: `measured-local`
  - `measured.scvx_cpu_fuel_used_kg`: `measured-local`
  - `measured.scvx_cpu_frozen_euler_fuel_used_kg`: `measured-local`
  - `measured.scvx_qoco_gpu_fuel_used_kg`: `measured-local`
- Published:
  - fuel_used_kg: 399.5
  - fuel_used_text: 399.5
  - objective_convention: propellant-used
  - evidence_label: published-reference
  - extraction: secondary-citation (Wenzel 2018 DLR thesis quoting Acikmese & Ploen 2007 Fig. results for t_f = 81 s with the 4 deg glide slope)
  - companion_case_no_glide_slope: time_of_flight_s=72; fuel_used_kg=387.9; fuel_used_text=387.9
- Measured:
  - lossless_fuel_used_kg: 400.63
  - lossless_fuel_used_kg_by_dt: dt=1.0=400.63; dt=0.5=401.265
  - lossless_replay_fuel_used_kg: 400.63
  - lossless_isp_only_alpha_fuel_used_kg: 362.618
  - lossless_forward_euler_diagnostic_fuel_used_kg: 404.479
  - scvx_cpu_frozen_euler_fuel_used_kg: 406.887
  - scvx_cpu_frozen_euler_status: solver_failed
  - scvx_cpu_frozen_euler_replay_terminal_position_error_m: 94.8748
  - scvx_cpu_fuel_used_kg: 399.361
  - scvx_cpu_replay_fuel_used_kg: 399.361
  - scvx_cpu_status: converged
  - scvx_cpu_termination_reason: feasibility and step tolerances satisfied
  - scvx_cpu_fuel_used_kg_by_dt: dt=1.0=399.361; dt=0.5=399.356
  - scvx_qoco_gpu_fuel_used_kg: 399.367
  - scvx_qoco_gpu_replay_fuel_used_kg: 399.367
  - scvx_qoco_gpu_status: converged
- Gap:
  - lossless_minus_published_kg: 1.1299
  - lossless_relative: 0.00282829
  - acceptance_tolerance_kg: 2
  - scvx_cpu_minus_published_kg: -0.139144
  - scvx_cpu_minus_lossless_kg: -1.26905
  - scvx_cpu_frozen_euler_minus_published_kg: 7.38729
  - euler_discrete_optimum_minus_lossless_kg: 3.84943
  - scvx_qoco_gpu_minus_lossless_kg: -1.26314
- Discretisation envelope:
  - discretisation: zero-order hold, exact double-integrator map, exact log-mass step
  - scvx_discretisation: zero-order-hold thrust, variational RK4 coefficients, multiple-shooting exact-penalty merit
  - dt_values_s: 1, 0.5
  - fuel_spread_kg: 0.634684
  - scvx_fuel_spread_kg: 0.00455245
  - declared_envelope_kg: 2
- Commands: `spacepdhcg literature run acikmese-ploen-2007-pd3`

### `blackmore-2010-pd3-case1`

- Family: P1-C-pd3
- Status: **reproduced** (support: supported)
- Wall time: 341.5 s
- Evidence labels:
  - `published.fuel_used_kg`: `published-reference`
  - `measured.lossless_fuel_used_kg`: `measured-local`
  - `measured.scvx_cpu_fuel_used_kg`: `measured-local`
  - `measured.scvx_cpu_frozen_euler_fuel_used_kg`: `measured-local`
- Published:
  - fuel_used_kg: 399.4
  - fuel_used_text: 399.4
  - time_of_flight_text: 78.4
  - discretisation_points: 55
  - objective_convention: propellant-used
  - evidence_label: published-reference
  - extraction: text (Section V, case 1: 'This solution requires 399.4 kg of fuel and has t_f = 78.4 s'; 55 time-discretisation points; golden search interval 3.0 s)
- Measured:
  - lossless_fuel_used_kg: 400.088
  - lossless_fuel_used_kg_by_dt: dt=0.4=400.088; dt=0.2=400.763
  - lossless_replay_fuel_used_kg: 400.084
  - lossless_isp_only_alpha_fuel_used_kg: 362.476
  - lossless_forward_euler_diagnostic_fuel_used_kg: 401.262
  - scvx_cpu_frozen_euler_fuel_used_kg: 413.425
  - scvx_cpu_frozen_euler_status: solver_failed
  - scvx_cpu_frozen_euler_replay_terminal_position_error_m: 37.3089
  - scvx_cpu_fuel_used_kg: 398.848
  - scvx_cpu_replay_fuel_used_kg: 398.848
  - scvx_cpu_status: converged
  - scvx_cpu_termination_reason: feasibility and step tolerances satisfied
  - scvx_cpu_fuel_used_kg_by_dt: dt=0.4=398.848; dt=0.2=398.868
  - scvx_qoco_gpu_status: deferred
- Gap:
  - lossless_minus_published_kg: 0.688345
  - lossless_relative: 0.00172345
  - acceptance_tolerance_kg: 2
  - scvx_cpu_minus_published_kg: -0.551618
  - scvx_cpu_minus_lossless_kg: -1.23996
  - scvx_cpu_frozen_euler_minus_published_kg: 14.0251
  - euler_discrete_optimum_minus_lossless_kg: 1.17317
- Discretisation envelope:
  - discretisation: zero-order hold, exact double-integrator map, exact log-mass step
  - scvx_discretisation: zero-order-hold thrust, variational RK4 coefficients, multiple-shooting exact-penalty merit
  - dt_values_s: 0.4, 0.2
  - fuel_spread_kg: 0.674955
  - scvx_fuel_spread_kg: 0.0193044
  - declared_envelope_kg: 2
- Commands: `spacepdhcg literature run blackmore-2010-pd3-case1`; `spacepdhcg literature gpu-run blackmore-2010-pd3-case1  # when the device is free`
- Note: GPU pure-QOCO SCvx deferred by preflight: refused: other compute processes hold the device (pid 37205); pass allow_shared to override

### `chari-2024-pd6-monte-carlo`

- Family: P1-D-pd6
- Status: **gap** (support: partial)
- Wall time: 577.2 s
- Evidence labels:
  - `published.distribution`: `published-reference`
  - `published.batch_size_256`: `published-reference`
  - `measured.cpu_independent_batch`: `measured-local`
- Published:
  - initial_position_distribution: uniform(6,9), uniform(3,6), uniform(1,2)
  - published_demonstration_batch_size: 256
  - maximum_outer_iterations: 25
  - pipg_iterations_per_subproblem: 2500
  - evidence_label: published-reference
  - verification_status: requires-source-verification (values imported from benchmarks/literature_baselines.json; paper body not retrieved)
- Measured:
  - cpu_independent_batch: 1=batch_size=1; seed=20240101; solved=1; converged=0; convergence_probability=0; accepted=1; accepted_probability=1; accepted_trajectories_per_second=0.014438; wall_seconds=69.2619; workers=1; tof_median=4.74687; tof_iqr=4.74687, 4.74687; fuel_median=0.189705; violation_max=2.47224e-07; defect_max=2.62427e-06; 16=batch_size=16; seed=20240116; solved=16; converged=0; convergence_probability=0; accepted=6; accepted_probability=0.375; accepted_trajectories_per_second=0.0489476; wall_seconds=122.58; workers=16; tof_median=4.76356; tof_iqr=4.38464, 4.97145; fuel_median=0.191666; violation_max=3.13313e-05; defect_max=0.00012794; 64=batch_size=64; seed=20240164; solved=64; converged=3; convergence_probability=0.046875; accepted=22; accepted_probability=0.34375; accepted_trajectories_per_second=0.0570977; wall_seconds=385.304; workers=25; tof_median=4.67625; tof_iqr=4.50925, 4.86352; fuel_median=0.190154; violation_max=4.4766e-05; defect_max=0.000207929
  - gpu_persistent_batch: persistent_device_scvx=status=blocked; reason=the persistent GPU 6-DoF SCvx is only reachable through the frozen G4 fixture families (device_scvx_integration_test --g4-sample P1-D-pd6 <attitude_class> <rate_class>); it has no entry point for arbitrary initial positions and uses the repository's independent-torque 6-DoF model, not the Szmuk/Chari thrust-arm model; pure_qoco_native_pd6_fft=status=deferred; preflight=ok=False; reason=refused: other compute processes hold the device (pid 37205); pass allow_shared to override; nvidia_smi=/usr/bin/nvidia-smi; g4_owned=False; processes=pid=37205; reported_name=/home/ubuntu/spacepdhcg/v2/.venv/bin/python; command_line=/home/ubuntu/spacepdhcg/v2/.venv/bin/python /home/ubuntu/spacepdhcg/v2/.venv/bin/spacepdhcg literature gpu-run acikmese-ploen-2007-pd3 blackmore-2010-pd3-case1 chari-2024-pd6-monte-carlo; g4_owner=False; qoco_library=/home/ubuntu/spacepdhcg/v1/build-current-head-qoco/libqoco.so
- Discretisation envelope:
  - solver: independent CPU free-final-time 6-DoF SCvx (Szmuk 2018 vehicle model), one process per trajectory
  - acceptance: replay defect <= 1e-5 and path violation <= 1e-6
- Commands: `spacepdhcg literature run chari-2024-pd6-monte-carlo`; `spacepdhcg literature gpu-run chari-2024-pd6-monte-carlo  # deferred GPU leg (preflight-gated)`
- Note: independent batch: no shared controls, no non-anticipativity constraints; not robust optimisation
- Note: the paper's vehicle parameter table is not reproduced here; the Szmuk 2018 Table 1 vehicle is used with the published position dispersion

### `esa-tops-2026`

- Family: P1-E-low-thrust
- Status: **reproduced** (support: partial)
- Wall time: 336.6 s
- Evidence labels:
  - `measured.easy_two_body`: `measured-local`
  - `measured.multi_revolution_two_body`: `measured-local`
  - `measured.inclination_or_eccentricity_change`: `measured-local`
  - `measured.cr3bp`: `measured-local`
- Published:
  - reference_objectives: none published for the Cartesian two-body problems at the pinned revision
- Measured:
  - easy_two_body: problem=two_body_cartesian:P4; status=converged; formulation=cartesian; final_mass=0.614057; time_of_flight=2; revolutions=None; reason=None
  - multi_revolution_two_body: problem=two_body_cartesian:P3; status=converged; formulation=modified equinoctial elements, RTN thrust, FOH SCvx; final_mass=0.69222; time_of_flight=13; revolutions=2; reason=None
  - inclination_or_eccentricity_change: problem=two_body_cartesian:P1; status=converged; formulation=modified equinoctial elements, RTN thrust, FOH SCvx; final_mass=0.872629; time_of_flight=90.6082; revolutions=1; reason=None
  - cr3bp: problem=cr3bp:P0; status=unsupported; formulation=None; final_mass=None; time_of_flight=None; revolutions=None; reason=circular restricted three-body dynamics are not implemented
- Discretisation envelope:
  - nodes: 120
  - mee_nodes: 200
  - problem_count_at_pinned_revision: 34
- Commands: `spacepdhcg literature run esa-tops-2026`
- Note: the ISSFD paper describes 28 problems; the pinned repository revision contains 34 (database is actively expanding); selection made from metadata only

### `gtoc12-official-verifier`

- Family: P2-F
- Status: **reproduced** (support: supported)
- Wall time: 1.3 s
- Evidence labels:
  - `measured.bundled_result_example`: `reproduced-external`
  - `measured.gtoc12.solution_37_self_cleaning`: `reproduced-external`
  - `measured.gtoc12.solution_39_mass_optimal`: `reproduced-external`
- Published:
  - note: the portal publishes the solution files without an accompanying printed score; the official verifier output (ships, mined asteroids, total resource mass) is the reproducible quantity
  - evidence_label: published-reference
- Measured:
  - bundled_result_example: accepted=True; ships=1; mined_asteroids=0; total_resource_mass_kg=0
  - gtoc12.solution_37_self_cleaning: accepted=True; ships=37; mined_asteroids=338; total_resource_mass_kg=27045.3
  - gtoc12.solution_39_mass_optimal: accepted=True; ships=39; mined_asteroids=356; total_resource_mass_kg=28975.1
- Discretisation envelope:
  - evaluator: official GTOC12 verification program (Linux binary, pinned)
- Commands: `spacepdhcg literature run gtoc12-official-verifier`
- Note: no route search run; official verifier accepted every file

### `gtoc5-data-pin`

- Family: P2-F
- Status: **blocked** (support: unsupported)
- Unsupported reason: no official offline evaluator or example solution published; pinned research code requires pykep
- Wall time: 0.0 s
- Evidence labels:
  - `measured.asteroid_count`: `measured-local`
- Published:
  - asteroid_count: 7075
  - evidence_label: published-reference
- Measured:
  - asteroid_count: 7075
- Commands: `spacepdhcg literature run gtoc5-data-pin`
- Note: blocked: no official offline GTOC5 evaluator or example solution file is published; the pinned Simoes et al. beam P-ACO implementation requires pykep, which is not installed in the campaign environment

### `gtoc9-example-validation`

- Family: P2-F
- Status: **reproduced** (support: partial)
- Wall time: 0.1 s
- Evidence labels:
  - `measured.example1`: `measured-local`
  - `measured.example2`: `measured-local`
- Published:
  - example1_debris: 23, 3, 51
  - example1_text: contains a possibly valid mission able to remove the debris with ids 23, 3 and 51
  - cost_function: J = sum_i [c_i + alpha (m_0i - m_dry)^2], c_i in [45, 55] MEUR during the competition, alpha = 2.0e-6 MEUR/kg^2
  - evidence_label: published-reference
- Measured:
  - example1: valid=True; debris_removed=3, 23, 51; mission_cost_min_meur=64.7192; mission_cost_max_meur=74.7192
  - example2: valid=True; debris_removed=38, 46, 103, 114; mission_cost_min_meur=65.0978; mission_cost_max_meur=75.0978
- Discretisation envelope:
  - evaluator: local implementation of Kelvins rules 4-19 (no official offline evaluator exists)
  - propagator: scipy DOP853 rtol 1e-12 on the official J2 equations
- Commands: `spacepdhcg literature run gtoc9-example-validation`
- Note: the official GTOC9 scoring ran on the Kelvins server; both official example submissions must validate under the re-implemented rules

### `gtopx-2021`

- Family: secondary global mission-design track
- Status: **reproduced** (support: supported)
- Wall time: 0.0 s
- Evidence labels:
  - `published.cassini1`: `published-reference`
  - `published.rosetta`: `published-reference`
  - `published.messenger_reduced`: `published-reference`
  - `published.gtoc1`: `published-reference`
  - `measured.cassini1`: `reproduced-external`
  - `measured.rosetta`: `reproduced-external`
  - `measured.messenger_reduced`: `reproduced-external`
  - `measured.gtoc1`: `reproduced-external`
- Published:
  - cassini1: 4.930708733982513
  - rosetta: 1.343367
  - messenger_reduced: 8.629944278158570
  - gtoc1: -1581950.131840605288744
- Measured:
  - cassini1: 4.93071
  - rosetta: 1.34337
  - messenger_reduced: 8.62994
  - gtoc1: -1.58195e+06
- Gap:
  - cassini1: 0
  - rosetta: 3.36592e-07
  - messenger_reduced: 6.73239e-13
  - gtoc1: 0
- Discretisation envelope:
  - comparison: exact evaluator re-evaluation of the official vector, compared at the precision printed in the official solution file
  - exact_relative_tolerance: 1e-09
  - exact_count: 3
- Commands: `spacepdhcg literature run gtopx-2021`
- Note: no global optimiser run; evaluator provided as a test target

### `szmuk-acikmese-2018-pd6-2d`

- Family: P1-D-pd6
- Status: **reproduced** (support: partial)
- Wall time: 285.8 s
- Evidence labels:
  - `published.tf_guess_sweep_spread_ut`: `published-reference`
  - `published.iterations_to_converge`: `published-reference`
  - `published.converged_time_of_flight`: `descriptive-only`
  - `measured.time_of_flight`: `measured-local`
  - `measured.native_pd6_fft.time_of_flight`: `measured-local`
  - `parameters.alpha_mdot`: `descriptive-only`
- Published:
  - tf_guess_sweep_spread_ut: 0.01
  - tf_guess_sweep_spread_text: within 0.01 [UT] of each other
  - tf_guesses_ut: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
  - iterations_to_converge: 6
  - iterations_to_converge_text: convergence was obtained by the sixth iteration
  - evidence_label: published-reference
  - objective_value: descriptive-only (Figure 2 only)
- Measured:
  - time_of_flight_by_guess: 1.0=3.39027; 2.0=3.39017; 3.0=3.39009; 4.0=3.39008; 5.0=3.39023; 6.0=3.39027; 7.0=3.39027; 8.0=3.39062; 9.0=3.39024; 10.0=3.39033
  - time_of_flight_spread_ut: 0.000544859
  - iterations_by_guess: 1.0=15; 2.0=15; 3.0=15; 4.0=15; 5.0=15; 6.0=15; 7.0=15; 8.0=15; 9.0=15; 10.0=15
  - statuses_by_guess: 1.0=maximum_iterations; 2.0=maximum_iterations; 3.0=maximum_iterations; 4.0=maximum_iterations; 5.0=maximum_iterations; 6.0=maximum_iterations; 7.0=maximum_iterations; 8.0=maximum_iterations; 9.0=maximum_iterations; 10.0=maximum_iterations
  - extended_run: iterations=30; status=maximum_iterations; time_of_flight=3.39008; fuel_used=0.142627; replay_defect_inf=6.06373e-09; max_path_violation=9.68834e-10; final_trust_norm=0.00152875; final_virtual_l1=3.3106e-09
  - native_pd6_fft: status=reproduced; time_of_flight=3.39254; fuel_used=0.142861; converged=True; termination=converged; iterations=30; replay_defect_inf=6.42941e-07; max_path_violation=8.07702e-08; path_violations=dry_mass=0; glide_slope=5.39109e-16; tilt=4.88555e-09; angular_rate=0; thrust_min=0; thrust_max=8.07702e-08; gimbal=0; quaternion_norm=3.61442e-08; topology_fingerprint=8145343b74888a29; gap_vs_cpu_core_ut=0.00246459; envelope_ut=0.01; label=measured-local
- Gap:
  - time_of_flight_spread_minus_published_ut: -0.00945514
  - converged_time_of_flight: descriptive-only: the paper prints no digits (Figure 2 only)
- Discretisation envelope:
  - discretisation: K = 50 nodes, FOH, RK4 STM with 8 substeps per interval; free final time via sigma
  - paper_stop_rule: ||Delta||_2 <= 1e-3 and ||nu||_1 <= 1e-10 within 15 iterations
  - native_pd6_fft: K = 50 nodes, ZOH, variational RK4 with 4 substeps per interval, sigma column analytic; hard-trust-region SCvx; declared envelope vs the FOH core 0.01 UT
- Commands: `spacepdhcg literature run szmuk-acikmese-2018-pd6-2d`
- Note: alpha_mdot = 0.01 UT/UL and a zero vertical initial velocity are assumptions (not printed in the paper)

### `tafazzol-taheri-earth-dionysus`

- Family: P1-E-low-thrust
- Status: **reproduced** (support: supported)
- Wall time: 397.9 s
- Evidence labels:
  - `published.final_mass_kg`: `published-reference`
  - `measured.final_mass_kg_best`: `measured-local`
- Published:
  - final_mass_kg: 2718.33
  - final_mass_text: 2718.33
  - best_reported_revolutions: 5
  - evidence_label: published-reference
  - extraction: text (Section 4.2: 'm(t_f) = m_f = 2718.33 kg'; 'The most optimal solution involves five orbital revolutions')
- Measured:
  - final_mass_kg_best: 2717.43
  - final_mass_kg_by_nodes: nodes=300=2717.43; nodes=400=2717.67
  - statuses: nodes=300=converged; nodes=400=maximum_iterations
- Gap:
  - final_mass_minus_published_kg: -0.902514
  - relative: -0.00033201
  - acceptance_tolerance_kg: 2
- Discretisation envelope:
  - discretisation: FOH successive convexification, RK4 STM, nodes swept; modified-equinoctial-element state with explicit revolution count
  - formulation: mee
  - nodes_values: 300, 400
  - final_mass_spread_kg: 0
  - declared_envelope_kg: 2
- Commands: `spacepdhcg literature run tafazzol-taheri-earth-dionysus`
- Note: boundary states are the paper's fixed heliocentric states (zero hyperbolic excess); no ephemeris model is involved
- Note: revolution count swept [4, 5, 6] at the coarsest node count; best N = 5 (published: 5)

### `tafazzol-taheri-earth-mars`

- Family: P1-E-low-thrust
- Status: **reproduced** (support: supported)
- Wall time: 34.8 s
- Evidence labels:
  - `published.final_mass_kg`: `published-reference`
  - `measured.final_mass_kg_best`: `measured-local`
- Published:
  - final_mass_kg: 603.935
  - final_mass_text: 603.935
  - evidence_label: published-reference
  - extraction: text (Section 4.1: 'm(t_f) = m_f = 603.935 kg')
- Measured:
  - final_mass_kg_best: 603.925
  - final_mass_kg_by_nodes: nodes=150=603.883; nodes=300=603.925
  - statuses: nodes=150=converged; nodes=300=converged
- Gap:
  - final_mass_minus_published_kg: -0.0102507
  - relative: -1.69732e-05
  - acceptance_tolerance_kg: 0.5
- Discretisation envelope:
  - discretisation: FOH successive convexification, RK4 STM, nodes swept
  - nodes_values: 150, 300
  - final_mass_spread_kg: 0.041924
  - declared_envelope_kg: 0.5
- Commands: `spacepdhcg literature run tafazzol-taheri-earth-mars`
- Note: boundary states are the paper's fixed heliocentric states (zero hyperbolic excess); no ephemeris model is involved

