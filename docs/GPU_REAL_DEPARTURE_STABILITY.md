# Real Earth-departure repeatability — 8 September 2026

The first full-catalogue-box CUDA campaign (v200) exposed a solver failure that
the earlier coast and frozen-QP test matrix did not cover. The same Earth to
asteroid 57530 departure, MJD 64328–64828 and 3,000 kg initial mass, passed once
and failed on a later candidate. An isolated H100 replay converged only once in
four default-coordinate attempts. The failed results did not pass the independent
physics gate and were not included in the certified fleet.

## Diagnosis and change

Snapshots from three fresh trajectory solves have a byte-identical first conic
subproblem (SHA-256 `6f173b7499c52445573e5a28d450e608a5e02c0a76534aa831a2fb4edd3dc52f`).
Its inputs include both topology and every numerical coefficient. Standalone QOCO
replays still vary, locating the first divergence after seed generation and
assembly. A Clarabel reference solves the same problem with independently audited
objective 0.1695985746634855 and relative gap 3.55e-12.

Disabling factorization graph capture did not remove the variation. Initcheck
reported no errors in two standalone replays; this is not a complete memory or
race audit. cuDSS documents nondeterministic atomic accumulation in its default
mode, consistent with the observed conditioning sensitivity, but these tests do
not identify a particular vendor kernel as the unique cause.
[NVIDIA reproducibility documentation](https://docs.nvidia.com/cuda/cudss/general.html#results-reproducibility).
Earlier deterministic-mode diagnostics remain in
[QOCO_DETERMINISM_DIAGNOSTICS.md](QOCO_DETERMINISM_DIAGNOSTICS.md).

The low-thrust factor regularization increases from **1e-9 to 1e-8**. This changes
the factorization used to calculate search directions. QOCO still refines against
the original linear system at 1e-12, and the conic residual and objective-gap gates
remain 1e-9. Dynamics, thrust, mass, mission rules and independent replay tolerances
are unchanged. Non-low-thrust settings are unchanged.

The standalone parameter sweep is deliberately retained, including its failures:

| Factor regularization | RTX 5090 original-equation audit | H100 original-equation audit |
| --- | ---: | ---: |
| 1e-9, baseline | 7/16 | 1/8 |
| **1e-8, selected** | **16/16** | **16/16** |
| 1e-7 | 1/16 | 0/16 |
| 1e-6 | 0/16 | 0/16 |
| 1e-5 | 0/16 | 0/16 |

Increasing regularization arbitrarily is not a solution. The selected value must
pass the trajectory and objective checks as well as the frozen-problem audit.

The route-search CLI also stops caching a failed SCvx leg as a proof of
infeasibility. That cache matched only the bodies and rounded departure time,
omitting arrival time and ship mass; it also suppressed retries of the identical
feasible boundary. The existing `--refine-top` attempt limit remains in force.

## Validation

- **32/32 real departure solves on each GPU** converge and pass independent
  physics: 16 in original coordinates and 16 with the state origin shifted.
  Final mass remains 2657.3638394 kg, within 2e-5 kg of the regression target.
- Default-coordinate median solve time is **0.519 s on RTX 5090 and 0.510 s on
  H100**. The old H100 median included failures and is not an equal-success
  throughput comparison.
- **328 local GTOC12 tests and 173 H100 GPU/CLI tests pass.** The new real-boundary
  test repeats four times in each of four coordinate/ordinary-or-outer-graph
  combinations, checking independent physics, fuel objective and zero seed
  trajectory uploads. The CLI regression confirms later candidates are attempted
  and the attempt cap still applies.
- **38 viewer tests pass.** Dataset checking validates both before/after missions
  and the incumbent fleet, including file hashes, event mass totals, archived
  samples and pinned Kepler elements.

Reproduce the focused regressions with the built libraries configured:

```bash
export PYTHONPATH=src
export SPACEPDHCG_GTOC12_GPU_TESTS=1
export SPACEPDHCG_GTOC12_CUDA_LIBRARY=/absolute/path/to/libspacepdhcg_cuda.so
export SPACEPDHCG_QOCO_LIBRARY=/absolute/path/to/libqoco.so
# Configure LD_LIBRARY_PATH before starting Python, including CUDA and cuDSS.
python -m pytest tests/test_gtoc12_gpu_earth_repeatability.py tests/test_gtoc12_run_refinement.py -q
```

[Evidence and source/runtime hashes](../results/lambda/2026-09-08/gpu-repeatability-v202/summary.json)
include original failed attempts, raw snapshots, independent audits, build logs
and test results. The compressed frozen-problem archive retains full primal/dual
vectors. Local loader and remote pytest import setup failures were corrected
before qualification; they are not successful tests.

## Complete campaign rerun

With the same search settings, v209 performs 18,484,006 Lambert branch evaluations
and 2,782,091 collection-option checks, returning the same 106 route candidates.
It refines ranks 0, 1 and 2, producing **two verified routes**, versus one in v200.
Rank 1 now completes all 13 legs and returns 475.154 kg; the best route remains
475.975 kg. Rank 2 passes departure but still fails the later 180-day transfer
from asteroid 45738 to 25792. That remaining failure is not proven physical
infeasibility and still needs diagnosis.

| Observed workload | Before, v200 | After, v209 |
| --- | ---: | ---: |
| Route search | 51.10 s | 50.16 s |
| Best route, 13-leg refinement | 10.52 s | 6.35 s |
| Complete campaign | 91.52 s | 77.28 s |
| Certified candidate routes | 1 | 2 |

These are single-run observations, not a paired median speedup claim; the
candidate attempts differ because the invalid failure cache was removed.
The selected mission passes the official checker and independent replay, with
maximum position discrepancy 0.354334 km and velocity discrepancy below
6.33e-8 km/s. It uses CUDA seed generation, discretisation, assembly, QOCO and
SCvx arithmetic/control. Python orchestration, route planning and verification
still contain CPU work. Fully GPU-native application execution remains open.

The incumbent 23-ship fleet stays at **12,805.194 weighted kg**. These missions
are candidate-pool additions, not an improved fleet score.
[Corrected mission and viewing instructions](../results/lambda/2026-09-08/gpu-native-campaign-v209/README.md).
