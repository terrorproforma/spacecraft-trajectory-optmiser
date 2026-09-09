# CUDA mass-scaled trajectory initialization

The new initializer reproduced an archived return successfully, but it did not recover the candidate's missing fuel. The verified fleet remains **13,023.704900978 weighted kg**, with **14,291.006160165 raw kg**, 23 ships and 199 asteroids. No new full-fleet result was emitted by this experiment.

The additive API accepts an archived physical initial state, exact ZOH thrust vectors and a target initial mass. CUDA multiplies the archived mass and thrust by the same positive factor, then runs the existing GPU DOP853 initializer directly into the native SCvx workspace. Scaling both quantities preserves the reference position/velocity dynamics analytically; it does not guarantee the required final mass. The original unscaled API remains available.

Python supplies `ZohTrajectorySeed(..., target_initial_mass_kg=...)` while retaining the archived initial-state and thrust bytes unchanged. The native entry points are `spacepdhcg_gtoc12_scvx_solve_scaled_zoh_seed_host` and `spacepdhcg_gtoc12_scaled_zoh_seed_evaluate_host`. The mission worker used the solve entry point, with no separate initializer inspection or CPU trajectory upload. See [the seed contract](../src/spacepdhcg/gtoc12/trajectory_seed.py) and [the native interface](../cpp/cuda/include/spacepdhcg/cuda/gtoc12_scvx_c_api.h).

The controlled experiment used ship 10's return from asteroid 17126 to Earth, MJD 69263–69713. The archive and candidate have the same 226-node grid: 106 exact two-day burn intervals and 119 coasts. No time remeshing or thrust interpolation was needed. Both calls used QOCO for the native conic subproblems and the same 40 main plus 4 polish iteration limit.

| Measured local result | Original-mass control | Candidate-mass return |
|---|---:|---:|
| Initial mass | 1472.1728669809 kg | 1438.8883038351994 kg |
| SCvx iterations / accepted steps | 2 / 2 | 22 / 0 |
| Native call time | 0.719377 s | 9.093305 s |
| Maximum normalized dynamics defect | 1.9984e-13 | 2.1322e-13 |
| Independent return certificate | Passed | Not attempted after optimizer rejection |
| Independently propagated final mass | 1193.9107054861288 kg | Unavailable |

The candidate retained its initial scaled trajectory. Its **unverified final node mass** was 1166.9174106318044 kg, below the fixed cargo-plus-dry-mass requirement of 1177.2073921971253 kg by 10.2899815653 kg. Four of its 22 conic reports qualified, but all four corresponding trial steps were rejected. The native status was `infeasible` with `virtual control remains inf`; this is an optimizer status and retained accepted-step diagnostic, **not a mathematical infeasibility certificate**. The raw small dynamics defect does not override the mass requirement.

The original control's independent GPU certificate measured a 0.637092 km position error and a 1.04119e-7 km/s velocity error, within the unchanged acceptance gates. Its certificate applies to the normal post-clamp controls; the package also retains the distinct raw native output. Each solve uploaded 5,480 bytes of archived trajectory payload. The time vector and other input transfers are separate from that number.

The complete worker took 10.787373 seconds on an RTX 5090. It made two native solves, one fresh flight-certification call and no whole-route reruns, new wait propagations, Lambert searches or full-fleet checker calls. Seventeen prior flight certificates and one wait certificate were available for a possible candidate emission, but the failed return prevented it. These are initialization and failed-refinement measurements, not solutions-per-second or a global speedup.

The frozen build passed the original ZOH native test, the new scaled initializer test, the SCvx controller test and Compute Sanitizer memcheck. Its CPU validation recorded 92 passes and 27 intentionally skipped GPU-dependent cases; the separate mission harness passed 25 CPU construction/emission tests. The [portable evidence package](../results/local/2026-09-09/mass-scaled-return-v631/README.md) contains exact source/runtime identities, all failed outputs, the original prefix evidence, test logs and a static hash verifier. It requires no new solver or trajectory propagation to inspect.

The subsequent source audit found minimum-mass constraints at every node in the conic model, while the outer reference merit omits those state-mass violations. For this physically monotone reference mass history, the terminal shortfall is the largest violation. A dynamically consistent starting point can therefore have a misleadingly low reference merit despite lacking the required cargo-plus-dry mass. The next step is a controlled correction of that accounting; this experiment does not establish that the correction will produce a feasible route.
