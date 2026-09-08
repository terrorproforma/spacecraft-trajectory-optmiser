# Reproduced fixed-input return-leg convergence failure

The local GPU mesh campaign's second alternative return leg failed after 40
iterations. Its retained best fleet still passes both full-fleet physics checks.
This diagnostic reconstructs the pinned boundary at MJD 69230–69805 from asteroid
13077 to Earth with exactly the recorded initial mass, 1339.295993103377 kg.

Three calls at unchanged settings produce converged / converged / infeasible.
Disabling retained QOCO workspaces produces the same outcome sequence, with fresh
workspace creation verified in all three conic reports. Five Ruiz equilibration
iterations with objective preservation produce converged / infeasible / converged.
Thus neither disabling the pool nor this scaling setting fixes the failure.
None of these diagnostic settings is promoted as a production workaround.

All nine calls have byte-identical captured numerical arrays. The six unscaled
calls share one complete numerical input-envelope hash; the scaled envelope hash
differs because the equilibration setting changes. The captured envelope excludes
wall-clock deadlines. All six converged legs pass the existing independent CPU
rollout and unchanged acceptance gates; the three failed legs are rejected.
No fleet or score is promoted by this diagnostic.

Differences already appear in first-conic numerical outputs. This isolates
variability below the Python mission-input boundary but does not identify its
root cause: GPU-generated seeds, assembled matrices and vendor factorization
were not compared. Capturing those is the next isolation step. A solver status
of infeasible here is a failed local refinement, not proof that the transfer
has no physically feasible solution.

`audit.json` verifies saved array hashes, envelope hashes and disabled-pool
workspace creation. `raw.tar.gz` retains all nine numerical input packets,
solutions, conic and SCvx histories, logs and exact drivers. `raw-manifest.json`
hashes every member. The source and binaries are the frozen mesh build archived
in `results/lambda/2026-09-09/gpu-joint-mesh-v670/local-raw.tar.gz` (SHA-256
421cd1a63bbd6921ad12194be75b404250bb5be15a58d72da48d2a6efbd89ca6).
Core: 6fd02e87f3e148c0e62aac2bee731bc58397605a1d29ef3aebec95b0b4303c6c.
QOCO: 0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315.
