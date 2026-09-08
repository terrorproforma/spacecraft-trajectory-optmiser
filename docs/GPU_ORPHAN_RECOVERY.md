# Verified improvement from recovering an abandoned miner

The local RTX 5090 experiment v595 improves the retained 23-ship fleet. Both
complete-fleet checkers pass at unchanged tolerances, and a fresh CPU audit
independently reproduces the score and verifies that the other 22 ships are unchanged.

| Metric | Retained v11 | Improved v595 |
| --- | ---: | ---: |
| Physical material returned | 14,047.802875 kg | **14,051.854894 kg** |
| Fixed-bonus weighted score | 12,805.194102 kg | **12,810.135953 kg** |
| Physical material per ship | 610.774038 kg | **610.950213 kg** |
| Ships | 23 | 23 |
| Asteroids whose material reached Earth | 194 | 195 |
| Asteroids with deployed miners | 196 | 196 |

Ship 15 now collects asteroid 19102 after a 456-day camp. That yields 12.484600 kg;
retiming other pickups costs 8.432580 kg, leaving **4.052019 net raw kg**. Its bonus
coefficient is 1; differently weighted changes in the other pickups make the
weighted gain **4.941851 kg**, or **0.038593%**. Ship 15's own haul rises from
605.639973 to 609.691992 kg. This is a small verified improvement, not evidence of
global optimality or a large advance toward the winning score.

The previous neighborhoods missed this structure: fixed-order retiming retained
the collection sequence, while insertion excluded asteroids already in the route.
Both tested ships deployed nine miners but collected eight. Inserting their own
abandoned miner as the first collection merges the last deployment and first
collection into a camp, adding no flight leg. Other insertion positions add one.

## What the GPU run evaluated

The two ships each had nine insertion positions, tested on 15-day and 5-day DP
grids. Of those 36 layouts, four closed the surrogate mass budget and two were
worth full route refinement. Both refined routes passed the full-fleet check;
the higher-scoring first candidate was retained.

| Recorded work | Count / time |
| --- | ---: |
| Collection orders | 18 |
| Retiming driver calls | 36 |
| Internal DP calls / forward calls | 277 / 277 |
| Joint schedule candidates / batches | 23,142 / 132 |
| Logical Lambert branch requests | 106,024,898 |
| Element-hop requests | 53,012,449 |
| Complete routes refined | 2 |
| Native leg solves | 36, all converged |
| Native iterations / accepted iterations | 159 / 153 |
| Sum of native solve-call time | 9.650684 s |
| Sum of complete route-refinement time | 11.487757 s |
| Campaign timer | 78.115478 s |

Logical requests include intermediate search work and repeated requests. They are
not unique solutions or independently certified low-thrust trajectories. The
campaign timer includes three complete-fleet CPU checks, GPU search/context setup,
refinement, viewer export and checkpoints. It starts after imports, data loading,
hashing and parsing the original solution; it excludes the later 20.405 s audit.
This single measurement does not establish comparative throughput or speedup.

The run uses CUDA Lambert screening, DP, joint candidate arithmetic, trajectory
seed, dynamics, conic assembly, QOCO and the SCvx outer loop. Python orchestrates
the campaign and CPU checkers provide independent acceptance. This recorded run
also chooses the winning joint row on CPU, using
`SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=0` with the pinned v590 core.
The preceding attempt failed before candidate work because that older binary
lacked the newer winner-selection entry point. Its evidence is retained.

SCvx settings remain 40 iterations, 2-day nodes, zero Ruiz, graph execution, and
the original physics and verification tolerances. No input, thrust, mass or
trajectory acceptance threshold was relaxed. Source and library hashes are in
the report; the successful frozen Python snapshot passed all 42 parity checks.

## Why a lower-haul pilot does not replace the best fleet

The richer three-ship pilot reproduced the historical first route but was a
separate candidate fleet. It did not replace the retained 23-ship solution.
Weighted returned mass is the objective; raw mean returned mass constrains the
allowed ship count:

\[
N \leq 2\exp(0.004\,M_{\rm raw}/N).
\]

With the improved 23 routes retained unchanged, a 24th ship needs **857.585005 raw
kg** just to satisfy this rule. Improving existing routes avoids that large
incremental hurdle. The current fixed-bonus score would sit ninth if inserted
into the published competition leaderboard, 295.626 weighted kg below ATQ. It
is a retrospective comparison, not an official submitted rank.

## Evidence and reproduction

[Published evidence](../results/local/2026-09-09/orphan-recovery-v595/) contains:

- `Result.txt`: the verified improved complete fleet.
- `report.json`, `compatible-launch.json`: results, exact runtime and flags.
- `raw.tar.gz`: both candidate routes, all proxies, native solve records, viewer
  export and the preceding failed-start evidence.
- `source.tar.gz`, `source-sha256.json`, `run.py`: frozen Python, inputs and driver.
- `snapshot-parity.*`: the pinned compatible implementation's 42-test pass.
- `audit/`: fresh independent and official verification, per-asteroid scoring,
  unmodified-ship comparison and reproducible integrity checks.
- `sha256.json`: byte counts and hashes of the published evidence.

Extract `source.tar.gz` to reproduce using the retained CUDA core and the recorded
library/data paths in `compatible-launch.json`. Use a fresh output directory and
retain both joint feature flags. The recorded CLI adds `--joint-seconds 15` to
the 18-order/two-grid search; source and binary hashes identify the actual tested
configuration. The original historical incumbent is preserved.

## Load the improved fleet

The existing visualiser has a new dataset named **Improved fleet v595**.

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath 'node' -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-orphan-v595&epoch=69807&preset=oblique&z=1'
```

The solution's full local path is:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\local\2026-09-09\orphan-recovery-v595\Result.txt
```

The importer verifies the solution/export hashes and pinned catalogue, retains
11,679 exact replay samples and checks 73,562 orbit context points. Its maximum
Kepler context disagreement is 3.59e-6 km. The viewer shows 196 visited/deployed
asteroids and 195 actually collected asteroids; those counts describe different
events. The compute panel shows the weighted score separately from physical haul.
