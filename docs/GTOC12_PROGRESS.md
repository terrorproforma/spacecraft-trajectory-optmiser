# GTOC12 score progress — 9 September 2026

The latest verified checkpoint is **12,842.970672 weighted kg / 14,043.750856 raw
kg**, across 23 ships. GPU fleet exchanges gained **32.834719 weighted kg**;
33 native leg solves and both complete-fleet physics checkers pass on both GPUs.
[Current result, measurements and visualiser](GPU_FLEET_EXCHANGES.md).

The subsequent retained CUDA workspace accelerates repeated fixed-pool fleet
selection by **2.03× locally / 2.32× on H100**, to **50 / 54 ms** after setup.
The score is unchanged. [Measurements, tests and retained API](GPU_FLEET_WORKSPACE.md).

The latest backend also computes fleet scores, eligibility and topology on CUDA.
One-shot selection improves to **72 ms locally / 87 ms on H100** with identical
full-pool score and search counts. Both GPUs pass 115 tests.
[GPU setup implementation and evidence](GPU_FLEET_TOPOLOGY.md).

## Earlier orphan-recovery checkpoint

The remainder of this page records the preceding checkpoint and its comparisons.

The retained best fleet has improved slightly. The lower per-ship figures in
recent GPU pilots describe separate trial fleets; they did not replace the
stronger 23-ship incumbent.

| Verified metric | Previous incumbent | Current incumbent | Change |
| --- | ---: | ---: | ---: |
| Ships | 23 | 23 | 0 |
| Physical returned material | 14,047.802875 kg | 14,051.854894 kg | +4.052019 kg |
| Physical material per ship | 610.774038 kg | 610.950213 kg | +0.176175 kg |
| Fixed-bonus weighted score | 12,805.194102 kg | 12,810.135953 kg | +4.941851 kg |

The gain comes from collecting an abandoned miner on ship 15. That adds
12.484600 kg from asteroid 19102 and loses 8.432580 kg through retiming other
pickups. The other 22 ships remain unchanged. Both complete-fleet checkers accept
the result; a fresh independent audit also passes. This is a **0.038593% weighted
score improvement**, not a major closing of the gap to the leaders.
[Verified result and per-asteroid audit](GPU_ORPHAN_RECOVERY.md).

## Why score progress has been slow

Much of the recent work accelerated numerical operations and moved existing
search arithmetic onto CUDA. That makes more exploration affordable, but does
not by itself discover better asteroid combinations. The richer GPU family pilot
produced a three-ship fleet with 1,587.269 raw kg, approximately 529.090 kg per
ship, versus the retained fleet's 610.950 kg per ship. It demonstrates a working
search path, but offers no incumbent improvement.
[Family search evidence](GPU_FAMILY_REPRODUCTION.md).

The successful change expanded the search to a previously uncollected miner.
The following experiment tried 61 collection orders around that improved ship,
evaluating 49,286 timing candidates in 286 batches. Its search stage took
2.157 seconds; none of the evaluated alternatives passed the surrogate screen
for improvement in both raw and weighted payload, so no further route was refined.
Its 20.762-second baseline independent
and official checks dominated the 23.108-second campaign timer. These times
exclude initial imports, loading, hashing and parsing.
[Complete negative result, source and work counts](../results/local/2026-09-09/collection-order-v597/).

The H100 replay also exposed a missed feasible refinement: one return leg failed
after conic gaps narrowly missed the unchanged numerical qualification gate.
Six local replays of its two recorded initial masses all passed independent
physics checks, but identical mathematical inputs took 13–17 outer iterations.
That establishes variability in convergence. It does not reproduce the H100
failure or prove its underlying cause. The affected alternative scores below the
incumbent even when successful, so fixing this particular failure would improve
reliability without immediately raising the record.
[Reconstructed inputs, exact local replay and audit](../results/local/2026-09-09/return-repeat-v598/).

The next score work needs broader asteroid, deployment and collection choices,
with fleet selection driven by independently verified weighted payload. Repeating
the same sampled starts and order moves or weaker pilot cannot establish progress.
Solver robustness remains part of that work so feasible improvements are not
discarded merely because a refinement failed to converge.

## What the speed figures mean

The measured joint timing-candidate stage is 7.58–8.88 times faster on RTX 5090
and 10.92–12.81 times faster on H100 for the tested incumbent neighborhoods.
Those are surrogate/controller timings. They do not measure independently
certified trajectories per second or an equivalent whole-campaign speedup.
The representative H100 recovery replay took 137.600 seconds, including checks
and export, and reproduced the improved fleet. It updates the targeted route
and retains the other historical routes; it does not construct all 23 ships
from scratch. Python orchestration, metadata preparation and independent CPU
checks remain, so the whole application is not yet GPU-native C++.
[Local measurements](GPU_JOINT_CANDIDATES.md),
[H100 campaigns and timing qualifications](GPU_JOINT_ITINERARY.md).

The score is weighted total delivered material, while kg per ship measures raw
physical haul. Both matter: retaining the current routes and adding a 24th ship
requires another 857.585005 raw kg just to satisfy the ship-count rule. Better
existing routes reduce that hurdle. Our fixed-bonus score would currently sit
ninth if inserted into the published competition table, 295.626 weighted kg below
ATQ. This is a retrospective comparison, not an official submitted position.
[Leaderboard comparison and scoring qualifications](../README.md#gtoc12-score-versus-the-published-leaderboard).

## Display the downloaded H100 result

Select **H100 GPU itinerary v642** in the existing web visualiser. Its downloaded
solution has SHA-256
`e408ff46f547890aaaa8d65505b969829ff9497b0264bf7b558f1bf1d6b9fb17`.
The full path is:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-joint-itinerary-v642\h100-best\Result.txt
```

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath 'node' -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-joint-v642&epoch=69807&preset=oblique&z=1'
```

The viewer displays the retained verified fleet, weighted score, raw haul and
H100 replay scope separately. Historical datasets remain selectable.
