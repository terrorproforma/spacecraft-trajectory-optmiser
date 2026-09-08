# Bounded fleet refinement recovery

The wider H100 v374 campaign stopped after three ships. Its first five candidates
for ship four all failed near the same early transfer, although the search had
198 candidate routes. Local SCvx failure was being treated as sufficient reason
to stop constructing the fleet. It was not evidence that the remaining routes
were infeasible.

An independent RTX 5090 diagnostic preserved all 198 plans and tested different
opening prefixes. Candidate 26 returned 419.028063 kg and passed both mission
checkers. [Diagnostic and exact archived inputs](../results/lambda/2026-09-08/gpu-fleet-recovery-v379/summary.json).

## Implementation

`--refine-top` retains its initial score-ranked attempts. If none passes the
mission checks, `--refine-recovery` allows up to 16 additional attempts by default
(zero disables recovery). Recovery first tries the highest-ranked representative
of each unseen first-three-leg prefix, then the other remaining candidates in
their original order. A prefix includes the bodies and exact departure/arrival
epochs, including camp legs. It is a scheduling hint, not an infeasibility test.
No route is permanently filtered because another route failed.

With CUDA screening selected, two CUDA kernels identify unseen prefixes and
compute a deterministic permutation across multiple blocks. The queue is created
only when initial refinement fails. Python packs the small prefix array and
dispatches attempts; this is still a CPU-orchestrated application. Explicit NumPy
screening uses the reference queue implementation. CUDA mode has no silent CPU
fallback when the new native symbol is unavailable.

Recovery stops after its first accepted route, its attempt limit, or the run's
wall-clock budget checked before the next recovery attempt. The wall-clock budget
does not interrupt an already running refinement. The original SCvx, conic,
thrust and physics tolerances are unchanged. A solver-certified candidate must
also pass the independent mission checker and the official checker, when
available, before entering the fleet or its candidate columns.

## Validation

The combined harvest/recovery implementation passed 100 selected tests on the
RTX 5090 and H100. The recovery suite exercises multiple blocks through 1,025
candidates, changed inputs, duplicate prefixes and preserved candidate ranks.
Compute Sanitizer ran the five recovery tests with zero errors on each GPU.
Six additional CPU integration cases passed locally after adding explicit
checker-rejection coverage; they verify bounded attempts and prevent a large
but rejected score from entering the fleet. These extra cases do not alter the
production code used by either GPU campaign.

The complete local four-ship campaign finished in **467.689219 seconds**:

| Result | Value |
|---|---:|
| Ships | 4 |
| Asteroids collected from | 29 |
| Physical returned mass | 2,095.961670 kg |
| Fixed-bonus weighted score | 2,088.668592 kg |
| Official fleet checker | Pass |
| Independent fleet checker | Pass |

For ship four, initial ranks 0–4 failed, followed by recovery ranks 11, 16 and 21.
Rank 26 passed and was retained. The fleet score is 419.028063 weighted kg above
the previous three-ship result, but remains below the separate 12,805.194 weighted
kg incumbent. This is a successful recovery demonstration, not a matched speed
comparison or a new leaderboard score.

[Complete local evidence, source snapshot and binary hashes](../results/lambda/2026-09-08/gpu-fleet-recovery-v380/summary.json)
are archived. The H100 v381 wider campaign completed using the same production
source in **3,419.051083 seconds**. It generated 32 individual routes and 74
candidate columns, evaluating **1,416,714,796 transfer branches** and
**137,350,630 collection options**. Fleet selection returned **15 ships and
105 mined asteroids**, with **7,820.533881 kg returned** and **7,802.295160
weighted kg**. Both final fleet checkers pass.

The fleet master closes its bound for this candidate pool; that does not prove
optimality over the full mission search space. More individual routes do not
automatically permit a larger fleet: the ship-count rule also depends on average
returned mass. The best route per search slot gives a top-16 average of about
516.8 kg, below the 519.9 kg required for 16 ships. Improving haul per ship is a
remaining search problem. This run does not improve the 12,805.194 incumbent.

[Full H100 results and checksums](../results/lambda/2026-09-08/gpu-fleet-recovery-v381/summary.json)
include the frozen source archive, complete raw reports, commands and binary
hashes. Retrieval verified all 326 archived files. The separate final-status fix
also ensures an assembled fleet rejected by either available checker exits with
failure and cannot be labelled as scored; three integration cases cover that
gate, independently of per-route solver certification.

## Display and reproduce

The existing web visualiser includes **H100 recovery v381**. Its 7,617 trajectory
samples match the archived export exactly; 39,735 Kepler context points were
cross-checked. The displayed geometry uses physical 1× vertical scale. The earlier
four-ship **GPU recovery v380** dataset remains available.

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4173
# Open this URL, or select "H100 recovery v381" in the dataset menu:
# http://127.0.0.1:4173/?dataset=gtoc12-v381&epoch=69807&preset=oblique&z=1
```

Exact solution file:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-fleet-recovery-v381\fleet\Result.txt
```

The archived `run_recovery_v380.py` records the environment, commands, source
hashes and runtime hashes. Rebuild the native library and run
`tests/test_gtoc12_refinement_queue.py` and `tests/test_gtoc12_run_refinement.py`
with the regular CUDA test environment before reproducing the fleet campaign.
