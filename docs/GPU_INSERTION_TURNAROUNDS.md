# Insertion coverage for later collection revisits

The insertion search used to require a visit that both deploys and collects a
miner. Valid self-cleaning routes can instead deploy at their turnaround, collect
other miners, and return to collect the turnaround miner later. Such routes were
silently given zero insertion schedules.

The shared pivot selection now preserves the existing first combined camp when
one exists, otherwise choosing the last deployment visit. New deployment slots
precede that pivot and new collection slots follow it. CUDA constructs and screens
the resulting layouts using the existing retained buffers. The scalar reference
and both CUDA wrappers use the same pivot selection. No CUDA ABI changes or solver
tolerance changes are needed.

The epoch generator retains its conservative one-year dwell reserve for borrowed
time, including deploy-only pivots. Midpoint seeds need no dwell. Each miner's
actual deployment-to-collection stay is still checked independently of that
reserve. No surrogate result becomes a fleet trajectory without refinement and
full physics verification.

## Wider search, without a new score claim

The v781 baseline and v783 fixed probes start from the v779 certified route
summaries, preserve the other 21 ships' asteroid exclusions, and use the same
calibrated cost model. Route 2297 deploys at asteroid 53527, visits other collection
stops, then returns to collect 53527. It has no combined camp.

| Route | Requested neighbours / radius | Available new asteroids | Evaluated schedules |
|---|---:|---:|---:|
| 1786 | 60 / 2.5 | 58 | 12,992 |
| 1786 | 600 / 4 | 387 | 86,688 |
| 1786 | 3,000 / 8 | 1,848 | 413,952 |
| 2297 | 60 / 2.5 | 30 | 1,890 |
| 2297 | 600 / 4 | 207 | 13,041 |
| 2297 | 3,000 / 8 | 1,356 | 85,428 |

Both GPUs produce identical counts and rejection categories. All six probes find
zero feasible insertions. The widest pair evaluates **499,380 schedules**; the six
probes total 613,991 evaluations with overlapping neighbourhoods. These are
surrogate schedules, not independently certified trajectory solves. Previously
every route-2297 probe evaluated zero schedules. Its zero-dwell pivot enables only
the midpoint seed; the three borrowed-time rows are disabled and excluded from
evaluation counts.

The two widest searches take 2.822 + 1.647 seconds locally and 3.931 + 2.533 seconds
on H100. These are single cold probe observations including neighbourhood lookup,
metadata setup and result handling, not a repeated speed comparison. The existing
neighbourhood routine also retains its coarse orbital-element prefilter; radius 8
does not mean an unrestricted catalogue search.

Thrust-authority rejection dominates (330,309 + 75,426 schedules in the widest
pair), followed by flight-time limits (77,616 + 9,492). This motivates exploring
better insertion epochs and route changes. It does not justify relaxing the
authority, mining-stay, mass or verification thresholds.

## Verification and full mission replay

Both RTX 5090 and H100 pass **126 joint regression tests**, with seven skips for
absent historical fixtures. All 20 insertion/layout tests run under memcheck,
racecheck and synccheck with zero reported errors or hazards. Tests that formerly
returned early for the no-camp fixture now compare generated layouts, every
enabled epoch, result rows, valid mass details and survivor ranking. Synthetic
tests also cover zero dwell, later collection revisits and a route with no deploy.

The v784 end-to-end replays perform 36 native solves and pass both the independent
and official full-fleet physics checks. They retain **23 ships, 195 asteroids,
14,044.353 raw kg and 12,843.556 weighted kg**. Numerical changes below 1e-8 kg from
v780 are not score gains. The raw campaign's `improved` flag compares against its
older v733 input; the separate audit records the change against v780 explicitly.
Of the 36 attempts, 34 converge and two return infeasible; all 33 accepted route
legs converge and certify on CUDA. Complete replays take 44.902 seconds locally
and 69.218 seconds on H100, including 20.782 / 39.016 seconds of independent
verification. These are single replay observations.

The frozen test source is `9b366c8f` plus the six insertion source/test overlays
listed in each source manifest. It uses the unchanged v776 CUDA cores. The
separately published archived-control initializer is not part of these measured
replays.

The default search now performs 17,638 joint evaluations in 182 batches, including
5,138 insertion layouts in two batches and 5,670 disabled seed rows. It computes
251,999 geometry hops and reuses 154 cached hops. Shared insertion input upload is
575,640 bytes; full result and insertion epoch downloads remain 13,687,528 and
6,368,768 bytes respectively. Python still compiles shared metadata, orchestrates
routes and sorts survivors. Full GPU control and survivor compaction remain work.

## Local evidence and display

[Retrieved evidence](../results/lambda/2026-09-09/gpu-turnaround-v785/) contains the
before/after probes, frozen source and GPU libraries, tests, raw native solve
records, mission trajectories, both physics reports, hashes and reproduction
scripts. The scripts preserve the original runtime paths and input dependencies.

The H100 replay is at:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-turnaround-v785\h100\v784\fleet\Result.txt`

There is no improved mission to promote. The existing web visualiser continues
to show the verified v780 fleet:

```powershell
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-layouts-v780&epoch=69807&preset=oblique&z=1'
```

[Full visualiser startup instructions](GPU_INSERTION_LAYOUTS.md#retrieved-h100-result).
