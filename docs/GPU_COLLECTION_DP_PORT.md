# Collection-tour DP: measured bottleneck and CUDA port contract

**Update:** the multi-block CUDA implementation and matched campaign measurements
are now documented in [GPU_COLLECTION_DP_CUDA.md](GPU_COLLECTION_DP_CUDA.md).
This document preserves the earlier profiling evidence and original port
requirements. Between-pass mass control and input preparation remain host work.

The complete H100 campaign spends **30.973 s in 378 collection-DP passes**, inside
a **91.430 s** directly timed CLI call (33.9%). Those spans include lazy GPU
Lambert table construction called by the host DP; they are not pure CPU samples.
This is a substantially larger complete-pipeline target than the remaining
millisecond-scale fixed-order retiming work. Collection-tour Held–Karp search in
`collectdp.py` is a separate algorithm from the CUDA fixed-order retimer.

## Evidence

[Downloaded evidence](../results/lambda/2026-09-08/gpu-collect-profile-v272/)
contains three unchanged-source campaign replays (v270/v271/v272), the corrected
campaign v275, raw logs, full reports, direct timers and real collection inputs.
All four campaigns return 548.255 kg and pass both mission checkers. No score
improvement or whole-campaign speedup is claimed in this tranche.

The first cProfile wrapper (v270) did not save its profile because the CLI raises
SystemExit. The corrected wrapper (v271) saves one, but its aggregate accounting
is inconsistent: approximately 322.573 s accumulated against a 109.109 s campaign,
including an anomalous 212.754 s self entry in HiGHS `check_option`. Keep the raw
profile for diagnosis; do not use its totals or percentages as timing evidence.
v272 instead wraps each DP pass with `perf_counter`, without cProfile, and is the
source of the 30.973/91.430 s measurement above. Its branch counters match v269:
44,722,546 transfer branches, 2,782,091 collection options and 106 initial routes.

Eight captured inputs cover two through nine asteroids. The first seven yield
tours; the nine-asteroid case has no proxy tour. Local RTX 5090 replay reproduces
the archived decisions and objectives, then exact-mass replays on both GPUs
match an uncached reference. These are proxy-DP regression inputs, not eight new
independently certified low-thrust missions.

## Correctness prerequisite: exact mass cache

The old fraction cache keyed moves by `(source, target, round(mass))`. Actual
mass affects both thrust authority and calibrated inflation. The original local
fixture replay records 479 different-mass hits among 29,822 fraction calls, with
mass differences up to 0.821355 kg. This is not permission to round the physical
model. A boundary regression demonstrates both failure directions: a heavier
ship can inherit a lighter ship's feasible flag, or a lighter ship can inherit
the heavier ship's infeasible flag.

Code `a02de775` keys the bounded cache by exact mass. Before the change, two
cached boundary tests fail while their uncached controls pass; after it all
four pass. The full collection suite passes 26 cases locally and on H100, and
15 additional local search, harvest-phase and GPU collection tests pass. Both
GPU fixture replays match the uncached reference and record zero different-mass
hits. The largest archived objective difference in the local fixtures is
1.14e-13 kg; selected routes are unchanged.

The complete corrected H100 campaign v275 takes 92.528 s (one observation, not a
matched performance comparison). Its independent checker accepts
548.2546201232 kg, eight asteroids and no violations. Maximum errors are
0.628645 km position, 1.099556e-7 km/s velocity and 7.208e-11 kg mass. The official
checker reports 548.255 kg. The existing viewer retains v269 samples and hashes;
the new replay has its own archived Result.txt. Incumbent fleet score remains
12,805.194 weighted kg.

## Native implementation requirements

The next change should move the complete collection DP and mass-dependent
pricing to CUDA, rather than issuing a Python/CUDA call per state transition.

1. Retain pair/return costs, policies and DP/backpointer buffers on the device.
   Represent state as `(collected subset, arrival asteroid, epoch)`, processing
   subset cardinalities in dependency order across multiple blocks. Handle the
   initial uncollected-camp transitions before their same-layer successors.
2. Compute camping prefix maxima on device. The existing reconstruction chooses
   the **latest equal arrival** at or before departure. This differs from the
   **earliest equal TOF** selected by NumPy argmax, and both rules must survive.
3. Gather incoming transitions per destination state/epoch. The existing
   `candidate > incumbent + 1e-9` rule is order-sensitive and is not equivalent
   to an unordered maximum reduction. Preserve predecessor expansion order and
   terminal departure-major/TOF ordering, with deterministic backpointers.
4. Price at exact subset mass. Preserve burn credits, the dry-mass floor,
   mining duration gates, weighted yield, banned pairs, optional calibrated
   inflation/phase penalties and certified return-sweep masks. Preserve the
   existing float32 pair-table representation when checking semantic equivalence.
5. Keep both mass passes, failed-heavy-pass retry and backtracking on device;
   return compact final results. Compare against the uncached CPU equations,
   then independently certify selected low-thrust missions.
6. Benchmark complete search and campaign time with identical inputs and
   accuracy gates on RTX 5090 and H100. Include infeasible tours, ties, partial
   tiles, alternate TOF grids, calibrated policies and sanitizer coverage.

## Reproduction

With `PYTHONPATH=src`, the pinned catalogue and CUDA library configured:

```sh
python scripts/gpu/replay_gtoc12_collection_fixtures.py \
  results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json \
  build/performance/collection-replay.json
```

The default compares the exact-mass cache with uncached equations. To reproduce
the old audit, pass `--reference archived --mass-key rounded` to
the current script while importing solver source `49fb0d47`. Runtime/source hashes and
SHA-256 manifests identify each observation.
