# Return-leg repeatability diagnostic

Six local RTX 5090 return solves all converged and passed the unchanged independent
leg checks. The H100 failure did **not** reproduce. Repeating identical captured
mathematical inputs nevertheless produced different conic results and outer
iteration counts, so the earlier tiny upstream mass differences are not sufficient
to explain repeat variation. The root cause remains unresolved.

| Recorded H100 mass variant | Local outer iterations | Accepted steps | Certified |
| --- | --- | --- | --- |
| candidate1: originally failed | 13, 17, 14 | 7 each | 3/3 |
| candidate0: originally successful | 13, 13, 13 | 7 each | 3/3 |

The two masses differ by 6.200480129336938e-10 kg. Each variant was repeated three
times, alternating in one process. All numerical input arrays are byte-identical
within a variant, and every recorded input hash matches its saved arrays. The
native input hashes exclude the wall-clock start and remaining timeout. Every call
finished in 1.84–2.61 seconds against the unchanged 900-second per-leg limit; none
reported a timeout.

Variation is already visible in the **first conic solve**, before outer acceptance
differs. For candidate1, its first QOCO solve takes 37/36/35 iterations. The first
outer acceptance difference between repeats 1 and 3 is iteration 7: one qualifies,
while the other retries the unchanged subproblem. The run taking 17 outer attempts
still accepts seven steps and independently certifies. Candidate0 also has distinct
first-conic numerical outputs despite all three completing in 13 outer attempts.

The diagnostic took **13.3831 seconds** after setup, including six solves, their
independent leg rollouts and evidence writes. Summed `solve_seconds` is 12.6284
seconds; those measurements include each call's CPU preparation/native invocation.
They are not pure kernel timings or an end-to-end optimizer speedup.

A separate CPU audit re-propagated all six saved trajectories with the frozen
source and ordinary thrust clamp. All six passed the original pipeline's normalized
0.5 check; every certificate value exactly matched the run's certificate. This
audit took 0.5446 seconds of rollout work. No new GPU calls were made by the audit.
This is a leg diagnostic, not a newly verified full-fleet submission or score gain.
The H100 second route, when it succeeds, scores below the retained v595 fleet.

Potential mechanisms remain hypotheses. The native QOCO path retains thread-local
solver resources and rebinds them between legs at
`cpp/cuda/src/gtoc12_qoco.cu:366`, with restart before reuse at line 391. Vendor
factorization graphs are retained at `cpp/cuda/patches/qoco_factor_runtime.cuh:58`.
The frozen QOCO540 source excerpt in `vendor-source-inspection.json` has no explicit
deterministic-mode configuration. These observations identify paths to isolate;
v598 neither compares fresh workspaces nor captures the first conic KKT matrix,
vendor factorization or generated GPU seed, so it does not locate the cause.
The project's [earlier determinism diagnostics](../../../../docs/QOCO_DETERMINISM_DIAGNOSTICS.md)
include rejected settings and GPU memory errors; they are not a validated workaround.

`report.json` preserves all six outcomes and settings. `replay-audit.json` validates
source/fixture hashes, input-array equality, first divergence and fresh independent
rollouts. `audit.json` is the preceding H100 comparison. `raw.tar.gz` contains 52
files: all local captured inputs, solutions and histories, CPU preparation captures,
and the eight original H100 reports used for comparison. `raw-sha256.json` lists
every archived member's byte count and hash.

The driver and fixture are separate from the unchanged frozen source at commit
`2408ca9b80f3f1d1041861c3cfebc27613114047`. The shared source archive is
[collection-order-v597/source.tar.gz](../collection-order-v597/source.tar.gz),
SHA-256 `693efcb7b80ae3a6bca34016cdb7a6858f583e51761cf0f3f70c0e3aa25499c8`.
`source-reference.json` and `source-sha256.json` preserve that linkage.

Core SHA-256: `86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671`.
QOCO540 SHA-256: `0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315`.
No numerical tolerances, production code, native geometry or incumbent fleet were
changed by this diagnostic. [Preparation and replay instructions](preparation.md)
retain the original bounded six-call recipe.
