# Batched CUDA route completion

The search can now evaluate a batch of forward route costs in the normal CUDA
library. The retained workspace computes mining gains, forward mass, thrust
authority, inflation, rocket-equation fuel and the final dry-mass-plus-cargo
gate. These are **route proxies**; every selected mission still needs low-thrust
refinement and both independent fleet checks.

This preserves the PDHCG-inspired trajectory solver. It removes recurring
arithmetic from the search around that solver; it does not change the conic
method, physics model, payload or mission objective.

## Scope and compatibility

Set `SPACEPDHCG_TEST_GTOC12_COMPLETION_BATCH=1` inside an existing CUDA Lambert
scope to select the new path. It remains off by default: small flat-model batches
regress, and no whole-search or H100 completion measurement is claimed.
Explicit selection requires the completion ABI in
`libspacepdhcg_cuda.so`; a missing library or unsupported model raises an error.

The Python adapter currently requires CPython 3.12 and finite, exact Python
`float` epochs. It rejects overridden numerical or tour-validation methods
that the native model cannot represent. Existing metadata lookup failures are
whole-call input errors, separate from native per-candidate feasibility results.
Host code still constructs route topology, looks up certified return cells and
supplies geometry features. Moving this arithmetic does not make the complete
search controller GPU-native.

Independent candidates occupy CUDA threads. Each thread walks its own short
mass chain in order. The C API retains device buffers, a stream and timing events
across calls; the Python adapter also retains its input and output arrays.
Heuristic schedules, the two DP weight choices and independent shortlisted
chain completions can share a batch. Existing route ordering, first-successful
penalty-round behavior and failure reasons are preserved.

## Numerical contract

Completion uses the original flat, authority-ratio, five-feature hop-fit,
generic-return, table-return and exact certified-return-cell models. Each
flight is priced at its actual forward mass and records the inflation it spends.
The original authority and dry-plus-cargo comparisons remain strict; no cargo
is reduced to make a route pass. Certified return prices retain the authority
check. Repeated pickups and cargo summation preserve the reference's insertion
order and CPython 3.12 compensated-sum behavior.

The native fit uses FP64 without fused multiply-add. Ordinary arithmetic is
checked against the scalar reference within prespecified tolerances; NumPy dot
products and exponentials need not be universally bitwise identical. Acceptance
decisions, failure indices and pickup flags are compared exactly. Numerical
comparison tolerance cannot convert a rejected route into an accepted one.

## Evidence so far

The full CUDA core was compiled from committed `7eb8828f` plus the four frozen
native completion files. Its SHA256 is
`cb977ff09b206de807996a8a21d7ccec41bb6a2a4c35a37c10ebe4883b83d2fb`.
The host Python snapshot has a separate identity. Later unrelated fleet changes
in the working branch are not part of this build.

The RTX 5090 native check passed **304 candidate evaluations in three kernel
calls**: 262 native test cases and 42 saved historical/synthetic controls.
All 3,132 saved comparison fields pass. The largest ordinary finite difference
is `4.547473508864641e-13` kg; exact cargo, boundary and classification checks
pass. These were correctness runs, with no new Lambert requests, trajectory
refinements, physics certifications or fleet promotion.

The production Python adapter also passes all four GPU model tests, with
**eight completion calls and 1,048 candidate evaluations**. Each model processes
259 ragged candidates followed by three candidates in the same retained
workspace. Failure reasons, route/cargo order, mass, fuel and recorded inflation
match the scalar reference within the unchanged comparison contract. All four
workspaces close and the observed Lambert request count is zero. The test
observer saves packed inputs and native readbacks; its instrumented timings
are not performance evidence.

## Measured performance

The bounded RTX 5090 comparison passed **60 GPU calls and 14,200 candidate
evaluations per backend**, with unchanged decisions and numerical parity. It
repeats only 20 distinct historical controls; these are not newly generated
mission candidates. Each model/size group has one first-use call and four warm
samples per backend, with alternating CPU/GPU order. Request construction,
validation and raw capture are outside both method timers and reported
separately. The complete evidence worker took 6.310 seconds.

The table uses warm medians. GPU method time includes normal packing, transfer,
execution and plan reconstruction. A ratio below one is a GPU regression.

| Candidates | Flat CPU / GPU (ms) | Flat CPU/GPU ratio | Fitted CPU / GPU (ms) | Fitted CPU/GPU ratio |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.333 / 0.751 | 0.44x | 1.212 / 0.815 | 1.49x |
| 24 | 1.210 / 1.256 | 0.96x | 4.850 / 1.617 | 3.00x |
| 48 | 2.250 / 1.945 | 1.16x | 9.544 / 2.812 | 3.39x |
| 64 | 2.934 / 2.357 | 1.24x | 13.190 / 3.389 | 3.89x |
| 256 | 10.559 / 6.898 | 1.53x | 49.677 / 11.228 | 4.42x |
| 1,024 | 42.672 / 25.061 | 1.70x | 200.120 / 41.207 | 4.86x |

The historical v616 shortlist setting was 24; the source default is 48, with
pruning potentially producing fewer rows. The four-candidate group uses the
same archived DP requests at a schedule-sized batch, not a measured heuristic
invocation. The larger groups are prospective batching probes. Four warm samples
on repeated controls establish a bounded component comparison, not SOTA or a
whole-search speedup.

At 1,024 candidates, warm method throughput is about **40,860 flat** or
**24,850 fitted proxy evaluations/second**. The fitted batch's average packing
time is 37.760 ms and its average CUDA kernel/event time is 0.101 ms, versus a
41.207 ms median complete method call. These separately summarized phases show
why retained native metadata and route construction should precede another
kernel micro-optimization. All raw samples, first-use costs and setup/close times
remain in the evidence. No certified-solutions/second claim follows from them.
The median of the four paired packing/method fractions is 91.39% for this fitted
batch (86.51% for flat); this avoids dividing unrelated phase summaries.

Keep this mode opt-in while implementing that residency work and measuring the
complete search. The fast arithmetic is useful, but transferring a tiny batch
and rebuilding its Python metadata can cost more than the scalar reference.

## What this means for mission progress

The fixed historical audit identifies a separate search-quality problem. At
archived measured burn masses, the generic return proxy overprices **20 of 23
Earth returns**, by a median **18.36 kg** and a total **519.94 kg** of propellant.
An already-certified historical ship-23 route is still rejected by the corrected
completion model. These observations use historical certificates; they are not
fresh feasibility results for new routes.

Reproducing the same decisions on CUDA cannot correct those false negatives.
The next mission experiment must test better whole-route cost prediction or an
explicit refinement queue for uncertain proxy rejections, with separate held-out
routes and unchanged final physics gates. Broader generation becomes useful
after this admission check succeeds. See the
[SOTA execution plan](SOTA_EXECUTION_PLAN_2026-09-09.md) for the separate solver
convergence and verified fleet-score requirements.

The [complete v621 evidence package](../results/local/2026-09-09/gpu-route-completion-v621/README.md)
retains source snapshots, CPU preparation failures, native and adapter readbacks,
all timing samples, independent reviews and the proposed mass-preconditioning
experiment. Completion was measured locally; the separate Lambda health check
found an idle H100, but no v621 H100 run or new fleet was produced.
