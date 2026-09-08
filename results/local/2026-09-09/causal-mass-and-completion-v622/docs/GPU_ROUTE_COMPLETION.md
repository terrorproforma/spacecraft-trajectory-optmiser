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

The additive `SPACEPDHCG_TEST_GTOC12_COMPLETION_NATIVE_MODEL=1` experiment moves
pair geometry, model selection, deployment-index lookup and certified return-cell
selection into CUDA. Use it together with the completion-batch switch. The
ordinary metadata path remains available for comparison. The native model is an
immutable device snapshot of catalogue elements, coefficients and return grids;
compact requests carry only route topology, epochs, DV and input inflation.
The Python controller still constructs those requests and checks source-array
hashes on every batch. Those costs belong in complete-method timing.

This experiment requires a real catalogue in official identity order. It derives
geometry from those elements and does not support synthetic values injected into
the geometry cache. Source-array/model changes rebuild the retained snapshot;
there is no route-identity cache. Return-grid preparation retains the existing
lazy behavior, which may compute Lambert values when a new certified sweep is
first converted. Such preparation must be measured and cannot be described as a
geometry-free run. Existing resolved grids require no new Lambert work.

The v622 compact-path correctness run now passes seven tests with no skips,
including three GPU model cases: 18 valid API calls evaluate 2,358 candidates,
and six malformed calls preserve caller outputs. Saved expanded metadata differs
by at most 1.43e-14; final mass/result and independently recalculated forward
costs differ by at most 4.55e-13 kg. Gate decisions, stages and cargo are exact.
The first attempt exposed unused metadata fields being populated differently;
the second reached all numeric comparisons but failed while constructing an
overflow test. Both are retained with their raw results. Another launch found
the shared GPU lock busy and started no worker. These are correctness
results, with no new Lambert, low-thrust refinement or fleet promotion.

The final combined-library integration check also passes the seven compact tests,
the existing 262 native completion fixtures and the seven small mass-solver calls.
This build contains committed `258e5c3a` plus the fourteen reviewed native/CMake
overlays; the full library SHA256 is
`6b32c2b5c4f87cd8990ee95816b00c0cb5dc863c7225cb244a17993ef344d8b1`.
It verifies that both additions coexist with the published fleet improvements.
It does not repeat or replace the isolated performance/convergence measurements.

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

## Historical v621 evidence

The v621 CUDA core was compiled from committed `7eb8828f` plus the four frozen
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

## v621 ordinary completion performance

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

## v622 compact requests versus ordinary GPU completion

The bounded RTX 5090 comparison passes 80 native evaluations: 13,520 repeated
candidate rows per backend, 27,040 total, drawn from only 20 historical controls.
Each group has one first call and four interleaved warm calls per backend. Both
arms use the same compiled completion source/library and real catalogue. The
compact method includes request packing, source/model validation, lazy snapshot
creation, native execution and plan readout. Input construction and evidence
capture are outside the method timers and recorded separately.

| Candidates | Flat ordinary / compact (ms) | Ordinary/compact | Fitted ordinary / compact (ms) | Ordinary/compact |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 1.225 / 1.860 | 0.66x | 1.773 / 1.831 | 0.97x |
| 48 | 1.973 / 2.203 | 0.90x | 2.689 / 2.196 | 1.22x |
| 256 | 7.081 / 5.822 | 1.22x | 10.839 / 5.775 | 1.88x |
| 1,024 | 26.693 / 18.776 | 1.42x | 40.945 / 18.966 | 2.16x |

These are warm medians of complete method calls. The fitted 48-row case takes
18.3% less time; its 1.225x reciprocal is a throughput ratio. At 1,024 fitted
rows the compact path reaches about 54,000 repeated proxy evaluations/second.
The median packing share of the matching warm calls is 84.6%, including
catalogue/model hash validation; the median native kernel/event interval is
95.8 microseconds.
Host topology and Python object construction remain concrete work to move into
the retained native search controller.

The first shared CUDA workspace creation takes 170.274 ms and is recorded
outside the method timer. First-method samples include lazy setup and depend on
call order, so their ratios do not establish a cold-start advantage. All saved
comparisons pass, with a maximum finite difference of 4.55e-13 kg and unchanged
decisions. The evidence worker takes 7.536 seconds. Small flat batches regress
and fitted 24-row batches are slightly slower, so compact requests remain opt-in.
This experiment performs no fresh Lambert solve, trajectory refinement, fleet
certification or score improvement; it establishes a component benefit at larger
batches, with no whole-search or H100 speed claim.

The [v622 evidence package](../results/local/2026-09-09/causal-mass-and-completion-v622/README.md)
includes all attempts, raw timing/readback files, exact source identities,
independent recalculations and the final combined-core integration checks.

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

## Bounded refinement of uncertain completion estimates

The optional `search.completion_capture` hook runs before a completed native
final-mass rejection is discarded. `CompletionRecorder` copies the complete
result, per-leg and per-deployment readbacks into an immutable envelope, together
with the exact cargo, schedule and input/model provenance. Only fully costed
outcomes with consistent finite costs and cargo can reach its consumer. Earlier
authority, mining or inflation failures are excluded.

`RefinementAdmissionQueue` retains at most 48 distinct prescriptions and permits
at most four claims, including at most two final-mass proxy rejections. Ranking
uses the smallest estimated deficit, then weighted gain; there is no fitted
acceptance threshold or blanket fuel correction. The consumer must prune its
envelope mapping to `queue.retained()` after every offer. A failed solve consumes
its claim. Archival control queues cannot launch refinement.

`run_refinement_queue` connects these claims to the CUDA fixed-cargo refiner. It
makes one pass through the prescribed flight schedule and preserves requested
cargo, including cargo below the mining maximum. It does not shrink the payload
or retime a route to obtain acceptance. The caller must supply the baseline fleet
and a verifier callback that emits the actual Result file and runs both full-fleet
checkers. Promotion requires matching Result identities, unchanged prescribed
events/cargo, a certified complete route, an improved fixed-bonus weighted score,
and the raw-mass ship-count rule. Proxy eligibility alone changes no score.

The refiner/executor remains explicitly selected. Its numerical implementation
uses the existing native SCvx/QOCO runner; success here does not establish a
PDHCG convergence advantage. The PDHCG core experiment is documented separately
in [persistent capture/replay](GPU_PERSISTENT_CAPTURE_REPLAY.md).
