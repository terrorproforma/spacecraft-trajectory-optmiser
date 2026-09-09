# Retained CUDA collection workspace — 9 September 2026

Collection dynamic programming now retains its CUDA buffers across changing
tours. The paired whole-search benchmark produces exactly the same candidate
files while reducing workspace creations from **2,184 to 51** over two routes
(**97.7% fewer**). Three measured repeats per mode, after separate warmups, give:

| GPU | Original ship | Fresh median | Reused median | Search throughput gain |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | 10 | 22.160 s | 20.949 s | 5.78% |
| RTX 5090 | 21 | 20.121 s | 19.276 s | 4.39% |
| Lambda H100 | 10 | 20.050 s | 19.391 s | 3.40% |
| Lambda H100 | 21 | 19.253 s | 18.790 s | 2.46% |

These are complete route-search timings, not isolated kernel timings. Runs
alternate fresh and retained modes and share frozen source, inputs, native
libraries and search settings. All eight candidate files per ship match exactly
within each GPU. `SPACEPDHCG_TEST_GTOC12_REUSE_COLLECT_DP=0` selects the original
fresh CUDA allocation path for the comparison; reuse is enabled by default.

The subsequent full-fleet search regenerates **11,007 route candidates** across
23 ships, with **11,737 beam expansions** and **454,295,808 logical Lambert
branch requests**. It takes **512.928 s locally / 480.684 s on H100**, or about
**21.5 / 22.9 candidate routes per second**. These are surrogate search
candidates, not fully certified trajectories. The roughly 0.886 / 0.945 million
logical branches per second are counts divided by whole-search elapsed time,
not a measurement of standalone Lambert kernel throughput. The telemetry also
includes 46 initial seed branch requests outside that search count.

Both GPUs record **531 workspace creations and 24,627 reuses**. This full search
uses a newer input fleet than the paired benchmark, so its elapsed time is not
an additional fresh-versus-retained speedup measurement. Candidate schedules,
cargo, order and body IDs agree across GPUs; 42,077 proxy-number differences are
at most 2.30e-11 in their respective units.

Fourteen proposed routes undergo GPU QOCO/SCvx refinement: **147 native leg
attempts**, with 138 converged, five infeasible and four failed solver statuses.
Five complete routes certify; only the new ship-4 route improves the selected
fleet. An optimizer failure does not establish physical impossibility. Native
call time is 56.243 s on RTX 5090 and 67.102 s on H100; this variable collection
of attempts is not a general solutions-per-second benchmark.

The resulting fleet passes both original full-fleet checkers on both machines:
**12,999.452764 weighted kg / 14,278.850103 raw kg**, 23 ships and 199 asteroids,
gaining **7.417102 weighted kg** over H100 v799. CUDA selection exhausts the
49-column certified pool after 84,238,236 nodes, taking 21.081 s locally /
7.390 s on H100 for the final pass. This proves the selected result within that
finite pool only, not global mission optimality.

The separately certified endpoint-merit correction affects ship 8. Applying
that exact section to the H100 result gives **12,999.824843 weighted kg /
14,279.288159 raw kg**, 23 ships, 199 asteroids and **620.838616 raw kg per ship**.
Both full-fleet checkers accept this exact composition locally and on Lambda.
Relative to H100 v799, only ships 4 and 8 change; the other 21 sections are
byte-identical. Composition performs no GPU optimization and must not be
described as a new H100 run of the corrected endpoint controller.

The C ABI adds updates for host input arrays and immutable resident GPU table
slices. Each shape dimension must fit the creation capacity. A fitting update
performs no device allocation; short-lived host packing vectors still exist.
All numerical inputs and topology are refreshed. Invalid, busy and capacity
errors preserve the previous problem; a failed CUDA upload disables solving
until a successful update. Inputs are copied into owned storage before return,
including resident table slices, so later table eviction cannot change a solve.
The Python owner transfers the handle on success, recreates it on capacity
growth, and retains compatibility with native libraries lacking the new ABI.

Validation passes on RTX 5090 and H100: **55 tests each**, plus **17 tests each
under memcheck, racecheck and synccheck**, with no reported errors. Tests cover
shape shrink/restore, changing topology and policy, fresh/reused exact result
bytes, source mutation and cache eviction, rejected updates, and legacy result
buffer size. Physics equations, DP pricing, precision and acceptance tolerances
are unchanged. The saved-evidence auditor rehashes 1,138 payload files in each
archive and independently recomputes emitted raw mass and fixed-bonus score.

The measured runtime is frozen from commit `9d77fcce` plus the five collection
workspace files. Its native core predates the endpoint-merit change in
`7f07c6b6`; the published branch includes that subsequent change separately.
The ship-8 correction has its own [endpoint-merit evidence](SCVX_ENDPOINT_MERIT.md).
Next, the integrated core needs matched replay on Lambda. Remaining work also
includes repeated catalogue hashing and host packing in completion searches,
Python search orchestration, progress-based convergence recovery, and reliable
PDHCG qualification. This checkpoint does not establish a fully GPU-controlled
C++ mission pipeline. Independent CPU verification remains intentional.

[Measured runtime, tests and H100 fleet](../results/lambda/2026-09-09/gpu-collect-reuse-v806/README.md).
[Combined H100/local fleet and exact verification](../results/lambda/2026-09-09/gpu-collect-composition-v807/README.md).
