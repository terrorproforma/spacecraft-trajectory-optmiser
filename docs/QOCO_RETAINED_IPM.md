# Retained GPU IPM graphs and resources

The experimental v123 extension retains the complete GPU IPM graph across
successive solves on a fixed-topology QOCO workspace. A 64-byte device parameter
record supplies current cost scaling, stopping tolerances and IPM/refinement
iteration limits before each replay. Numerical matrix and RHS values continue
to use their existing device buffers. Static regularisation changes invalidate
the graph; they are never silently frozen at capture time.

This removes graph reconstruction from ordinary SCvx updates, but the latest
complete-transfer measurements show **no additional speedup over v121**.
Initialisation, initial setup/analysis, final reporting/restoration and full
SCvx dispatch still involve the host. No new GTOC12 fleet result was produced.

## Resource ownership

Each solver workspace retains its own cuBLAS handle, 32 MiB cuBLAS workspace,
scalar scratch, parameter record, iteration counter and graph. During IPM
execution it temporarily installs those resources into the queued operator
API's thread-local slots. Completion restores the enclosing caller's resources,
which remain responsible for initialisation and terminal reporting. Nested
scopes and interleaved workspaces therefore do not free or overwrite graph
operands. Workspace destruction waits for completion, destroys graphs, then
releases the retained buffers and handle before cuDSS teardown.

Execution is bound to the creating host thread and CUDA device; migration and
recursive entry are explicitly rejected. This is a per-workspace cache, not a
cross-thread global graph cache. QOCO's fixed-topology ownership contract keeps
captured vector/matrix addresses and dimensions stable. The graph also checks
the owning workspace, control-state address and static regularisation before
reuse. Changing topology requires a new QOCO workspace.

Apply the postprocessor after the complete v121 chain:

```sh
python scripts/gpu/prepare_qoco_ipm_cache.py --destination /path/to/isolated/source
```

Compile prepared CUDA sources with `--default-stream per-thread`, and select
`SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1`. Normal builders remain unchanged. The optional
`SPACEPDHCG_TEST_QOCO_IPM_CACHE_DISABLE` switch rebuilds the graph each solve while
retaining resources for a controlled ablation. `SPACEPDHCG_TEST_QOCO_IPM_CACHE_TRACE`
prints build/replay totals at destruction. Presence enables either diagnostic
switch. The original IPM mode's queued-scope and audit restrictions still apply.

## Evidence

Both v122 and v123 pass the native controller executable, 51 trajectory
integration tests, seven convergence/failure tests and independently certified
PD6 N20/N500 trajectories under the unchanged 1e-8 reference-objective gates.
v123 also passes all 51 integration tests with the IPM graph disabled. Its PD6
reference-objective errors are 4.141e-14 and 9.715e-10.

The original 16-case QP probe builds one v122 IPM graph for 16 replays. A new
32-case probe alternates two workspaces, changes coefficient/RHS values,
tolerances and iteration limits, changes static regularisation, and exercises
maximum-iteration exits followed by successful solves inside nested scopes.
All printed objectives, statuses and IPM/refinement/final-step counts match
exactly across v121, host-loop v123, uncached v123 and cached v123. Each cached
workspace builds twice for 16 replays; the uncached variant builds 16 times.

The scaled version explicitly enables three reference Ruiz iterations during
setup/update and produces **32 distinct cost-scaling factors**. It preserves
the same exact four-way parity and cache build counts. The initial unscaled
probe changed objective coefficients but kept `k=1`; it alone would not have
tested changing captured cost scales. The scaled probe's setup uses the host
reference API, not a claim that Ruiz setup has moved to the GPU.

A separate resource-lifetime probe executes cuBLAS operations with two retained
workspaces across eight enclosing-scope lifetimes, checking scratch identity,
data retention and restoration of the caller's scratch and handle. Normal
execution and CUDA memcheck pass with **zero errors and zero leaked allocations**.
This isolated test covers the new resource ownership. Full v123 solver memcheck
still aborts in the initial conditional refinement graph with CUDA 999, before
entering retained IPM resources; it reports 104 errors and 102 outstanding
allocations after abort. The full solver is **not sanitizer-qualified**.

## Complete-transfer measurements

Six balanced triples ran a warmup and measured transfer per process through the
same v107 GPU seed/SCvx core. All 36 transfers passed independent physics checks
and the unchanged 1e-5 kg gate against 2445.3111007852112 kg final mass.

| Variant | Median measured complete transfer |
|---|---:|
| v121, rebuild whole IPM graph per solve | 332.363 ms |
| v123, retained resources, graph cache disabled | 387.147 ms |
| v123, retained resources and graph | 340.241 ms |

These measurements do not establish an additional end-to-end gain over v121.
The local RTX 5090 drives the display and clocks are unlocked. All warmups and
outliers are preserved. An earlier v122 timing run partially overlapped the CPU
build of v123; it is retained as qualified trajectory evidence and excluded from
the performance conclusion. The v123 experiment ran after compilation and GPU
correctness tests completed.

The [v123 checkpoint](../artifacts/performance/qoco-ipm-cache-v123-checkpoint.json)
embeds six current sources, both prepared runtime variants, binary hashes,
helpers, full validation/timing reports, failures and the read-only Lambda
check. All nine changed prepared files reproduce exactly after formatting.
Lambda's H100 remains occupied at 100% by its existing campaign.
