# GPU terminal recovery and unscaling

Experimental v125 completes numerical terminal handling inside the retained
initialisation/IPM graph. After the IPM WHILE exits, a device kernel decides
whether to restore the best iterate. The existing parallel restoration kernel
copies the selected vectors, and a parallel kernel unscales primal, equality
dual, cone slack and cone dual vectors. Host dispatch no longer reads status to
select or launch this terminal numerical work.

The recovery decision shares the previous implementation: numerical-error and
maximum-iteration exits restore a valid best iterate, and upgrade the status to
inaccurate-solved when its saved metric is at most one. Successful exits retain
their current vectors. Dual unscaling reads the current device `kinv`; explicit
rounded multiplications preserve the original two-operation arithmetic order.
The terminal nodes depend on completion of the entire IPM WHILE.

Legacy callers still receive completed host status/count metadata and, unless
device I/O is selected, downloaded solution vectors. The API skips its old
restoration/unscaling block when the graph has already performed that work.
Initial topology/setup/analysis, one-time vendor warm-up and full SCvx dispatch
still involve the host. Default builders remain unchanged. No new fleet result
was produced.

## Preparation and ablation

Apply after the [v124 preparation chain](QOCO_GPU_INITIALIZATION.md):

```sh
python scripts/gpu/prepare_qoco_ipm_terminal.py --destination /path/to/isolated/source
```

Compile CUDA with `--default-stream per-thread` and select
`SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1` inside the existing queued scope. Presence of
`SPACEPDHCG_TEST_QOCO_IPM_TERMINAL_DISABLE` preserves the host terminal path.
Changing this mode invalidates the retained graph. Existing initialisation,
cache, control and audit restrictions remain in force.

## Verification

- Enabled and disabled IPM modes each pass the native controller and 51
  trajectory integration tests; seven convergence/failure tests pass.
- PD6 N20 and N500 pass independent certificates and the unchanged 1e-8
  reference-objective limits. Errors are 9.776e-11 and 1.110e-16.
- A 32-case interleaved two-workspace QP probe changes cost scales, coefficients,
  RHS, tolerances, iteration budgets, warm starts and regularisation. It also
  exercises iteration-limited exits followed by successful solves. Status,
  objectives, IPM/refinement counts and **all returned primal, slack and dual
  vectors** match exactly across v124, host-loop v125, host-terminal v125 and
  captured-terminal v125. Successful vectors additionally pass independent
  analytic checks. The probe uses host reference Ruiz setup.
- A second run toggles terminal mode on each update. It retains exact parity
  and rebuilds each workspace 16 times for 16 replays, as required. Stable-mode
  runs build twice for 16 replays, including a static-regularisation change.
- A direct probe replays the prepared terminal kernels across 240 cases with
  five statuses, valid/missing best iterates, metric values below/at/above the
  upgrade threshold, changing cost scales and four vector layouts. State and
  every vector bit match an independent reference; allocation canaries remain
  intact. Normal execution, memcheck, initcheck and synccheck pass. Memcheck
  reports zero errors and zero leaks. The probe's initial compilation failed
  from an ambiguous nested initializer list; its source and failure are retained.
- All 12 updates of the analytic pure-SOC probe pass the existing 1e-8 objective
  and feasibility gates in all three graph variants. These results are not
  bit-identical. The stricter precision failure documented for v124 remains
  unresolved and was not rerun here.

Full v125 solver memcheck still aborts with CUDA 999 in the one-time initial
conditional-refinement warm-up, before combined graph execution. It reports
110 errors and 108 outstanding allocations after abort. Isolated terminal
sanitizer success does **not** qualify the full solver.

## Complete-transfer measurements

Six balanced triples use the same v107 GPU seed/SCvx core, with a warm-up and
measured transfer per process. All 36 transfers pass independent physics and
the unchanged 1e-5 kg gate against 2445.3111007852112 kg final mass.

| Variant | Median measured complete transfer |
|---|---:|
| v124 captured initialisation/IPM | 321.820 ms |
| v125 host terminal ablation | 402.392 ms |
| v125 captured terminal handling | 315.516 ms |

This does not establish an additional end-to-end speedup over v124: timings
vary substantially on the display RTX 5090 with unlocked clocks. Warm-ups and
outliers are retained. These measurements use external complete-transfer wall
time. QOCO's internal timer now includes captured terminal operations, whereas
the previous timer stopped before terminal work; those internal timings have
different boundaries.

The [v125 checkpoint](../artifacts/performance/qoco-ipm-terminal-v125-checkpoint.json)
embeds four sources, prepared source, 12 runtime hashes, 41 helpers and 15
evidence reports. All six changed prepared files reproduce exactly; Ruff passes.
SHA-256: `9950610d8dd22a087b0e38d7609d773b0fb51eb16b6e190c89e1b72619101ad2`.
Lambda's read-only retry confirmed its existing H100 campaign at 100% utilisation;
the initial missing-key SSH failure is also retained. No work was offloaded.
