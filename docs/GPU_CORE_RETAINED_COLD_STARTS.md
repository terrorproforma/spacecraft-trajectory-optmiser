# Retained workspaces and cold starts, v630

**PDHCG qualified 0/4 outputs; QOCO qualified 1/4 under the unchanged original-equation gate.** Retaining allocations worked, but did not resolve PDHCG's cold primal convergence. No dual correction was invoked: all four PDHCG points failed the invariant primal eligibility test. These results establish neither a qualified speedup nor SOTA performance.

The [portable evidence package](../results/local/2026-09-09/gpu-core-retained-v630/) preserves the two exact inputs, standalone frontends and frozen headers, build/CPU validation, full raw vectors, timings, process cleanup, both original numerical auditors and saved findings. The inputs are early and near-converged accepted SCvx stages of **one synthetic 76-node fixture**, not independent flown missions. “Near-converged” describes the captured SCvx stage; its new inner solve began at zero.

Four prescribed processes each performed two cold trials. Trial 0 created a workspace; trial 1 retained handles/storage but reset iterates. PDHCG used RESET_FULL before both calls, exact L1 with unit weight, common-KKT stopping, 128 blocks and 10,000 updates per call. QOCO preserved all thirteen snapshot settings, two original numeric updates and disabled primal-start action before each call. There were eight primary solves, 40,000 PDHCG updates, 115 QOCO IPM iterations and 783 aggregate refinement iterations; zero bootstrap calls, correction calls or correction-workspace creations. The first launch attempt encountered a busy lock before any child; it is retained separately. The terminal attempt cleaned up all owned processes and left an empty compute inventory.

Both unchanged Decimal65 and Linux long-double audits agree on every result. Relative primal, stationarity, gap and maximum block complementarity must each be at most 1e-9; absolute cone defects must be at most 1e-8.

| Capture / backend | Trial | Native status | Relative primal | Relative gap | Common qualified |
|---|---:|---|---:|---:|---|
| Early / PDHCG | 0 | iteration limit (2) | 5.49097e-4 | 0.999982 | No |
| Early / PDHCG | 1 | iteration limit (2) | 5.49097e-4 | 0.999982 | No |
| Near / PDHCG | 0 | iteration limit (2) | 1.13900e-3 | 1.000012 | No |
| Near / PDHCG | 1 | iteration limit (2) | 1.13900e-3 | 1.000012 | No |
| Early / QOCO | 0 | solved inaccurate (2) | 2.03672e-9 | 1.60700e-8 | No |
| Early / QOCO | 1 | solved (1) | 5.77592e-13 | 1.65098e-11 | Yes |
| Near / QOCO | 0 | solved inaccurate (2) | 2.64850e-10 | 6.22932e-8 | No |
| Near / QOCO | 1 | solved inaccurate (2) | 6.26385e-8 | 2.69203e-6 | No |

Native status 2 has different meanings in the two backends. QOCO's three inaccurate results retain that status and fail the stricter common gate: early trial 0 exceeds primal/gap by 1.03672e-9/1.50700e-8; near trial 0 exceeds gap by 6.12932e-8; near trial 1 exceeds primal/gap by 6.16385e-8/2.69103e-6. All other QOCO common gates pass. No status or threshold was changed to make the comparison pass.

| Capture / backend | Trial 0 work (s) | Retained trial 1 work (s) | Complete process, both trials (s) |
|---|---:|---:|---:|
| Early / PDHCG | 0.398800 | 0.385750 | 1.172283 |
| Near / PDHCG | 0.395404 | 0.387497 | 1.123803 |
| Early / QOCO | 0.120950 | 0.072812 | 0.522828 |
| Near / QOCO | 0.130620 | 0.107120 | 0.570537 |

Trial work includes reset/updates, solve, readback, conversion and serialization; it excludes shared initial setup and final destruction. Complete process time includes both trials, parsing/loading/context creation, setup and cleanup. Shared PDHCG setup was 0.259526/0.219609 s (early/near). QOCO context, topology, update-workspace and upload setup totaled 0.261000/0.243289 s; parsing and cleanup are separately recorded. These are host elapsed scopes, not kernel timings. The retained trial is not a second complete process or a median. PDHCG kept the same workspace and allocation counter (64) across calls; scaling refreshed on each zero reset. Actual solve times remained approximately 0.374–0.377 s. The allocation ledger excludes vendor/context memory.

The saved original CSC coefficients prove that equality row 526 is exactly **x[1]=0**, the scaled initial departure y-position. Its absolute residual is 0.00220737 early and 0.00479747 near. First-interval y-continuity row 1 is nearly as large (0.00219727/0.00477838), and terminal y-position row 533 also fails (0.00217876/0.00402550). Thus this is a coupled boundary/dynamics feasibility problem, not just a removable zero-target dual coordinate. Worst thrust-cone defects are 4.65427e-6 at node 0 early and 2.35769e-5 at node 40 near. Adjusting y/z while fixing x/s cannot repair any of these failures.

The v623 warm experiment supplied a separately replayed qualified predecessor iterate. Its near-stage primal residual was 2.47117e-10, allowing the later v629 dual correction. No such point entered v630. The larger independent canonical captures inventoried in the design also have preserved primal failures and exceed the present correction workspace limits; no broader applicability is claimed.

The near input, QOCO/cuDSS library hashes, thirteen settings, first setup and per-trial numeric call sequence match the [v629 cold comparison](../results/local/2026-09-09/qoco-exact-comparison-v629/), which qualified once. The new first result differs, and the second trial has retained workspace history. These samples do not identify the cause of the inaccurate-output variation; timing variation is not an explanation or proof of a numerical cause.

The next targeted investigation should address the combined state-continuity/boundary primal block, including the thrust cones and original virtual-control objective. A bounded candidate is the existing exact-equality-projection reference with its explicit L1 dual split: verify its coefficient metric, adjoint, original dual reconstruction and feasibility on these two inputs. Validate those identities before a fixed cold comparison. Earlier projection experiments on larger captures preserved equalities but failed the objective/gap, so projection alone is not a demonstrated cure. Avoid an unchanged rerun or broad parameter grid; retain the original gates and charge complete cost.
