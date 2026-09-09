# Retained allocations with cold starts, v630

**PDHCG qualified 0/4; QOCO qualified 1/4 under the unchanged original-equation gates.** No dual correction was invoked because all four cold PDHCG points failed the fixed-primal eligibility gate. This is not a qualified speedup or SOTA result.

Two accepted SCvx stages of one synthetic 76-node fixture were tested, not independent flown missions. Four prescribed processes each ran two cold-reset trials while retaining allocations for the second trial. The terminal batch used eight primary solves, 40,000 PDHCG updates, 115 QOCO IPM iterations and 783 refinement iterations, with zero bootstrap/correction calls. A prior busy-lock attempt launched no child and is preserved. All terminal process/compute inventories were empty.

PDHCG used original common-KKT stopping, exact L1, unit weight, 128 blocks and 10,000 updates per call. QOCO retained all thirteen snapshot settings and the original two-update/cold-start/solve order. Native status is preserved: PDHCG 2 is iteration limit; QOCO 2 is solved inaccurate and does not by itself satisfy the stricter common gate.

Complete two-trial process times were 1.172283/1.123803 s for early/near PDHCG and 0.522828/0.570537 s for QOCO. These include loading, context/setup, both solves, transfers, serialization and cleanup. The full table in the archived REPORT.md separates first/retained trial work from shared setup. No average/median/p95 or GPU kernel-time claim is made. PDHCG kept its workspace and allocation counter but still failed initial y-position, linked state-continuity and thrust-cone constraints. Fixed-primal dual correction cannot repair those errors.

Both unchanged Decimal65 and Linux long-double auditors agree on all eight original x/y/z/s readbacks. The archive contains their source dependencies and saved results, exact-FP64 vector hashes, original constraint localization, and the source/settings/runtime comparison against the previously qualified single QOCO run. QOCO's three common-gate failures remain prominent; the small sample does not establish why inaccurate results differed.

The evidence.zip archive uses repository-relative paths. It preserves the isolated frontend/header source archive, complete build and CPU validation, the busy attempt, the terminal supervisor/readbacks/timers, diagnostic helpers, both exact snapshots, source/input maps, and the frozen doc recommendation. Compiled executables/libraries/cuDSS, keys, caches and Git data are omitted. Exact runtime binary paths/sizes/hashes are declared in index.json; the portable audit does not reverify their absent bytes. Historical inventory/preparation scripts refer to prior published evidence and are retained as provenance, not promised standalone launch tools.

Run with Python 3.10 or later:

~~~
python audit_package.py --package . --expected-index INDEX_SHA256 --roundtrip
~~~

This stdlib audit verifies exact safe member/byte hashes, the nested 29-source build, zero-work busy attempt, one-line output-directory continuation, source/runtime/counter/cleanup bindings, every saved raw-vector hash and the 0/4 versus 1/4 qualification result. It executes no archived code, numerical auditor, optimizer, propagation, native library or GPU call. The roundtrip option writes a fresh temporary tree and rereads all byte hashes. Optional --output PATH refuses overwrite.

To independently recompute only the saved numerical audits after unpacking, use Python with NumPy/SciPy and extended-precision Linux long double, hide GPUs, and run:

~~~
python build/performance/gpu-core-retained-v630/audit_saved.py --report-sha256 460e79d92b9b1940af925d408a17a26168b26f42cb6f6411d0f14dc3ee450169 --output NEW_FILE.json
~~~

It imports only the two pinned CPU auditors, invokes no solver or propagator, and never overwrites the saved evidence.

The unchanged numerical libraries and prior experiments are documented separately in ../qoco-exact-comparison-v629/ and ../gpu-dual-isolation-v629/; those packages are dependencies, not new results of this batch. The proposed next investigation targets joint boundary/dynamics primal feasibility while retaining the original virtual-control objective and cones. Earlier equality-projection experiments failed objective/gap on larger captures, so no demonstrated cure or default promotion is inferred.
