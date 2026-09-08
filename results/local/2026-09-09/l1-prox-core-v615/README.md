# L1 proximal diagnostic: no qualified cold solution

The opt-in exact L1 path preserves both supplied qualified real points at zero
optimization iterations. All four cold runs reach 100,000 iterations and fail
the unchanged original-equation accuracy gate. It remains default off. These
results establish neither a verified-solution speedup nor a better fleet score,
and do not integrate this diagnostic with the production GTOC12 trajectory path.

| Capture | Method | Native solve event seconds | Relative gap | Qualified |
|---|---|---:|---:|---|
| Conditioning | Generic dual-first | 5.424073 | 0.999924306 | No |
| Conditioning | L1 prox | 3.610990 | 1.000290548 | No |
| Difficult | Generic dual-first | 5.549215 | 0.011227050 | No |
| Difficult | L1 prox | 3.656115 | 0.003878469 | No |

These are one measured fixed-iteration comparison per method/capture, on the
same RTX 5090, binary, explicit 128-block strategy and fixed accuracy gates.
The event timing includes native iteration/report work and the existing empty
recovery-event scope; it excludes scaling, importer preparation and independent
CPU audit. End-to-end repeat/setup/scaling times are separate in the saved
records. All four cold runs finish their iteration budgets within the 30-second
deadlines. No cancellation, numerical failure, recovery or retry occurred in
the real comparison. A smaller gap on one case is not full qualification.

The accepted seeds are the original captured QOCO points. Their x/y/z FP64 bits
are preserved, and completion runs zero times before acceptance. Each seeded
executable separately performs the documented one-iteration bootstrap, full
reset, explicit primal/dual seed and residual-only measurement before the
zero-step solve. These two bootstrap updates are counted separately.

The [tiny report](tiny/report.json) records one test process, nine solve APIs and
ten actual updates. Positive/negative/±zero two-step prox and SOC oracles pass;
the mode-off transition matches a fresh original workspace seeded with the exact
GPU output, including restored original scaling. The weak original seed passes
at zero steps. Cancellation wins before initial certification, and a nonfinite
seed fails. Invalid maps/nonzero Q are rejected. Tiny wrapper time 0.495646 s is
not trajectory throughput. The [real report](real/report.json) records six
executions/eight solve APIs, 400,000 cold updates plus two bootstrap updates.
Across tiny and real evidence: seventeen solve APIs and 400,012 actual updates.

The exact maps contain 1470/1631 epigraph pairs. Logical working dimensions are
3791 variables/7811 rows and 4208/8666. Allocations retain the original layouts
5261/10751 and 5839/11928; this is not a compact-memory implementation. The
reduced cone-preserving Ruiz preconditioner includes both smooth c/D and
lambda/D_v in its objective norm. GPU B/O/eta match the independent CPU
calculation to FP64 rounding. The spectral estimate still uses 20 power steps
and is not a proved upper bound or general convergence guarantee.

Original coefficients, objective and audits remain intact. Q must be exactly
zero; selected t columns must have positive cost and only the exact two
zero-RHS rows v-t<=0/-v-t<=0. The GPU map proof rechecks the CPU detection.
The soft threshold uses the actual original-coordinate diagonal step times
lambda. After each update t=abs(v); dual completion uses strict sign for nonzero
v and a clipped retained gradient only at exact zero. No snapping or relaxed
tolerances are used. Common gates remain relative 1e-9 for primal, stationarity,
gap and maximum scalar/SOC block complementarity, and absolute 1e-8 for cone
membership. Original natural residuals remain separate telemetry.

The [CPU analyses](analysis/) include the first failed ARPACK extended-precision
dtype attempt, its corrected FP64 run, and the balance investigation. That
balance study uses approximate projected reference distances, not deployable
weights or newly qualified reduced solutions. Its different behavior between
the two captures does not justify a universal objective-normalization change.

Build a is preserved but was never run on GPU: review found an atomic/non-atomic
race when traversing off-diagonal structural-zero Q entries in its initializer.
Build b removes that dead traversal. Build c changes only the tiny fresh-control
seed to use the exact GPU point and recompiles test/replay metadata against the
unchanged b core. Neither a nor b diagnostic executable was launched; the
corrected b core is the core exercised by c. The initial manual CPU compilation
also missed the project include directory and failed before any solver call;
the corrected strict-warning build and all later recorded CPU stages pass.

Default/common cooperative resources remain 80/96 registers and zero stack;
single-block default/common remain 148/204 registers and 40-byte stack. Existing
experimental Halpern changes 94 to 96 registers with unchanged stack/shared
memory; no runtime-parity claim is made for it. New L1 uses 198 registers and a
40-byte stack; its corrected initializer uses 58 registers/zero stack. The native setter
checks solve/scaling occupancy and the real run confirms the requested 128
blocks. All resource outputs are preserved under [builds](builds/).

Final frozen source commit: `32efbed13eae47d2c8352771e884a2ad5ff5d07e`.
Core SHA256: `83487574646fe67fce156c9a2055f448341c99f189ac2c66d856bfcf1b280047`.
Manifest SHA256: `ee312d9988b4ee9c1ca68d5cc0236d00583eeb637c911af7ca101abe9d90c12a`.
The [ten owned source files](source/cpp/cuda/) match the final archive/live hashes.
Complete source freezes, commands, parser checks and core reuse provenance are
retained under builds/a, builds/b and [builds/c](builds/c/manifest.json).

All six complete raw outputs, including original x/y/z/slack vectors, are stored
in [raw-logs.tar.gz](real/raw-logs.tar.gz) with individual byte hashes in
[raw-members.json](real/raw-members.json). The original inputs and retained
runner are under real/. A fresh [CPU re-audit](real/audit/findings.json) agrees
with every native and original-equation verdict. No compiled binaries or
credentials are included. The package SHA256 index covers every retained file.

To reproduce the saved original-equation audit on CPU, use Python with NumPy
and SciPy from this package's directory; the output path must be new:

```text
python scripts/recheck_saved_outputs.py --output fresh-original-audit.json
```

This reads the compressed raw logs directly and verifies their individual
hashes. It does not load the CUDA library or launch a solver.

The independent [Decimal65 review](real/independent-review/REPORT.md) agrees
with all six gates and both untouched seeds. At the cold L1 endpoints, every
epigraph has t=abs(v) and every removed-pair complementarity product is zero.
The remaining failures are in retained stationarity and equality feasibility;
the difficult capture also fails a retained SOC gate. The smaller difficult
gap therefore does not mean every accuracy metric improved.
