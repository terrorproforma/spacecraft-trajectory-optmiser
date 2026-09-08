# Reciprocal L1 weight: no qualified cold solution

Both original supplied points remain qualified at zero optimization iterations.
All four cold runs reach 100,000 iterations and fail the unchanged original
accuracy gate. This experiment remains default off. It establishes no qualified
solution speedup, fleet-score change, SOTA claim or production GTOC12 integration.

| Capture | Weight policy | Native solve event seconds | Relative gap | Qualified |
|---|---|---:|---:|---|
| Conditioning | unit | 3.5465415 | 1.000290548 | No |
| Conditioning | cancel-global | 3.7551157 | 0.05150589395 | No |
| Difficult | unit | 3.6972158 | 0.003878468805 | No |
| Difficult | cancel-global | 3.7706675 | 0.05652660104 | No |

These are four single fixed-budget measurements on one RTX 5090, using the same
frozen binary/core, explicit 128-block grid and original common-KKT gates. All
finish inside their 30-second cold deadlines. There is no retry, recovery,
cancellation or numerical failure in this real batch. Solve-event timing
excludes scaling, importer setup, diagnostic downloads and independent CPU
audit; those scopes are separately recorded. A smaller gap alone is not full
qualification. The conditioning gap improves and the difficult gap worsens.

The [tiny report](tiny/report.json) records two processes, 22 solve APIs and
22 optimization updates. Unit and fixed .25/4/cancel-global scalar/SOC/prox
oracles, exact mode-switch versus fresh-control behavior, untouched weak seeds,
cancellation and nonfinite/effective-step failure checks pass. These vector
oracles execute inside the tiny test; its raw logs export metrics, not all
vectors, so they are not independent raw-vector replay evidence.

The [real report](real/report.json) records six executions/eight solve APIs,
400,000 cold updates and two separately counted one-step seed bootstraps.
After each bootstrap, a full reset, explicit original primal/dual seed and
residual-only check precede the zero-step accepted solve. Supplied original
x/y/z FP64 bits are preserved. Across tiny and real: 30 solve APIs and 400,024
actual updates. None of the failed preparation attempts below ran GPU solvers.

The explicit fixed mode uses a caller-provided positive omega. The separate
`cancel-global` mode chooses omega=O/B on-device after reduced scaling, once per
solve. It reads coefficients only, with no reference solution, pilot solve,
adaptive update or parameter sweep. Native-scaled base steps are eta/omega and
eta*omega; original diagonal factors then give mathematical eta/D² and eta/R²
for cancel-global. Lambda uses the actual original-coordinate primal step.
The same reduced D/R/B/O and eta heuristic are retained. The mathematical step
product is unchanged; the 20-power-step norm estimate is still only a heuristic.

Original coefficients, objective, dual completion and common gates are intact:
relative 1e-9 for primal equations, stationarity, objective gap and maximum
scalar/SOC block complementarity; absolute 1e-8 for cone membership. Natural
residual telemetry remains a separate legacy unweighted diagnostic. Original
seeds are checked before any completion. Weight-record validity does not imply
KKT qualification. Invalid weights/steps fail without silently changing them;
cancellation wins before acceptance. Each fresh L1 enable restores unit policy.

The exact L1 maps remain 1470/1631 pairs, logical dimensions 3791/7811 and 4208/8666
variables/rows. Full original 5261/10751 and 5839/11928 layouts remain allocated.
The experiment does not compact memory or change the production trajectory
backend. The [coefficient-only derivation](analysis/MATH.md) and its reproducible
checks document the policy and its mathematical limits.

Existing default/common cooperative resources remain 80/96 registers and zero
stack; single-block variants remain 148/204 registers and 40-byte stack. Halpern
remains 96 registers. Unit and nonunit L1 templates now use 94 registers and zero
stack, versus the prior L1 kernel's 198 registers and 40-byte stack. The initializer
remains 58 registers. This compiler/resource change prevents a claim of byte or
runtime parity with the old build; this comparison uses the same new core in
both arms. The native setter checks occupancy without silently reducing 128 blocks.

The tested source is an uncommitted frozen tree, not a synthetic Git commit:
`e476bbf3065173ed7b06fca49306537ffe1d47804c281bacd441e5a42104e1a9`. The assembly base is v615c commit
`32efbed13eae47d2c8352771e884a2ad5ff5d07e`; workspace parent was 9c8f2e96. Later fleet changes
are separate work. Every archived source member and all ten [owned source
files](source/cpp/cuda/) match the retained manifest. Replay metadata reports
`source_commit=uncommitted`, explicit source scope, tree SHA and base identity.
Core SHA: `1002c69e2418ab8b4ba1cdb7376f2959b8af455ae7ea4815db751508009fceec`.
Manifest SHA: `6ad3dfc586c457afbeaa0da13417f1208484f908fb22977fd086ea41a4f3a4c2`.

The first manual CPU compilation failed from a local test variable-name collision
and was corrected. Build a compiles and passes parser checks but CMake's source
Git lookup fails for its Git-free archive. Its `git_operations:0` field intended
zero mutations; read-only CMake identity queries did occur. Build b adds explicit
frozen-tree source provenance without bypassing pinned third-party checks and
reuses the exact a core. A b wrapper assertion then mistook expected parser
rejections for failures; its pre-continuation manifest and exact continuation
are preserved. These are preparation failures, not GPU solver attempts.

All six complete raw outputs, original x/y/z/slack vectors and metadata are
losslessly stored in [raw-logs.tar.gz](real/raw-logs.tar.gz), with per-member
hashes in [raw-members.json](real/raw-members.json). Inputs, exact runner and
pinned auditor are retained. The fresh [original-equation re-audit](real/audit/findings.json)
agrees with every runtime result. No compiled binaries, caches or keys are
included; excluded local CPU binaries are identified by hash.

The independent [Decimal65 review](real/independent-review/REPORT.md) agrees
with all six original-equation verdicts and both untouched seeds. The package-local
`scripts/recheck_saved_outputs.py --output /absolute/path/new-audit.json` can
recheck the compressed outputs on CPU with NumPy/SciPy. No solver/CUDA loading
is performed. The final flat SHA256 index covers every retained file.
