# Causal mass and compact route completion — v622

The causal mass experiment did **not** produce a qualified cold solve and was
slower on both captured problems. Compact completion improved the measured
component at larger batch sizes, with regressions at smaller sizes. Both options
remain explicit experiments. The final combined library passed its bounded
integration smoke. This tranche produced no new trajectory certification,
mission gain, or SOTA result.

All GPU work recorded here ran locally on the RTX 5090. There were no Lambda
runs. The mission baseline in the experiment documentation is the frozen
v622/v733 baseline; subsequent fleet work is separate.

## Original-equation mass result

The comparison used one frozen library, explicit 128 cooperative blocks, the
same two captures, and unit-weight exact L1 treatment in both arms. Each cold
case completed its 100,000-iteration cap within its 30-second deadline. Every
returned vector was independently audited in the original coordinates, including
iteration-limit outputs. The original 1e-9 primal/dual/gap/block-complementarity
and 1e-8 cone thresholds were unchanged.

| Capture | Policy | Native solve seconds | Original normalized primal | Original normalized dual | Original relative gap | Qualified |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Conditioning | Unit L1 | 3.596469 | 1.739190e-4 | 2.247180e-4 | 1.000290548 | No |
| Conditioning | Mass + unit L1 | 7.376578 | 5.074707e-5 | 1.005723e-7 | 0.229510083 | No |
| Difficult | Unit L1 | 3.659972 | 5.464090e-6 | 1.888358e-5 | 0.003878469 | No |
| Difficult | Mass + unit L1 | 7.648103 | 3.105017e-5 | 1.502402e-8 | 0.087891389 | No |

Both supplied original certificates were accepted at zero seeded iterations,
with their original x/y/z bits preserved. The real batch had six executions and
eight solve API calls, including two separately counted bootstrap updates and
400,000 cold updates. These are solve timings, not end-to-end trajectory timings.
See the [real report](mass/real/run/report.json),
[independent review](reviews/mass-real/REPORT.md), and
[API/math documentation](docs/GPU_PERSISTENT_CAPTURE_REPLAY.md).

The new path proves a causal mass chain, retains every virtual control and affine
constant, and applies structured prefix/suffix operations. It reconstructs
original mass equality duals before L1 pair completion. Its direct diagonal
metric uses conservatively rounded coefficient sums and tied SOC row steps;
the independently checked bound is at most 0.9025 on these captures. That bound
concerns represented coefficients and stored steps, not cumulative iteration
roundoff or convergence speed. Full original-layout storage remains allocated.

## Compact completion result

The benchmark used the same frozen production Python method and compact-g
component library for both arms. The method timer includes Python request
packing, source/model hashing and validation, native work/transfers, and plan
reconstruction. Catalogue geometry came from the pinned catalogue. Saved delta-v
values were reused; no Lambert or SCvx solve ran.

| Model | Batch | Ordinary warm median ms | Compact warm median ms | Ordinary / compact |
| --- | ---: | ---: | ---: | ---: |
| Flat | 24 | 1.225 | 1.860 | 0.658x |
| Flat | 48 | 1.973 | 2.203 | 0.896x |
| Flat | 256 | 7.081 | 5.822 | 1.216x |
| Flat | 1024 | 26.693 | 18.776 | 1.422x |
| Existing fit | 24 | 1.773 | 1.831 | 0.968x |
| Existing fit | 48 | 2.689 | 2.196 | 1.225x |
| Existing fit | 256 | 10.839 | 5.775 | 1.877x |
| Existing fit | 1024 | 40.945 | 18.966 | 2.159x |

Each group has one first call and four warm calls per arm in one worker process.
First-call setup/import effects are recorded separately; they are not independent
cold-process trials. All 80 calls and 27,040 repeated proxy evaluations derive
from 20 unique historical controls. At fitted batch 1024, the median matching-call
packing share was 84.6% and the median CUDA event interval was 95.776 microseconds.
The worker's 7.536-second duration includes evidence writing and must not be used
as the production method denominator. The component benchmark does not measure
the final combined library's performance.

All saved classifications, cargo and pickup decisions matched; ordinary floating
arithmetic matched the bounded FP64 checks. The independent reviewer checked
all 80 raw arrays against the frozen formula. See the
[complete measurement analysis](completion/analysis/README.md),
[numerical summary](completion/analysis/summary.json), and
[independent review](reviews/benchmark/REPORT.md). These ratios must not be
multiplied by earlier v621 results from a different benchmark.

## Admission and finite execution record

The optional admission queue retains fully costed final-mass proxy failures for
a bounded fixed-cargo refinement attempt. Admission is not feasibility or a fleet
certificate. Its evidence contains 34 fresh scalar CPU controls plus 20 reused
historical readbacks, and a later, distinct final integration revision with 76
passing CPU tests. No queue trajectory refinement ran in this tranche. The
[queue contract](admission/HOOK_CONTRACT.md) requires downstream physical and
whole-fleet qualification before accepting a gain.

| Evidence | Actual scope | Outcome |
| --- | --- | --- |
| Mass native tiny | One process, seven solve APIs, five updates | Passed |
| Mass real | Six processes, eight APIs including two bootstraps | Two seed passes; four cold failures |
| Compact initial e | Six valid calls, 1,554 candidate evaluations | Failed test attempt preserved |
| Compact second f | Twelve valid calls, 1,572 evaluations, three malformed calls | Partial/failed attempt preserved |
| Compact lock-busy c | No worker and zero evaluations | Busy outcome preserved |
| Compact final g | Seven pytest cases, 18 valid calls, 2,358 evaluations, six malformed calls | Passed |
| Packing benchmark | 80 calls, 13,520 repeated evaluations per arm | Passed; mixed timing result |
| Combined smoke | Mass seven APIs/five updates; legacy completion two batches/262 cases; compact 18 valid calls/2,358 cases plus six malformed calls | Passed |

Native tiny logs report metrics and summaries; their full-vector oracles execute
inside the pinned tests. They are not raw exports of every asserted tiny vector.
Six compact NPZs contain 786 paired saved candidates; later capture assertions
account for additional test calls. The combined review found those 786 pairs
numerically identical to standalone results, and its mass log was byte-identical
to the earlier tiny log. See [combined report](combined/smoke/report.json) and
[review](reviews/combined/REPORT.md). No cold 100,000-step comparison was repeated
on the combined library.

## Source scopes and preservation

[source-scopes.json](source-scopes.json) distinguishes the following identities:

- Original mass build: base fdf52ae3 plus eleven owned overlays; source tree
  `8d72fb3e441dfcdfcabe28d65d0e7d0a959e099622c05dfb8b5e93d1c83039a7`.
- Compact-g: twelve owned native/Python/test overlays, with its independently
  frozen full 341-file host tree. This host tree is separate from compiled C++.
- Combined build: published base 258e5c3a plus fourteen owned C++/CMake overlays;
  source tree `cb0aed789a3fc788f296a80818818c35b3149fb017aba6e0d3c6df88ec805078`.
- Queue `source/` is the earlier revision used for scalar controls;
  `implementation/` is the later integration revision. Their exact maps remain
  separate in the portable manifest.

These source trees are uncommitted frozen snapshots, not synthetic Git commits.
All owned source bytes matched the tested maps at sealing. Binary identities,
compiler flags and resource records are preserved, but compiled binaries,
object files, caches, Git directories and keys are excluded. Persistent kernel
resource records show the combined build retains the tested mass/default-path
footprints; this is not a runtime-parity claim.

Every original source archive is stored once by its SHA in `sources/archives/`.
Repeated materialized source revisions share content-addressed bytes in
`sources/objects.tar.gz`. Raw large logs/NPZs are stored losslessly in
`raw/readbacks.tar.gz`, with member sizes and hashes. The
[portable manifest](portable-manifest.json) maps every logical original file to
its stored location; original filenames, report bytes and historical paths are
preserved. Run materialization before executing original kit scripts or following
their original relative links. Historical machine paths in logs are evidence,
not required portability dependencies.

All compact a–g source/build attempts remain, including recovered exact source
bytes from the failed a attempt. Earlier style/audit attempts, raw failures, the
zero-call busy attempt and the benchmark's pre-retention preparation are kept.
`package-validation/` also retains packaging failures: a missing helper, an
inventory guard catching changing benchmark preparation, exclusive-create audit
output handling, and a CPython 3.10 float-sum mismatch. The final check uses the
original CPython 3.12 sum semantics and passes.

The initial frozen documentation snapshots are in `docs/`. A final roadmap
snapshot in `docs/publication/` labels the v622 fleet baseline as historical and
links to ongoing fleet progress; it does not attribute later gains to this work.

## Portable saved-data recheck

From this package directory, with CPython 3.12:

```sh
python3.12 -B reproduce/portable_causal_mass_v622.py --package .
python3.12 -B reproduce/recheck_causal_mass_v622.py --package . --out /a/new/local/directory
```

The second command creates a fresh tree and runs eight saved-data audits
sequentially: source/fixture, tiny, original mass KKT/metric, compact revisions,
compact arrays, admission identities, benchmark arrays and combined outputs.
The wrapper blocks native library loading and nested process execution in its
audit children. It performs no trajectory evaluation, solver call, GPU work or
Git operation. The offline proof recovers all original baseline source bytes and
checks both mass and combined overlay maps. The final run passed under
CPython 3.12.13; its [report and logs](package-validation/final/report.json) are
included. Original build-review Git evidence is retained but is not rerun.

[sha256.json](sha256.json) indexes every other physical package file. It excludes
only itself; archive/member hashes and portable source maps provide the nested
file identities. The package is complete and sealed only when `STATUS.json`
records that state and the index verifies.
