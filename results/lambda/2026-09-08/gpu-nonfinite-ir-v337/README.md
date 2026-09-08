# Nonfinite iterative-refinement guard — development evidence

The GPU controller no longer accepts NaN/Inf corrections or overwrites its finite
backup with them. Invalid initial residuals/tolerances do not launch refinement;
outer solver and physics qualification remain mandatory. The diagnostic host IR
path has matching guards. No qualification tolerances changed.

## Verification

The previous kernel fails the new regression. RTX 5090 and H100 pass 14 controller
cases, 9 norm cases, four controller sanitizer modes, and 38 CLI/trajectory tests.
Eight further cases exercise actual conditional graphs on each GPU, with H100
graph memcheck passing. Probe sanitizer results do not cover all vendor kernels.

Balanced exact-QP replay counts, using the original independent residual/gap gates:

| GPU | Previous graph solver | Guarded graph solver |
|---|---:|---:|
| RTX 5090 | 27/32 | 30/32 |
| H100 | 28/32 | 27/32 |

Qualification failures remain. This fixes the demonstrated NaN acceptance bug;
it does not establish a general reliability improvement or complete the solver
repair. The fleet incumbent remains 12,805.194 weighted kg.

Four complete H100 campaigns run guard/previous/previous/guard in that order.
They take 53.293, 58.885, 59.477 and 52.865 seconds, respectively. Median time
falls **59.181 -> 53.079 seconds (10.31% less time)**. All four return 548.255 kg
and pass both official and independent mission checkers. Two observations per
mode on one configuration are a limited performance result.

`remote/replay/summary.json` contains exact timings and qualification counts.
`remote/validation` retains build, controller, sanitizer, trajectory and actual
graph logs. `remote/replay` and `remote/confirmation` retain full mission outputs,
commands, rejected attempts and raw QP replay vectors. Native library hashes and
the patched prepared-QOCO sources identify the tested implementations.
`local-validation` and `local-replays` contain RTX results. `final-source` and
`source-sha256.json` preserve the repository changes. `retrieval.json` pins the
downloaded archive; `remote/remote-sha256.json` verifies its 131 files.

## Compensated residual experiment remains rejected

The standalone operator in `residual-operator-v334` uses the earlier compensated
GPU kernels. With correct upload ordering, direct and graph execution agree on
all 138 captured linear-system inputs. There are 106 finite inputs. At each
finite input's worst long-double residual row, the GPU result agrees with an
independent 80-digit decimal computation within 2.989e-21. This is a sampled-row
Decimal check, not a claim that every component has that error bound. Full-vector
comparison with long double differs by at most 1.832e-12, consistent with the
separately measured long-double rounding error. Input samples/hashes are retained
in the preceding gpu-execution-v328 diagnostics/linear-v333 evidence.

The first standalone harness omitted completion of pageable uploads before using
a nonblocking stream. That invalid test was discarded; the preserved probe
explicitly orders uploads before direct and graph launches.

Accurate standalone residual evaluation does not produce a reliable solver:
the compensated QOCO graph qualifies only 2/32 exact-QP replays on RTX, compared
with 28/32 baseline replays in the same experiment. `rejected-precise-graph`
preserves the complete results. The compensated prototype remains excluded from
production. Cone-scaling arithmetic and the mismatch between stored WtW and
compact scaling remain investigation leads, not established universal causes.

## Current publication state

This is development evidence. Main has not received this candidate while broader
solver qualification failures remain unresolved. The earlier v332 viewer dataset
retains its own source, samples and timings; it must not be relabelled as v342.
The new final solution is downloaded under
`remote/confirmation/v342/output/fleet/Result.txt`.

The visualiser now includes **GPU IR guard v342 (548 kg certified)** with 510
exact archived samples. Import verifies the solution/catalogue hashes and 3,010
context samples. Dataset checks pass; browser inspection confirms both mission
checker passes and physical 1x geometry.

Open <http://127.0.0.1:4173/?dataset=gtoc12-v342&epoch=69807&preset=oblique&z=1>.

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-v342&epoch=69807&preset=oblique&z=1'
# If the server is stopped, run this and keep the terminal open:
node scripts/serve.mjs --port=4173
```

Full solution path:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-nonfinite-ir-v337\remote\confirmation\v342\output\fleet\Result.txt`
