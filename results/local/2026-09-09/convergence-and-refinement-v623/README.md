# v623 convergence and native-refinement evidence

The numerical and mission tests did not produce a qualified improvement.
The verified incumbent remains **12,843.555695585188 weighted kg** across
23 ships. [Results and implications](../../../../docs/PDHCG_WARM_START_AND_PROJECTION.md).

- Warm transfer: four GPU primary solves, four diagnostic bootstraps,
  40,004 updates, zero qualified primary outputs.
- Equality projection: two fixed CPU policies, four cold solves,
  40,000 cold updates, zero qualified outputs. Virtual L1 costs are normalized
  subproblem objective units, not kilograms.
- Mission control: one native GPU leg attempt, 41 SCvx iterations and
  3,104 reported IPM iterations. Physical certification fails; both new
  candidates remain untested. No replacement fleet or score gain.

`evidence.zip` contains 472 individually indexed files: exact source snapshots,
inputs, raw vectors, controls, reports, failed attempts and independent audits.
`index.json` binds every included file. Two copies of the official checker
executable are omitted; their hashes, invocation records and outputs remain.
The portable audit explicitly reports those 374,400 binary bytes as not
reverified. No compiled solver or SSH identity is included.

The frozen mission worker incorrectly names the count of three returned tuples
`ordinary_proxy_survivors`; the actual number of successful ordinary plans is
zero. Its CUDA certificate's legacy `rk4_vs_dop853_km` field is not a new DOP853
comparison. The saved-data review records both interpretations without changing
raw evidence.

`verification.json` records four successful CPU-only audits from a fresh
extraction. These audit saved evidence and do not launch optimization, GPU work
or the official checker. From the repository root, with Python 3.12 and NumPy:

```powershell
Expand-Archive -LiteralPath .\results\local\2026-09-09\convergence-and-refinement-v623\evidence.zip -DestinationPath .\build\v623-evidence
python -B .\build\v623-evidence\build\performance\warm-review-v623\review_build.py
python -B .\build\v623-evidence\build\performance\warm-review-v623\review_results.py
python -B .\build\v623-evidence\build\performance\core-projection-v623\audit_final_decimal.py --out .\build\v623-projection-audit.json
python -B .\build\v623-evidence\build\performance\refinement-review-v623\portable_publication_audit.py --index .\results\local\2026-09-09\convergence-and-refinement-v623\index.json --root .\build\v623-evidence --output .\build\v623-mission-audit.json
```

The older capture-generation helpers retain references to their historical
archives. The saved-vector audits above are self-contained in this package.
Full GPU reproduction also requires the separately identified CUDA/QOCO
libraries and hardware; it is distinct from this read-only verification.

Archive SHA-256:
`a9d5393a22991a8f486fd1266670a3731debe8af33945596621a8f55204d9d5d`.
