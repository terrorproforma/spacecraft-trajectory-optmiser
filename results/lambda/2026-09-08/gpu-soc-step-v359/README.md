# SOC step correction — development evidence

The GPU line search no longer falsely returns zero for the captured strictly
interior SOC in `live-step-v351/step-0000.bin`. A 100-digit independent
calculation permits 0.8768290586498959 for cone 230 and 0.5534743033528263 for
the complete vector (limited by cone 226). `live-step-v351/report.json` retains
every cone's reference result. The old function fails case 13 in the new
regression; the corrected header passes all 14 cases and output canaries.

Only the line-search coefficients and discriminant use compensated arithmetic.
Normalization, the original problem, physical checks and acceptance tolerances
are unchanged. This is one demonstrated numerical defect fixed, not a completed
solver repair or an entirely GPU-resident application.

## Results

| Final packaged build | Baseline qualified QPs | Corrected qualified QPs |
|---|---:|---:|
| RTX 5090 v358 | 28/32 | 31/32 |
| H100 v359 | 28/32 | 28/32 |

Both final GPUs pass 38 CLI/trajectory tests. Both pass the 14-case SOC probe
under memcheck, initcheck, racecheck and synccheck. These sanitizer runs cover
the probe, not all vendor solver kernels. Full QP replay vectors and independent
original-equation audits are retained. Qualification failures remain.

The earlier step-only prototype has the same arithmetic as the final header.
Its four matched H100 campaigns took:

| Run | Mode | Complete runtime | Result |
|---|---|---:|---|
| v354 | Corrected | 53.581 s | 548.255 kg; both checkers pass |
| v355 | Baseline | 53.541 s | 548.255 kg; both checkers pass |
| v356 | Baseline | 54.669 s | 548.255 kg; both checkers pass |
| v357 | Corrected | 52.434 s | 548.255 kg; both checkers pass |

Median **54.105 → 53.007 s**, or **2.03% less time**, in this small comparison.
The final packaged library's separate v360 confirmation takes **54.010 s**,
returns **548.255 kg**, and passes both mission checkers.

The verified fleet incumbent is unchanged at **12,805.194 weighted kg**.
Main has not been updated with these development changes while the broader
solver qualification work remains open.

## Evidence and reproduction

- `final-source` and `source-sha256.json`: exact repository correction,
  preparer, regression and preparation entrypoint.
- `lambda-final-v359`: final source preparation, library hash, complete
  independent QP audits and raw vectors, sanitizer/trajectory logs, v360 mission.
- `lambda-comparison-v353`: prototype source/binary hashes, full QP replays
  and four complete timing campaigns. `failed-launch.log` records a launcher
  syntax error before any build or GPU work; the corrected launcher was run
  only after that process was confirmed terminal.
- `local-final-replays`, `local-final-validation`, `local-header-validation`:
  RTX replays, trajectory tests, old/new regression, sanitizers and full
  pinned-source preparation/provenance check.
- `live-step-v351`: the actual small-step input, independently calculated
  bounds and GPU results. Synchronous observation is not performance evidence.
- `diagnostics`: rejected determinant/normalization combinations, live NT
  inputs, independent reference, two original trace snapshots and hashes for
  all 36 snapshots. Full rejected-experiment replay logs remain locally under
  `build/performance`; this directory retains their per-replay audit summaries.
- `recipes`: the exact build/replay/audit scripts. These use the existing frozen
  source/runtime paths and the pinned QP identified in their manifests; they
  are not a fresh-machine installer. The original QP SHA-256 is
  `14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080`.

Large non-mission files are losslessly gzip-compressed. Decompress `*.gz` to
their original names when replaying a recipe or checking a downloaded remote
manifest. `evidence-sha256.json` hashes files as stored; remote manifests hash
the original uncompressed bytes. Retrieval records pin the downloaded archives.

Rejected NT normalization matches a 100-digit reference far more closely but
does not improve full QP qualification. It is not in the production header.
The next diagnosis must address the remaining invalid/stalled directions and
their recovery, rather than treating improved scalar arithmetic as sufficient.

## Display the final mission

The new dataset contains 510 exact archived samples. Import checks its solution
and catalogue hashes, plus 3,010 Kepler context points. It is separate from v342.

Open <http://127.0.0.1:4173/?dataset=gtoc12-v360&epoch=69807&preset=oblique&z=1>.

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-v360&epoch=69807&preset=oblique&z=1'
# If the server is stopped, run this and keep the terminal open:
node scripts/serve.mjs --port=4173
```

Full final solution path:

`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-soc-step-v359\lambda-final-v359\v360\output\fleet\Result.txt`
