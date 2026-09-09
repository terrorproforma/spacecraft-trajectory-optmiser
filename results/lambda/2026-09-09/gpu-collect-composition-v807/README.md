# Verified H100 fleet plus local ship-8 correction v807

[Result.txt](Result.txt) passes both original full-fleet checkers locally and on
Lambda: **12,999.824843382517 fixed-bonus weighted kg / 14,279.2881587958 raw kg**,
23 ships and 199 asteroids. Its SHA-256 is
`4ae67c220ab7d50f3dad0f3899bd27a3a161a524ecd8323df8d8f33c86a32b5c`.

The fleet combines the new H100 v804 ship-4 route with the locally GPU-refined
endpoint-merit-corrected ship-8 route from v627/v628. Relative to the H100 v799
baseline, the other 21 ship sections are byte-identical. The [plan](plan.json)
pins all three input identities and expected score/mass. There are no new GPU
solves in this composition. Fresh CPU dynamics checks run on both machines;
the H100 machine's verifier produces the stored display propagation.

The archive and receipt bind the downloaded H100 output. `local/report.json`
and `h100/report.json` contain both full-fleet summaries tied to the exact
Result hash. The original official verifier prints rounded total mass; the
independent summary retains the full precision. Original physics tolerances
are unchanged. The 1e-8 kg assertion checks score-reporting agreement only.

The existing visualiser displays this exact H100-based composition, with
11,676 retained replay samples and 74,645 independently checked Kepler context
points. The separately installed RTX-based union v629 reaches the same score;
its slightly different trajectory bytes keep their own identity.
[Open the downloaded H100-based composition](http://127.0.0.1:4173/?dataset=gtoc12-collect-composition-v807&epoch=69807&preset=oblique&z=1).

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs','--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-collect-composition-v807&epoch=69807&preset=oblique&z=1'
```

Full downloaded H100 composition path:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-collect-composition-v807\Result.txt`.

The saved audit rehashes sources and reports, checks the retained ship sections,
and independently sums emitted cargo with the fixed bonus table. It does not
rerun dynamics:

```powershell
python results/lambda/2026-09-09/gpu-collect-composition-v807/audit_saved.py
```
