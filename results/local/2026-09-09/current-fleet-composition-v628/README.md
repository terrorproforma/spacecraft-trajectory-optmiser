# Verified local current-fleet composition v628

The exact [Result.txt](Result.txt) passes both original complete-fleet checkers: **12,992.407741224697 fixed-bonus weighted kg**, **14,271.4852840526 raw kg**, 23 ships and 200 asteroids. It improves the pinned [H100 v799 fleet](../../../lambda/2026-09-09/gpu-regeneration-v799/h100-best/campaign-report.json) by **0.37207919333013706 weighted kg** and **0.43805612590040255 raw kg**.

Only ship 8 was replaced. Its bytes come from the certified local [boundary-merit regression v627](../endpoint-merit-and-gpu-dual-v627/README.md); every other ship section is byte-identical to H100 v799. This is a local composition of previously certified trajectories. The score gain was enabled by correcting the GPU SCvx endpoint merit and refining that prescribed route in v627, using GPU QOCO inner solves. It is not evidence that PDHCG produced the new route or that a complete mission pipeline is GPU resident.

The [checker binding](checker-binding.json) ties both passes to Result SHA-256 `63446ebf3ff1298911bccade7b789a8fb68a0f7db4171906c8e6783a7f174bab`. The [independent saved-evidence audit](review/REPORT.md) checks all retained sections, the replacement, source identities, official ScoreData and fixed bonus coefficients. The score-reporting bound of 1e-8 kg checks serialization/arithmetic agreement; all original physics acceptance tolerances are unchanged.

The finite v628 execution performed one independent CPU full-fleet check and one original official check, with no GPU, native solve, search or additional leg certification. The worker took 23.890 seconds: 21.878 seconds for the independent check and 0.507 seconds for the official check. Its hard deadline was 120 seconds, with 10 seconds for termination and 30 seconds for forced reaping. The saved worker's `incumbent_promoted: false` records its execution boundary; the default viewer was switched only afterward, following independent audit clearance.

The visualizer now defaults to **Local fleet composition v628**, retaining H100 v799 and the historical v627 dataset. All 23 stored histories are reused exactly: 22 from v799 and ship 8 from v627. They contain 446 events and 11,676 replay points. There was no new display propagation. JSON export and import took 0.333 and 0.156 seconds separately; the importer also checked 75,006 Kepler context samples. The six served files matched local hashes and schema/JavaScript checks passed. The HTTP audit does not claim a screenshot-based visual inspection.

Open [the current fleet in the existing viewer](http://127.0.0.1:4173/?dataset=gtoc12-current-composition-v628&ship=8&epoch=69807&preset=oblique&z=1).

```powershell
# The viewer is already running.
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-current-composition-v628&ship=8&epoch=69807&preset=oblique&z=1'
```

After a restart, if the viewer is unavailable, start its existing server in a terminal and leave it open:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4173
```

Full Result path: `C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\local\2026-09-09\current-fleet-composition-v628\Result.txt`.

Installed dataset: `C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser\data\gtoc12-current-composition-v628\fleet.json`.

`index.json` covers every package file except itself. Raw text is protected by `.gitattributes`. The two large viewer JSON files are losslessly gzip-compressed; `archive-audit.json` records compressed and original identities and exact roundtrips. The original scripts, reports, intermediate report, source/input bindings, ScoreData, independent audit and display evidence are retained. The official executable, duplicated official Result/catalogue and native binaries are deliberately excluded. Original execution scripts retain their workspace-relative paths and are not standalone numerical rerun recipes; their frozen checker environment and source lineage are linked through v627.

This portable audit requires only Python's standard library and does not run dynamics or either checker:

```powershell
python results/local/2026-09-09/current-fleet-composition-v628/reproduce/verify_package.py
```
