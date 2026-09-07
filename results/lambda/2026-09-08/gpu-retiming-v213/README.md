# Certified 524 kg mission

The H100 re-timing of v209 returns **524.024640657 kg**, a **10.0949%** increase
over 475.975359343 kg. It keeps the same six asteroids and changes the schedule.
All 13 flight legs pass both the official checker and independent verification.
Retiming takes 0.703 seconds; GPU trajectory refinement takes 8.139 seconds.
These are one-run measurements. The 23-ship incumbent remains 12,805.194 weighted kg.

The downloaded solution is:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-retiming-v213\v213\output\Result.txt
```

Open the already running viewer at:

http://127.0.0.1:4173/?dataset=gtoc12-v213&epoch=69807&preset=oblique&z=1

Select **GPU retiming v213 (524 kg certified)** in Dataset. The scene uses 508
exact verifier propagation samples, at physical 1× vertical scale in this link.
Click Ship 1 for the mission sequence, or Play mission for playback.

To restart it, paste this into PowerShell:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4173
# Open in your browser:
# http://127.0.0.1:4173/?dataset=gtoc12-v213&epoch=69807&preset=oblique&z=1
```

`v213/report.json` records refinement and both checks. `v213/hop-v210` and
`v213/hop-times-v211` retain the failed-leg diagnosis and timing sweep.
`retrieval.json` hashes the original downloaded files; `evidence-sha256.json`
also covers the locally generated viewer artifacts and import inputs.

The mission was subsequently refined with the new GPU endpoint-table path;
that independently checked result is retained in
`../gpu-elements-v215/h100/mission-v216/output/Result.txt`.
The viewer intentionally identifies the displayed v213 archive explicitly.
