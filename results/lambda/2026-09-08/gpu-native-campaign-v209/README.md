# Corrected CUDA campaign v209

The H100 run completed in 77.276 seconds, evaluated 18,484,006 Lambert branches
and 2,782,091 collection/return options, and refined three candidate routes.
Two candidates passed both the official checker and independent replay. The
selected six-asteroid mission returns 475.975359343 kg; its 13-leg refinement
took 6.346 seconds. The existing 23-ship fleet remains unchanged.

Open the corrected result in the running viewer:

[GPU campaign v209](http://127.0.0.1:4173/?dataset=gtoc12-v209&epoch=69807&preset=oblique&z=1).

Full local solution path:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-native-campaign-v209\v209\output\fleet\Result.txt
```

If the viewer is stopped, paste into PowerShell and keep the terminal open:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4173
```

Then paste this address into your browser:

```text
http://127.0.0.1:4173/?dataset=gtoc12-v209&epoch=69807&preset=oblique&z=1
```

Select Ship 1, rewind the Epoch slider and press Play mission to follow the
deploy/collection sequence. The dataset selector also offers the pre-fix v200
mission and the incumbent fleet. The link uses physical vertical scale (1x).
Displayed tube segments connect archived samples; they are not additional
integration samples.

`v209/report.json` records commands, changed source hashes and runtime hashes.
`v209/output/run_report.json` records all candidates and checks, including the
remaining failed later leg. `evidence.tar.gz` preserves the downloaded run and
`retrieval.json` records its verified digests. The remote repository commit is a
temporary build snapshot, not a GitHub commit: the source base is `712714f3`,
with the two changed files identified by their SHA-256 hashes in the report.
