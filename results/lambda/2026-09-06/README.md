# Lambda results — 6 September 2026

Downloaded and SHA-256 verified 1,028 files at 2026-09-06T03:26:06 UTC.

Open the running visualiser:
http://127.0.0.1:4178/?dataset=gtoc12&epoch=69807&preset=oblique

Full local result directory:
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06

The visualiser is a standalone copy of the existing web/trajectory-viewer, with the latest Lambda fleet imported. Previous viewer data remains unchanged.

To restart it later, paste into PowerShell (Node.js is already installed):

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4178
```

Keep that terminal open and visit the URL above. If it is already running, just use the link. Do not double-click index.html: the viewer loads its data over local HTTP.

Select GTOC12 fleet. Drag to orbit; scroll to zoom; select a ship in the left rail. Use the Epoch slider to rewind to 2035 and Play mission to animate. The link opens the completed mission at 2050. Set Vertical exaggeration to 1x for physical geometry; the viewer defaults to 6x for visibility of inclinations. The inspected tab has already been set to 1x.

Files:
- fleet_master_v11/fleet/Result.txt: official solution file, SHA256 f0688310d1a2fcd572e35ada7ec8b10edca1136ccf8b57abd2979b6614104f91.
- fleet_master_v11/fleet/fleet.json: fleet summary, official and independent verifier results.
- fleet_master_v11/fleet/viewer/: original Lambda trajectories and manifest.
- visualiser/data/gtoc12/fleet.json: imported browser dataset, SHA256 0ebd0dfaf0b483418e8ebd261eac2671528cac283429216777ecda79a55deb93.
- catalogue/: pinned asteroid catalogue.
- g4-partial/: finished attempt outputs, consistent SQLite SQL dump, completed events and hardware data. 145 completed groups, one active at capture. 594 numerical dispositions and 711 timeouts; these are not accuracy-qualified successes. This is not the final campaign report.
- download-manifest.json: each downloaded file's original remote path, byte size and SHA256.
- download-verified.json: verified download summary.
- lambda-results.tar.gz: original compressed snapshot.

The fleet has 23 ships, 196 visited/deployed asteroid locations, 194 collected asteroids, and 14047.802874743327 kg collected (official rounded score 14047.8 kg). Both verifier reports pass. It is not proven optimal.

Viewer import checked the catalogue and solution hashes, manifest, fleet event masses, exact replay samples and Kepler context. It displays 11,677 exact replay samples. Kepler maximum differences: asteroid 3.59e-6 km, Earth 3.07e-7 km. Existing viewer check.mjs passed. Browser inspection confirmed the selected fleet, verifier pass and active WebGL2 on RTX 5090.

Lambda remains occupied by G4. An isolated sm_90 gather benchmark was compiled at /home/ubuntu/spacepdhcg-offload/gather-v82-20260906; its readiness guard refused GPU execution while the campaign was active. No new H100 GPU benchmark results are claimed, and no running campaign was interrupted.
