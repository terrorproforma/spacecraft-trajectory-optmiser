# Verified current fleet composition v628

This local composition replaces only ship 8 in the H100 v799 fleet with the independently certified local v627 route. Both original full-fleet checkers passed the exact composed Result. It carries 14,271.4852840526 raw kg and scores 12,992.407741224697 fixed-bonus weighted kg, with 23 ships and 200 asteroids.

The viewer is now set to this composition by default. H100 v799 and the historical v627 regression remain available. All 23 replay, transcription and event histories are reused exactly from those certified datasets; no new propagation, GPU work or optimization was performed for this display. Exporting took 0.333 seconds and importing took 0.156 seconds, separate from the 23.890-second CPU checker worker.

Open <http://127.0.0.1:4173/?dataset=gtoc12-current-composition-v628&ship=8&epoch=69807&preset=oblique&z=1>. The existing server returned six exact local assets; schema and JavaScript syntax checks passed. This is HTTP/data verification, not a browser screenshot assertion.

The published Result is `C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\local\2026-09-09\current-fleet-composition-v628\Result.txt`. The installed dataset is `C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser\data\gtoc12-current-composition-v628\fleet.json`.

```powershell
# The viewer is already running; open the verified composition.
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-current-composition-v628&ship=8&epoch=69807&preset=oblique&z=1'
```

After a computer restart, if port 4173 is no longer serving the viewer, run this in a terminal and leave it open:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4173
```
