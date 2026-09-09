# Verified 24-ship fleet, 9 September 2026

The fleet now returns **14,915.044490 raw kg** for **13,526.961241 fixed-bonus
weighted kg**, with **24 ships and 208 asteroids**. This adds **503.256340
weighted kg (3.86%)** to the preceding frontier. Raw cargo averages **621.460187
kg per ship**. Both original full-fleet checkers accept the exact final file.

The added ship was recovered from a complete archived trajectory in
`fleet_master_v10`, where it was ship 9. Its nine asteroids are disjoint from
the preceding fleet. The original 23-ship Result remains an exact byte prefix;
the new ship's identifier changes to 24. After the first official checker
rejected a terminal newline as an empty EOF line, only that final CRLF was
removed. Both checker pairs and all original row values are preserved.

This is an improvement from recovering and selecting an existing route. There
were no new trajectory optimizations, GPU calls or PDHCG solves. The two fresh
checker pairs took 47.144632 seconds of summed worker time, including the first
format rejection. This is verification cost, not optimization throughput.

The raw-mass rule uses the average for the proposed fleet size. For 24 ships,
the required aggregate raw cargo is 14,909.439899 kg, leaving **5.604591 kg** of
margin. The next fleet change must satisfy this updated budget as well as the
weighted objective and asteroid conflicts.

If those 24 routes remain unchanged, a 25th ship would need at least
**870.759537 raw kg**: the 25-ship total must reach 15,785.804027 kg. Improving
several existing routes can reduce that requirement. This is a concrete reason
to optimize fleet composition and timing jointly rather than repeat independent
ship additions.

[Final Result and complete evidence](../results/local/2026-09-09/fleet-addition-v633/README.md)
include both full checker reports, official per-asteroid mass output, frozen
checker source, archival provenance and a portable saved-data audit.

Final Result SHA-256:

```text
1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da
```

Full local path:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\local\2026-09-09\fleet-addition-v633\Result.txt
```

The viewer dataset is `gtoc12-fleet-v633`. It retains the 23 preceding ship
records exactly. Ship 24 displays its 20 archived event states and explicitly
marks the 19 unsampled intervals as gaps. The dense history for that archived
ship was not recovered; connecting those event states would imply unsupported
trajectory geometry. Existing sampling gaps in the preceding records remain.

Copy into PowerShell to open the dataset:

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath 'node' -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-fleet-v633&ship=24&epoch=69807&preset=oblique&z=1'
```

The [performance assessment](PERFORMANCE_POSITION_2026-09-09.md) separates this
fleet gain from the numerical-core experiments and states the remaining plan.
