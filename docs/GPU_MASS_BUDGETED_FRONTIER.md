# Verified mission frontier — 9 September 2026

The latest fleet scores **13,023.704901 weighted kg** and returns **14,291.006160
raw kg**, with 23 ships and 199 asteroids. Physical haul is **621.348094 kg per
ship**. Both original complete-fleet checkers pass, with unchanged physics and
numerical thresholds. Only ships 8 and 19 change from the preceding combined
fleet; the other 21 Result sections are byte-identical.

The improvement is **23.880058 weighted kg / 11.718001 raw kg**. The search had
previously excluded some alternatives because their individual ship returned
less raw material. The new finite batch considers their contribution against
the fleet's shared raw-mass budget. Selected ship 8 gains approximately 12.676890
weighted kg while losing 12.813142 raw kg; the complete fleet remains feasible.
Weighted competition score and physical cargo mass are separate quantities.

Four already generated, fixed-cargo alternatives were refined with the native
CUDA SCvx controller and QOCO on RTX 5090. All four qualified. There were no new
Lambert evaluations, retries, retiming or cargo reductions in this batch.

| Work or elapsed scope | Observed value |
| --- | ---: |
| Native flight solves | 72 |
| SCvx iterations | 286 |
| GPU flight / wait certificates | 72 / 6 |
| Archived first-leg seeds / native cold starts | 4 / 68 |
| Complete batch worker, including checks | 43.832 s |
| Independent CPU fleet checker within that worker | 21.431 s |
| Separate final composition and checker pair | 22.804 s |

These are unpaired measurements of a finite refinement batch. They do not
measure a from-scratch fleet search or establish a speedup. The final composition
reuses the two newly certified ship sections in the newer fleet and performs
no additional GPU solve. The latest result has not been rerun on H100.

The experiment retains 153 CPU ephemeris calls in orchestration and emission.
Search control and independent CPU certification also remain. The complete
application is therefore still progressing toward fully GPU-native C++/CUDA.
The separate PDHCG correction experiment is a numerical milestone; this fleet
gain does not demonstrate PDHCG convergence or a performance advantage.

## Next search decisions

Retain fleet-level raw-mass feasibility when ranking weighted improvements.
Extend route construction beyond fixed Earth-leg seeds and asteroid sets,
including compatible exchanges between ships. The final raw-mass rule permits
24 ships, but a feasible additional route and its actual cargo contribution
still have to be found and certified. More iterations over an exhausted route
pool cannot create those opportunities.

Measure best verified weighted score versus complete elapsed time from the same
incumbent. Count candidate generation, failed refinement, all native solves and
certification. Pair that mission comparison with PDHCG/QOCO/hybrid comparisons
at the same original numerical gates. These are the concrete next steps in the
[SOTA execution plan](SOTA_EXECUTION_PLAN_2026-09-09.md#current-checkpoint-and-next-work).

## Exact result and display

[The sealed evidence package](../results/local/2026-09-09/mass-budgeted-frontier-v629/README.md)
contains the exact Result, both raw checker reports, immutable source and input
archives, all flight and wait readbacks, fleet composition lineage and a
package-only audit. Result SHA-256 is
`765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da`.

Full local result path:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\local\2026-09-09\mass-budgeted-frontier-v629\Result.txt
```

The dataset is installed and loaded in the existing web visualiser. Ships 8 and
19 display saved native solver nodes, certified endpoints and exact Result
events. Three coast intervals lack saved dense histories: the viewer leaves
those edges blank and hides the current-position marker while time is inside a
gap. It does not interpolate missing trajectories. The other 21 archived
histories are reused exactly. Source labels and missing-interval counts appear
in the selected-ship details.

Copy and paste into PowerShell to start the viewer if needed and load this fleet:

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath 'node' -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-frontier-v629&ship=8&epoch=69807&preset=oblique&z=1'
```

The numerical data, evidence packages and display samples have distinct
provenance. Displaying saved nodes adds no new spacecraft propagation or physics
qualification beyond the retained original checker reports.
