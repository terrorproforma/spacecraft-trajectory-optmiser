# Native route regeneration and constrained dual correction — v625

The complete archived ship-23 route now regenerates through native GPU refinement
with certified mass handoffs. All 19 flown legs and both waits pass; both original
full-fleet checkers accept the emitted fleet with unchanged cargo and score.
The other 22 ships remain byte-identical. A fresh H100 build and the first-leg
portability control also pass.

| Measurement | Result |
| --- | ---: |
| Local native solves / SCvx iterations | 19 / 19 |
| QOCO subproblems / inner iterations | 19 / 468 |
| Local route refinement, certification and callbacks | 3.875972 s |
| Local worker including full-fleet checks | 25.192757 s |
| H100 first-leg native call | 0.287122 s |
| Fleet weighted / raw score | 12,843.555696 / 14,044.353183 kg |

This is regeneration of a known feasible route, with no search or score gain.
GPU QOCO is the mission backend. Python ephemeris and thrust postprocessing remain;
this result does not establish a fully GPU-controlled pipeline or a PDHCG speedup.
[Implementation, timing scopes and physical checks](../../../../docs/GPU_FIXED_ROUTE_INITIALIZATION.md).

Separately, one 11.618 ms CPU dual-correction call makes a saved PDHCG iterate pass
all original numerical gates without changing its primal variables. Native PDHCG
still has its original iteration-limit status, and the corrected gap has only
9.36e-17 of margin. This is a hybrid diagnostic requiring a GPU implementation,
more numerical margin and wider validation before a performance claim.
[Formulation, results and limitations](../../../../docs/PDHCG_CONSTRAINED_DUAL_POLISH.md).

The final production API passes 73 CPU tests, including a post-run reporting fix
that accumulates scheduler counts across legs. The measured frozen source and
its original last-leg scheduler summary remain in the archive; independent call
logs establish the actual 19 solves. Eighteen separate mocked route-harness tests
pass. H100 passes three CUDA test binaries and three sanitizer runs. Saved route,
H100 and dual-vector audits are included.

## Display and paths

[Open the local unchanged-score control, focused on ship 23](http://127.0.0.1:4173/?dataset=gtoc12-seeded-route-v625&ship=23&epoch=69807&preset=oblique&z=1).
The existing viewer defaults and previous datasets are retained. Its HTTP assets
and schemas were checked; browser rendering was unavailable to the automation,
and opening the tab was queued in Codex.

Verified Result, SHA-256 `6fa61648cbc958aec0f93b941d0bafc650fa6f0bcb2497b60743481a88e53ba8`:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\build\performance\seeded-route-v625\output\fleet\Result.txt
```

Viewer data:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser\data\gtoc12-seeded-route-v625\fleet.json
```

Copy into PowerShell to load it, starting a hidden HTTP helper only if needed:

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
$probe = 'http://127.0.0.1:4173/data/gtoc12-seeded-route-v625/manifest.json'
try {
    $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 $probe
} catch {
    Start-Process node -ArgumentList @('scripts/serve.mjs','--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-seeded-route-v625&ship=23&epoch=69807&preset=oblique&z=1'
```

The separate CPU display replay samples only ship 23 in 0.857576 s. It reuses the
22 unchanged ship records after proving their Result sections identical. Display
work is excluded from the route and worker times above. The viewer contains
23 ships, 195 asteroids, 436 events and 11,680 sampled trajectory points.

## Archive and independent readback

`evidence.zip` contains 472 files: frozen inputs and executed source, raw native
and certificate arrays, both full-fleet reports and Result copies, H100 source/
build/test evidence, dual-polish vectors, display evidence and source/tests/docs.
It excludes executables, credentials and caches. The original official checker
binary is represented by its recorded hash and invocation/result evidence.

- Archive SHA-256: `3d25c4929d8fdfca59fb26698b3023cacc1b332afa4871c97d8fbb6a4dcea684`.
- Index SHA-256: `73b46a3d88d3e3d33d42618792018dce58d077ca01a67bfad6a80fbac73eef6a`.
- `audit_package.py` SHA-256: `25f39d19819ab7c95d60317377d3db41ab0b7f0610d148c79be9507d57adf561`.

The independent package audit verifies both archives, extracts the prior v624
dependency before v625 overlays into a fresh temporary tree, and replays the two
pinned saved-data auditors. All physical findings match; the only permitted
output-map difference is the explicitly omitted official checker binary. It
performs no GPU work, propagation, optimization or original checker execution.
`verification.json` records that completed readback.

From the repository root, use a fresh output directory:

```text
python results/local/2026-09-09/native-route-and-dual-polish-v625/audit_package.py --package results/local/2026-09-09/native-route-and-dual-polish-v625 --previous results/local/2026-09-09/native-trajectory-seed-v624 --expected-index-sha256 73b46a3d88d3e3d33d42618792018dce58d077ca01a67bfad6a80fbac73eef6a --output-dir build/performance/v625-publication-readback-new
```
