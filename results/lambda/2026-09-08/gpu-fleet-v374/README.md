# Wider GPU fleet search v374

The H100 run completed in **333.633 seconds**, returning three ships collecting
from 24 asteroids. Both the locally executed official checker and independent
verifier accept the fleet. Physical returned mass is **1,676.933607 kg**;
the frozen-bonus weighted score is **1,669.640529 kg**. This does not replace
the incumbent fleet's **12,805.194102 weighted kg**.

The request allowed up to 32 ships and a one-hour search budget. The run stopped
at ship 4 because its first five refinement candidates failed to certify. All
five share the early transfer 23045 → 10816, with virtual control remaining
1.447e-3. There were 198 candidate routes for that ship. These failed solves do
not prove that every remaining candidate or the transfer itself is infeasible.

The run used the validated v359 GPU QOCO library and v314 CUDA core with graph
execution. The subsequent GPU harvest-window pricing change was not used.
Beam width was 32, with five refinement candidates and eight retiming attempts.
It performed 148,555,144 transfer-branch requests, 18,250,121 collection-option
evaluations and 2,900 collection-DP passes. Counts include overlapping and
repeated screening work; they are not counts of certified full missions.

`retrieval.json` records the downloaded archive hash and verification of all
42 remote files. The original outputs, failures, source/binary hashes and runner
are preserved under `spacepdhcg-fleet-search-v374/`. The viewer import retains
1,525 exact archived samples and passes 9,030 Kepler context checks.

Open the displayed result:

<http://127.0.0.1:4173/?dataset=gtoc12-v374&epoch=69807&preset=oblique&z=1>

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-v374&epoch=69807&preset=oblique&z=1'
# If the server is stopped, run this and leave it running:
node scripts/serve.mjs --port=4173
```

Full solution path:

`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-fleet-v374\spacepdhcg-fleet-search-v374\output\fleet\Result.txt`
