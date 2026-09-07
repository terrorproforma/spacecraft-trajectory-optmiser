# Certified 526 kg mission and GPU return sweeps

H100 v235 returns **526.488706366 kg** from six asteroids, up **2.464065709 kg**
from the previous 524 kg mission. All 13 legs pass refinement and both official
and independent physics verification. The incumbent 23-ship fleet remains
**12,805.194 weighted kg**. See [implementation and measurements](../../../../docs/GPU_RETURN_SWEEPS.md).

The solution is downloaded locally at:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-sweeps-v234\h100\mission-v235\output\Result.txt
```

Open [the new mission in the existing visualiser](http://127.0.0.1:4173/?dataset=gtoc12-v235&epoch=69807&preset=oblique&z=1).
Dataset is **GPU return sweep v235 (526 kg certified)**. Click Ship 1 to inspect
the sequence, or Play mission to animate the 508 exact archived propagation
samples. The link uses physical 1× vertical scale. The compute panel identifies
`gpu_sweep_v235`, source commit `1197361f858a`, 0.398 s cold retiming and 9.434 s
GPU refinement.

Copy/paste to open it:

```powershell
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-v235&epoch=69807&preset=oblique&z=1'
```

If the local server is stopped, restart it in a separate terminal first:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4173
```

`h100/mission-v235/report.json` contains the two checker results and solver
telemetry. `h100/validation` contains the build, test, sanitizer and alternating
benchmark logs. `input-return.json` retains the source of the measured sweep
sample; `viewer-export` contains the locally regenerated verifier replay.
`evidence-sha256.json` covers the evidence files and `viewer-qa.json` records the
browser inspection and generated dataset hash.
