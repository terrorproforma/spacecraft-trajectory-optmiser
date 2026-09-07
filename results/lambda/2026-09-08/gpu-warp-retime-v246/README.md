# Parallel flight-time selection: downloaded evidence

Code `9ec9afbd`, based on published `89922bdd`.

Cached pricing improves 2.19× on H100 and 2.86× on RTX 5090. At five-day timing
resolution the gains are 4.57× and 4.40×. Full retiming improves 1.040× and 1.023×.
All matched schedules and objectives agree. See the
[implementation and trace report](../../../../docs/GPU_WARP_RETIMING.md).

- `h100/validation`: build, 99-test suite and three 36-case sanitizer runs.
- `h100/mission-v247`: grid-scaling benchmark, CUDA trace and verified mission replay.
- `local`: RTX 5090 benchmarks, tests and new edge-case results.
- `evidence-sha256.json`: exact artifact byte hashes.

Full local repeat-solution path:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-warp-retime-v246\h100\mission-v247\output\Result.txt
```

The [existing visualiser](http://127.0.0.1:4173/?dataset=gtoc12-v235&epoch=69807&preset=oblique&z=1)
retains the earlier v235 samples and timing for the same selected 526.489 kg
mission. The fleet score remains 12,805.194 weighted kg.
