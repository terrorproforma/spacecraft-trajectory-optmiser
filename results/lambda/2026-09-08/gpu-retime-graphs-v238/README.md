# CUDA retiming graphs: downloaded evidence

Code: `a9fb798e`, based on published `f1e8f2f7`.

The graph path and old published implementation select the same schedule and
objective. Cached price evaluation time falls by 3.3% on H100 and 4.0% on RTX
5090; full retiming changes by less than 1%. See the
[implementation and measurement report](../../../../docs/GPU_RETIMING_GRAPHS.md).

- `h100/validation`: build, 91-test suite, 28-case sanitizers and same-binary benchmark.
- `h100/mission-v239/release-comparison`: paired old-runtime versus graph measurements.
- `h100/mission-v239/boundary-profile.json`: native versus Python boundary timing.
- `h100/mission-v239/output/Result.txt`: full 526.489 kg replay, accepted by both checkers.
- `local`: RTX 5090 evidence, plus the three graph cases rerun after test formatting.
- `h100/failed-configure-v237`: preserved initial Git-snapshot configuration failure.
- `evidence-sha256.json`: byte hashes for the archived evidence files.

Full local repeat-solution path:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-retime-graphs-v238\h100\mission-v239\output\Result.txt
```

The existing viewer shows the earlier v235 replay of the same selected mission;
it retains that run's own trajectory samples, source commit and timings:

[Open the 526 kg mission](http://127.0.0.1:4173/?dataset=gtoc12-v235&epoch=69807&preset=oblique&z=1).

The fleet score remains 12,805.194 weighted kg. This experiment is a runtime
improvement and a repeated physics check, not a new fleet score.
