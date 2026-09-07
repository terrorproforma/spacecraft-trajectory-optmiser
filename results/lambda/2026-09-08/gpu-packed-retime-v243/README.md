# Consolidated retiming output: evidence

Code `41293ed5`, based on published `17c5b77d`.

Cached price evaluation time falls by 4.4% on H100 and 15.0% on RTX 5090; complete
retiming changes by less than 1%. Both implementations select the same schedules.
See the [implementation and trace report](../../../../docs/GPU_PACKED_RETIMING_OUTPUT.md).

- `h100/validation`: selected pageable-buffer build, tests, sanitizers and benchmarks.
- `h100/pinned-v241` and `local-pinned`: tested but unselected pinned-buffer trial.
- `h100/mission-v244`: 526.489 kg verified replay and H100 Nsight trace.
- `local`: selected RTX 5090 measurements, tests and API-only Nsight trace.
- `evidence-sha256.json`: exact artifact byte hashes.

The H100 kernel trace identifies leg selection as 83.7% of kernel time. Host
copy API durations include GPU waiting and must not be interpreted as PCIe time.

Full local repeat-solution path:

```text
C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-packed-retime-v243\h100\mission-v244\output\Result.txt
```

The [existing visualiser](http://127.0.0.1:4173/?dataset=gtoc12-v235&epoch=69807&preset=oblique&z=1)
retains the earlier v235 samples and timing for the same selected mission. The
fleet score remains 12,805.194 weighted kg; this is a runtime change and repeated
physics verification, not a new score improvement.
