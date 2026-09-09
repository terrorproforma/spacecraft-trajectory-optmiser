# New departure search: 3,502 plans, three profitable prescriptions

Eight H100 searches explored ships 2, 5, 12 and 18, each using three different
Earth departure targets and both unit and official bonus weights. They produced
3,502 saved plans representing 2,076 distinct physical requests. Three ship-18
requests meet the current fleet's conflict, raw-budget and positive-weighted-gain
screens. These are forecasts, not certified deliveries.

The seven searches with retained timers produce 3,019 plans in 43.812279 seconds
(68.91 candidates/second) and record 104,076,562 Lambert branches. The first search
saved 483 plans before reporting rejected an infinite disabled-limit setting.
The continuation preserves those plans without rerunning them; missing telemetry
is explicitly null. Both original and corrected worker sources are archived.

`evidence.tar.gz` contains 50 files: all eight candidate pools, exact inputs,
screening results, settings, telemetry, initial failure, terminal process checks
and the pinned bonus table. Archive SHA-256:
`fc61c5e09aa4c67885fc7bb71b4f2cdd4607b69989689a6342152602fdaf1722`.
`manifest.json` records every member's bytes and SHA-256.

The native core is `a2ec27ce1b20ec472b0044a5c554751cb704b496f479aff487a852b7cda0527d`;
its source manifest is `2cbf1221dc2eace0b0d1b10623a629be3daaa985bd1839769a55d306b6500014`.
The exact source and binary are already archived in
[the shared-return runtime package](../gpu-shared-return-options-v846/README.md).
The incumbent Result is `1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da`.

Run the standard-library audit from this directory:

```text
python audit_saved.py
```

It checks all archive bytes, preserves the aborted run's saved output, recomputes
fleet and candidate cargo with 65-digit Decimal arithmetic, and deduplicates
physical requests independently of proxy costs. It writes `saved-audit.json`;
it executes no archived code, GPU call or trajectory verifier. Historical launch
scripts in `reproduction/` retain their original workspace/host paths and are not
portable launch commands. Runtime/data dependencies must be restored explicitly
to reproduce GPU execution.

The [native follow-ups](../gpu-family-refinement-v855/README.md) do not produce a
complete qualified route. See the [combined report](../../../../docs/GPU_NEW_DEPARTURE_CAMPAIGN.md).
