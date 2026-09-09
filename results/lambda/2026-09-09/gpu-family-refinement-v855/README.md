# Native refinement: 16-flight prefix, unresolved Earth return

The new ship-18 requests produced **38 unique native solves, 369 outer updates,
155 accepted updates, 30 unique qualified flight trajectories and 86 flight
certificate calls including repeated prefix checks**. No complete route qualified;
there were no full-fleet checker calls and no score gain.

The original three requests fail the same third leg. Two larger virtual-control
penalties do not resolve it. Allowing that leg 15 or 30 additional days, shortening
a later wait and reducing cargo to account for lost mining time, gives 16
certified flights. Both resulting Earth returns fail. Three further return dates
with fixed cargo and the exact +30-day schedule prefix also fail. All original
physical and numerical tolerances remain unchanged.

Five campaign directories preserve every native solve's boundary, raw states,
thrust, solver histories, post-clamp flight arrays, certificates, fixed requests
and reports. The timing-preparation failure ran zero native calls; its corrected
successor computes cargo directly from the production mining limit. Exact matching
boundaries/settings reuse saved solver outputs, with independent flight checks
retained. Neither cache hits nor repeated certificates count as new solved flights.

`evidence.tar.gz` contains 1,159 files, including the QOCO runtime and its source,
build records, dynamic-library identities and final process inventory. SHA-256:
`663098aca788b0e05e6eb7af2e7e953869deabd65227165d257a72cd10db2037`.
`manifest.json` records every member's bytes and SHA-256. Large CUDA runtime
dependencies are identified by hashes rather than copied. The trajectory core
and frozen Python source are the unchanged
[v846 runtime](../gpu-shared-return-options-v846/README.md), core SHA
`a2ec27ce1b20ec472b0044a5c554751cb704b496f479aff487a852b7cda0527d`.
The [v850 search archive](../gpu-family-departures-v850/README.md) supplies all
original plans, fleet and bonus inputs.

Run from this directory:

```text
python audit_saved.py
```

The standard-library audit verifies every byte, checks changed epochs and cargo,
reconstructs physical-request hashes, confirms unchanged settings except the two
explicit penalty probes, compares reused array payloads, and reconstructs work
counts. It writes `saved-audit.json`. It does not propagate trajectories or run
archived programs. Historical launch scripts in `reproduction/` need their exact
source/runtime/data dependencies and original path layout to execute again.

The saved v854 rank-1 prefix is the useful input for a distinct future return or
collection-geometry intervention. All owned H100 workers exited, and the final
GPU inventory was idle. The verified score remains **13,526.961241 weighted kg**;
the existing visualiser retains the v633 fleet.

See the [combined measurements and decisions](../../../../docs/GPU_NEW_DEPARTURE_CAMPAIGN.md).
