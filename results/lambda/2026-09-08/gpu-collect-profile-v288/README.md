# Remaining collection-table cost after CUDA DP

Direct nested wall timers in `instrument.py` instrument the same H100 source and
binary as the published CUDA collection DP. This instrumented observation is
not included in the matched speed benchmark. Its 63.938 s CLI campaign passes
both checkers at 548.255 kg. Exact outputs and runner commands are retained.

`timing.json` records:

- 248 complete collection plans: 2.881 s inclusive.
- 378 native DP solves plus Python result formatting: 0.173 s.
- 248 workspace creations: 2.587 s inclusive, 0.148 s excluding instrumented children.
- All 7,124 pair-table calls: 4.701 s; 1,288 return-table calls: 0.365 s.

Table calls also occur outside the timed collection-plan spans. Inclusive spans
overlap and must not be added. Exclusive spans subtract only instrumented child
calls, not all children; all durations include any native execution and GPU
synchronization. No profiler aggregate CPU attribution is inferred.

The next measured collection target is the pair/return table path, currently
using host ephemerides before CUDA Lambert screening. Existing orbital-element
GPU screening provides a concrete route to move this preparation onto CUDA.
The remaining Python mass-pass control matters for full GPU residency but is
small in this measured campaign. Broader search/refinement still requires
separate profiling; this is not a complete attribution of total runtime.

Downloaded archive SHA-256:
`69c0d45d74de58af9c413b50f99c656113ad0cf02972235f7361b58a0ca67fa9`.
`sha256.json` pins every extracted evidence file except itself.
