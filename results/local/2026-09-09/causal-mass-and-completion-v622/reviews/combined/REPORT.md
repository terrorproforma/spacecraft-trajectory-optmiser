# Independent combined-core integration review, v622

The combined CMake build and its bounded integration smoke pass the independent source and saved-output review. The reviewer made no native or GPU calls.

The 385-file source archive is exactly committed base `258e5c3aad1fd8e592527c759f1514925039cf2a` plus the previously reviewed 11 mass files and three compact-completion C++ files. Every overlay matches both its isolated component pin and the live file. Other working-tree changes are excluded. No compiled objects were reused; all 17 expected build and GPU-hidden CPU stages passed. The nine checked persistent solve/init resource footprints match the isolated mass build, and its compile command is identical apart from the frozen directory. Completion retains `--fmad=false` under the normal RDC build.

Build manifest: `9bff6c54af1edca6323502908d440305b4edc8447550572a5d93b50a421e3730`. Combined library: `6b32c2b5c4f87cd8990ee95816b00c0cb5dc863c7225cb244a17993ef344d8b1`.

The supervisor initially bound only compact-g's 12 owned-file archive while claiming to cover all host dependencies. This was corrected before launch: the final supervisor also binds the already-frozen 341-file full source index and archive, and verifies every host source byte. The final runner `54477378ae310a58ebd291a08a14ab15e3110d096ef8130b8ae17172ea977d3e` uses the shared nonblocking lock, pinned local GPU, inherited lock descriptor, fresh output, three owned processes and a 90-second deadline per process. The tests and numerical gates were unchanged.

The parent executed the integration smoke once. All three processes exited successfully and the final GPU process inventory was empty:

- Mass: seven solve calls, nine requested iteration caps and five actual updates. Its complete raw log is byte-identical to the isolated mass tiny run. This covers the three-step fixture, disabling back to L1 versus a fresh control, preserved qualified seed, cancellation, nonfinite input and overflow guards. Full vectors and actual diagonal steps are asserted inside the unchanged native test; the log emits summaries rather than those vectors.
- Legacy completion: two nonempty evaluations, 262 candidates, unchanged native assertions all pass. The log emits a summary, so this is not an independent replay of 262 exported vectors.
- Compact completion: all seven pytest cases pass with no skip. The frozen test flow makes 18 valid calls/2,358 candidate evaluations and six malformed calls without kernels. The six saved NPZs contain 786 original/compact candidate pairs. Independent forward replay, exact integer gates and cargo checks pass. The largest formula difference is **4.547473508864641e-13**. Every compared numerical field, including expanded model metadata, equals the prior standalone-g result; timing fields are excluded. Production capture checks remain test assertions rather than extra NPZ exports.

Final smoke report: `aae7e90de93a7a13887712dea85167c2e5f9d8c5ef93cf22f3642b9f47e73a44`. Independent findings: `ff83b223d1285bea3bd33efa9a8379d0eb135799cc052b41e45cc26e13fbaec8`.

This establishes correct combined linkage for the fixed tests. There was no new 100,000-iteration cold replay, timing experiment, fleet-score change or default promotion. The separate packing benchmark measured the standalone compact-g component library, not this combined library. Earlier cold mass solves remain unqualified.

Reproduction uses the three standard-library scripts in this directory. `review_build.py` performs a read-only `git archive` of the pinned base; all other operations read source archives, manifests, logs or NPZ files. The NPZ parser and forward-formula helper are compiled from their explicitly pinned source bytes without importing project/native code.
