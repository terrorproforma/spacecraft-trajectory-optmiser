# Collection DP work count, v637

This is a source and integer-topology audit of the validated v844/v846 implementation.
It ran no solver, model, GPU or new timing experiment. The source is preserved here
because another native scheduling change is being developed in the shared worktree.

For k asteroids the old state array allocates k*2^k locations per time sample.
Only k*2^(k-1) have the current location outside the collected subset. The transition
kernel launches the full array for each of k cardinalities. At k=10 this means
102,400 scheduled coordinates versus 5,119 passing the structural filters (the camp
destination is also skipped at cardinality zero). CUDA block padding is excluded.

Compact layer scheduling could remove most immediately discarded coordinates;
compact state storage could halve these state arrays. This is not a 20x speedup
claim: the expensive transition arithmetic still has to run. Preserve the ordered
predecessor traversal and epsilon tie rule, then compare exact numerical results
and complete search time before enabling a new implementation.

The integer compact-state bijection was exhaustively checked through k=10. The
closed-form count is checked for all supported k=1..16. Saved Python profile entries
come from v846; native-call time includes synchronized GPU execution and readback.
It must not be described as time spent solely computing on the CPU.

work-counts.json is the original report; historical_audit.py is the exact executed
script with its original repository layout. analysed-source.cu is immutable input,
not a replacement production file. index.json binds all bytes and the prior archive.
No production code or result was changed by this audit.
