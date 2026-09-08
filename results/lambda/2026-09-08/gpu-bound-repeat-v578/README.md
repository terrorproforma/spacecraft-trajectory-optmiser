# Repeated H100 bound-classification comparison

This supplements `../gpu-bound-types-v577` with a second full 225-leg pass per
mode on the same frozen binary. The initial pair ran host then GPU classification;
the repeat ran GPU then host, after the intervening campaign/default validation.
Both modes use device numerical initialization, zero Ruiz and workspace reuse.
Original solver budgets and physics tolerances are unchanged.

All four passes retain the same 205 independently certified trajectories. The
maximum final-mass difference relative to the first baseline is 1.042e-7 kg.
Host solver times are 157.0263 and 154.5054 seconds; GPU times are 164.3139 and
153.9951 seconds. The two-run medians are 155.7659 and 159.1545 seconds: GPU
classification takes 2.18% more time in this sample. Timing ranges overlap.
The repeated pair alone slightly favours GPU classification, illustrating the
variation. This does not establish either an overall speedup or its cause.

`lambda-raw.tar.gz` contains the second pair, source snapshots, exact commands,
per-leg trajectories and certificates. First-pair raw data and the actual CUDA
binaries are retained in `../gpu-bound-types-v577/lambda-raw.tar.gz`. Runtime and
source identities were checked across both pairs. `summary.json` summarizes all
four passes. Member and published-file SHA-256 manifests accompany this archive.
