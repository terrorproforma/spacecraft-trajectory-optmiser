# One-interval defect localization, v633

The single CUDA diagnostic reproduced the failed v632 return's maximum defect exactly. Only the first interval's vy component exceeds the original gate: 82.85482786091636 m/s. There was no optimizer, new certificate or score gain. See [RESULTS.md](RESULTS.md) for the measured scope and [NEXT_MISSION.md](NEXT_MISSION.md) for a bounded, unexecuted arrival-window proposal.

`evidence.zip` preserves all 25 sealed preparation files plus ready.json, the launch marker, all eight output files, the saved-only audit and publication scripts. Earlier lint failures remain included; the final 12 CPU tests and Ruff passed. The preparation README is historical.

Native source, build/runtime proof and original v632 inputs are linked to the adjacent [gpu-mass-merit-v632 package](../gpu-mass-merit-v632/README.md), with exact index and member hashes. No source archive, binary, credential or full prefix data tree is duplicated here. The dependency package must remain adjacent for portable verification.

Run `python audit_package.py --roundtrip` for saved-byte/array-subtraction verification only. It uses the standard library, checks all indexed and linked bytes, and does not execute archived workers, load CUDA, evaluate orbital dynamics or run an optimizer. An optional `--index-sha256 HASH` binds the independently supplied index.

[Portability correction](PORTABILITY.md): only one reporting-only expm1 estimate receives a two-ULP allowance; all raw and physics checks remain exact. The first failed Windows audit and original package are retained.
