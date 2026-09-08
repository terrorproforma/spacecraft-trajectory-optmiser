# GPU fleet numerical setup evidence

See [implementation and measured performance](../../../../docs/GPU_FLEET_TOPOLOGY.md)
and [exact summary](summary.json). The verified mission remains at
12,842.970672270894 weighted kg; these are packing tests using existing route
records, with no new trajectory or mission physics qualification.

* `local.tar.gz` and `h100.tar.gz` contain v740 initial and v742 final frozen
  source/libraries, build/test logs, benchmark scripts, input pool and hardware
  identity. Embedded `FILES.json` and external receipts pin contents and hashes.
* v741 is a **failed** exact comparison: the initial naive GPU sum differs by
  approximately 1.82e-12 kg. The corrected v742 uses compensated sums and faster
  provider/conflict construction. Failed evidence is retained without a speedup
  claim for that prototype.
* v743 is the successful seven-repetition, four-variant comparison. All selected
  IDs, objectives, bounds and work counts match exactly. Published medians exclude
  the first repetition; workspace creation samples are reported separately.
* v744 contains 115 passing tests per GPU, ten catalogue-dependent skips, and
  successful full-pool memory/race/synchronisation checks with zero leaked bytes.
* `reproduce/replay.py` verifies and extracts the final variant from its archive
  and runs the retained API in a compatible CUDA/Python runtime. It does not
  depend on the original machine's project directories or rerun mission physics.
* `replay-smoke.json` records a successful local replay, including the usable
  column count. The final solution remains in `gpu-fleet-exchange-v733`.

`sha256.json` pins every published evidence file except itself. Native libraries
and the first prototype's source remain inside the verified raw archives.
