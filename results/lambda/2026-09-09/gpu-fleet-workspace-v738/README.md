# Retained CUDA fleet workspace evidence

See [measurements, API and limitations](../../../../docs/GPU_FLEET_WORKSPACE.md)
and [exact summary](summary.json). The verified mission score remains
12,842.970672270894 weighted kg. These tests reuse the same route pool and do not
produce a new trajectory or rerun mission physics.

* `local.tar.gz` / `h100.tar.gz`: v735 frozen source and measured native libraries,
  v736 interleaved performance comparisons, v737 tests/full-pool sanitizers,
  hardware identity and pool inputs. Every archive member is checked against its
  embedded `FILES.json`; receipts pin archive hashes and byte lengths.
* `local/` / `h100/`: readable copies of raw build, benchmark and safety reports.
* `integration/`: v739 rebuild of the later combined revision on both GPUs,
  including the separate route-cost fix. Each side includes full source, logs and
  library hashes, but does not duplicate the binary library. Both pass 106 tests
  with ten catalogue-dependent skips.
* `reproduce/replay.py`: verifies/extracts the appropriate v735 archive and runs
  the retained selector without depending on the original machine paths. Use
  an installed Python/CUDA environment compatible with the selected GPU.
* `replay-smoke.json`: successful local portable replay, two repeated searches.

The raw local Nsight Systems trace is archived, but contains no CUDA kernel data.
Its successful command exit is not evidence of kernel profiling. Benchmark
timings are measured wall times; setup and repeated solves are separate.

`sha256.json` pins all published files except itself. The unchanged, independently
verified mission and visualiser remain in `gpu-fleet-exchange-v733`.
