# GPU return-sweep pricing — 8 September 2026

Return-sweep retiming now keeps both Lambert tables and measured-return masks on
CUDA. Previously `_return_override` downloaded a return table, constructed a
departure × TOF × sample distance array on CPU, found the nearest measured sample
and uploaded the resulting masks/inflations with the full DP snapshot.

The host now packs only attempted, lattice-aligned samples: departure index,
TOF index, measured delta-v and certification flag. One CUDA thread per grid
cell finds the nearest sample, divides its measured cost by that sample's
Lambert cost, and applies the existing strict reach/refusal policy. Ties prefer
the first input sample, including when the input order is unsorted. Far cells,
refused cells and nonfinite inflation measurements remain rejected; a sweep
with no applicable attempted samples restores the model, as the CPU reference
does. This does not relax thrust, dynamics, mass or objective checks.

`spacepdhcg_gtoc12_retime_set_sweep` retains sample storage and updates the existing
device table region. Refusals and replacement sweeps change only the mask, not
the transfer tables. The selected-path output now includes measured inflation
and its validity flag, so forward mass bookkeeping does not fetch the full
return mask. Both cache and sweep revisions guard selected-path reuse. The old
C entry points remain; updated Python needs a rebuilt library exposing the new
sweep functions. The explicit `read_sweep` diagnostic export is used in tests,
not in ordinary scheduling.

## Measurements and verification

Six alternating pairs compare fresh host-table and resident-table retiming
workspaces using the same rebuilt library and recorded certified return sample.
Medians exclude the first pair. This measures full retiming with a return sweep,
not just the new mask kernel and not full low-thrust refinement.

| GPU | Host-table path | Resident sweep path | Speedup |
|---|---:|---:|---:|
| H100 80 GB | 72.469 ms | 63.691 ms | 1.138× |
| RTX 5090 | 234.605 ms | 224.419 ms | 1.045× |

Both paths select exactly the same schedule and objective. Each run evaluates
412,116 Lambert branches (206,058 hop cells) in 24 bounded batches and six DP
calls. The resident path records one table build, one compact sweep update and
zero host table uploads. This comparison includes the resident-table improvement
for a workload that previously could not use it; it is not an additional 1.138×
on top of v231's unswept measurement.

Four new GPU tests compare every mask and inflation cell with the CPU reference,
then compare selected paths and forward masses. They cover unsorted ties,
off-grid/unattempted samples, NaNs, all-refused and empty sweeps, refusal updates,
stale path rejection, and atomic rejection of malformed native sample indices.
The complete selected suite passes **88 tests locally (28.61 s)** and **88 on
H100 (40.89 s)**. H100 memcheck runs 25 cases in 32.29 s and initcheck runs the
same 25 in 13.36 s; both report zero errors.

## Independently verified mission

The input sweep contains the actual final-leg measurement from the previously
certified v232 mission: asteroid 57530 → Earth, MJD 69113 → 69518, departure mass
1192.536333 kg and measured delta-v 4.518288727 km/s. The existing sweep policy
allows neighbouring lattice cells to inherit that inflation. It changes the
schedule's predicted payload from 524.024641 to **526.488706 kg**. This gain comes
from using the recorded return measurement; the CPU reference finds it too.
The GPU port supplies faster equivalent computation, not a different objective.

H100 v235 re-flies the resulting 13 legs in **9.434 s**. Both checkers pass:
official **526.489 kg**, independent **526.4887063657 kg**, six mined asteroids,
no violations. Maximum independent replay errors are 0.451159 km position,
7.72947e-8 km/s velocity and 8.55e-11 kg mass, under the unchanged verifier rules.
The complete cold retiming step is 0.398 s; it must not be confused with the
warm-pair benchmark median. These are single refinement observations, not a
matched refinement speedup study.

The 23-ship fleet remains **12,805.194 weighted kg**. This separate 526 kg ship is
still below the incumbent's lightest ship, so it has not replaced a fleet member.
The new mission is selectable in the existing visualiser, with its own provenance,
508 exact replay samples and physical 1× vertical scale in the supplied link.
The importer validates 2,288 context-orbit points; all five fleet dataset checks
and 38 viewer tests pass.

## Evidence and reproduction

[gpu-sweeps-v234](../results/lambda/2026-09-08/gpu-sweeps-v234/) contains the
benchmark inputs, measured return, logs, native/source hashes, independent checks,
full solution, and viewer export. Configure `PYTHONPATH=src`, the pinned
`SPACEPDHCG_GTOC12_DATA`, and the rebuilt `SPACEPDHCG_GTOC12_CUDA_LIBRARY`, then run:

```sh
python results/lambda/2026-09-08/gpu-sweeps-v234/benchmark_sweeps_v233.py \
  build/performance/sweep-replay \
  results/lambda/2026-09-08/gpu-sweeps-v234/input-refinements.json \
  results/lambda/2026-09-08/gpu-sweeps-v234/input-return.json
```

The script serializes GPU use with the existing per-device lock. Set
`SPACEPDHCG_GTOC12_GPU_TESTS=1` to run the GPU tests. H100 build commands and the
complete certification runner are archived with the results. Viewer loading
instructions and the full local solution path are in the artifact README.

Price/mass iteration, forward bookkeeping, visit-order orchestration and fleet
selection still run on CPU. Compact sweep input filtering also remains on CPU;
the numerical grid, nearest-neighbour selection and masks run on CUDA. Explicit
custom host tables retain their compatible path. Full GPU orchestration, batching
and improved fleet search remain active work.
