# GPU-deferred validation: `integration/single-gpu-v2-candidate`

Machine-readable twin: [`benchmarks/gpu_deferred_validation_v2.json`](../benchmarks/gpu_deferred_validation_v2.json)
(checked by `tests/test_gpu_deferred_manifest.py`).

The candidate consolidates `feat/planner-cli` (c74fdb7), `feat/literature-targets` (f6e8140) and
`feat/gtoc12-asteroid-mining` at fa91b43 on top of `integration/single-gpu-v1` (63271d5). Everything
that needs the RTX 5090 was **deferred**, not skipped silently: the G4 H5/H6 claim-core campaign
(`run_g4_campaign.py run --claim-core`, source commit 9a4cbea, capability `e546583b…`) owns the
device from `/home/angus/worktrees/spacepdhcg-single-gpu-integration`. The CPU matrix ran with
`CUDA_VISIBLE_DEVICES=''`; both CUDA trees were configured and built for `sm_120` but never executed.

## Gate before any item

1. `pgrep -f 'run_g4_campaign.py|--g4-server|--g4-session'` prints nothing.
2. `/home/angus/.spacepdhcg-gpu.lock` is not held by another task (this integration never created it).
3. `spacepdhcg literature gpu-preflight` exits 0 with `"reason": "device free"`.
4. Common exports (see `environment.common_exports` in the JSON): `V2`, `PATH` (repo `.venv/bin`
   CMake/CTest, CUDA 12.8, Linux node), `PYTHONPATH=$V2/src`, `CUDA_VISIBLE_DEVICES=0`,
   `SPACEPDHCG_NATIVE_LIBRARY=$V2/build-v2-relwithdebinfo/libspacepdhcg.so`,
   `SPACEPDHCG_QOCO_LIBRARY=<cuda-algebra QOCO 09f0495 + abstol patch>`, `LD_LIBRARY_PATH` with the
   cuDSS shim, deterministic thread/hash seeds.

Run the items serially; the device is shared with other agents and Windows-side jobs.

## planner-gpu-pytest

```bash
cd $V2 && SPACEPDHCG_PLANNER_GPU_TESTS=1 \
  SPACEPDHCG_PLAN_EXECUTABLE=$V2/build-v2-cuda-release/cuda-tools/spacepdhcg_plan \
  $V2/.venv-v2/bin/python -m pytest -q -p no:cacheprovider tests/test_planner_gpu.py
```

Expected: `9 passed` (four parametrised certified native plans matching the CPU reference, warm-start
reuse, `pdhcg` backend selection recorded, honest infeasible-target report, time-limit report,
`cpu_reference` documents rejected); no skips once the GPU QOCO library is set. Currently 9 SKIPPED
(gated) in the CPU run.

## planner-memcheck

```bash
cd $V2 && mkdir -p build-v2-gpu-deferred
compute-sanitizer --tool memcheck --leak-check full build-v2-cuda-release/cuda-tools/spacepdhcg_plan \
  examples/planner/hcw_rendezvous.json --output build-v2-gpu-deferred/memcheck-hcw.json --quiet \
  > build-v2-gpu-deferred/memcheck-hcw.log 2>&1; echo exit=$?
compute-sanitizer --tool memcheck --leak-check full build-v2-cuda-release/cuda-tools/spacepdhcg_plan \
  examples/planner/powered_descent_3dof.json --output build-v2-gpu-deferred/memcheck-pd3.json --quiet \
  > build-v2-gpu-deferred/memcheck-pd3.log 2>&1; echo exit=$?
for tool in initcheck synccheck racecheck; do
  compute-sanitizer --tool $tool build-v2-cuda-release/cuda-tools/spacepdhcg_plan \
    examples/planner/hcw_rendezvous.json --output build-v2-gpu-deferred/$tool-hcw.json --quiet \
    > build-v2-gpu-deferred/$tool-hcw.log 2>&1; echo $tool exit=$?
done
```

Expected: every `exit=0`; `ERROR SUMMARY: 0 errors` (memcheck/initcheck/synccheck) and
`RACECHECK SUMMARY: 0 hazards`; both result documents `status.code == "certified"`,
`certificate.certified == true` (HCW N=20 `pdhcg`; 3-DoF hover `pure_qoco`, objective ≈ 0.4927 as
measured on 2026-09-03 before the campaign took the device).

## planner-cuda-ctest-subset

```bash
cd $V2 && for tree in build-v2-cuda-release build-v2-cuda-debug; do
  ctest --test-dir $tree --output-on-failure -R \
   'spacepdhcg_plan_capabilities|device_variational_test|persistent_cw_test|pointer_contract_test|allocation_lifecycle_test|stream_lifetime_test|cone_inventory_test|dlpack_contract_test|recovery_test|persistent_soc_test'
done
```

Expected: `100% tests passed, 0 tests failed out of 10` per tree; `spacepdhcg_plan_capabilities`
prints the capability JSON with a real device.

## planner-gpu-examples

```bash
cd $V2 && for f in hcw_rendezvous powered_descent_3dof powered_descent_6dof low_thrust; do
  $V2/.venv-v2/bin/spacepdhcg plan examples/planner/$f.json \
    --executable build-v2-cuda-release/cuda-tools/spacepdhcg_plan \
    --output build-v2-gpu-deferred/plan-$f --export-viewer build-v2-gpu-deferred/plan-$f/viewer
  echo $f exit=$?
  (cd build-v2-gpu-deferred/plan-$f/viewer && node scripts/check.mjs); echo $f viewer exit=$?
done
```

Expected: `exit=0` (certified) for all four; each export bundle is a self-contained viewer copy and
passes `node scripts/check.mjs` as a `planner-export` dataset while the default archive keeps its assertions (re-verified in this
integration: `Validated 5 archive trajectories`, data SHA `b160734e…`).

## literature-device-time-dilated-ctest

```bash
cd $V2 && ctest --test-dir build-v2-cuda-release --output-on-failure -R device_time_dilated_test
cd $V2 && ctest --test-dir build-v2-cuda-debug   --output-on-failure -R device_time_dilated_test
```

Expected: `100% tests passed, 0 tests failed out of 1` per tree. Internal gates: pd3_fft device
A/B/z and sigma-column parity `< 5.0e-11`, affine reconstruction `< 1.0e-8`; pd6_fft parity
`< 2.0e-9`, reconstruction `< 1.0e-8`.

## literature-device-time-dilated-sanitizers

```bash
cd $V2 && for tool in memcheck racecheck initcheck synccheck; do
  compute-sanitizer --tool $tool build-v2-cuda-release/cuda-tests/device_time_dilated_test \
    > build-v2-gpu-deferred/time-dilated-$tool.log 2>&1; echo $tool exit=$?
done
```

Expected: `exit=0` ×4; `ERROR SUMMARY: 0 errors` / `RACECHECK SUMMARY: 0 hazards`.

## literature-gpu-run

```bash
cd $V2 && $V2/.venv-v2/bin/spacepdhcg literature gpu-preflight; echo exit=$?
cd $V2 && $V2/.venv-v2/bin/spacepdhcg literature gpu-run \
  acikmese-ploen-2007-pd3 blackmore-2010-pd3-case1 chari-2024-pd6-monte-carlo; echo exit=$?
cd $V2 && git status --porcelain -- results/literature docs/REFERENCE_REPRODUCTION_REPORT.md \
  benchmarks/literature/reference_reproduction.json
```

Expected:

- preflight `exit=0`, `"ok": true`, `"reason": "device free"` (exit 3 = still owned: stop);
- `gpu-run` `exit=2` is the expected final code (chari stays `gap` by design); the two pd3 targets
  stay `reproduced`;
- `acikmese-ploen-2007-pd3`: `measured.scvx_qoco_gpu_status == "converged"`, GPU fuel within the
  1.5 kg tolerance of 399.5 kg (CPU SCvx 399.361 kg);
- `blackmore-2010-pd3-case1`: `measured.scvx_qoco_gpu_status == "converged"`, within 1.5 kg of
  399.4 kg (CPU SCvx 398.84 kg);
- `chari-2024-pd6-monte-carlo`: `details.gpu_persistent_batch.pure_qoco_native_pd6_fft.status ==
  "measured"` with batches 1/16/64; `persistent_device_scvx` stays `blocked`;
- the report, its JSON twin and `results/literature/*.json` are rewritten; commit the tracked
  report/JSON after review.

## gtoc12-gpu-lambert-parity

```bash
cd $V2 && ctest --test-dir build-v2-cuda-release --output-on-failure -R 'orbitweaver_gpu_test|orbitweaver_g3_adapter_test'
cd $V2 && SPACEPDHCG_GTOC12_DATA=$V2/benchmarks/gtoc12/data \
  $V2/.venv-v2/bin/python -m pytest -q -p no:cacheprovider tests/test_gtoc12_pipeline.py -k lambert
```

Expected: `orbitweaver_gpu_test` passes (device Lambert family batch matches the CPU
`enumerate_lambert_families` universal parameter and departure velocity); the CPU parity test
(NumPy vs native `spacepdhcg_lambert_*` on the 8192-sample set) passes as it already did here.
Follow-up: a catalogue-sample harness feeding the same triples to
`spacepdhcg_orbitweaver_lambert_evaluate_async` with `|ΔV_gpu − ΔV_cpu| ≤ 1e-9 km/s` is not yet
scripted; GTOC12 deliberately screens on the CPU while the campaign runs.

## current-head-g2-g3-reseal

```bash
cd $V2 && test -z "$(git status --porcelain=v1)" && git rev-parse HEAD
cd $V2 && bash scripts/gpu/run_g2_evidence.sh   # out=/py=/tool= adapted to this worktree, as at b6afb49
cd $V2 && bash scripts/gpu/run_g3_evidence.sh   # pass --force-export=true to nsys stats
cd $V2 && ctest --test-dir build-v2-cuda-release --output-on-failure && ctest --test-dir build-v2-cuda-debug --output-on-failure
```

Expected: full CUDA CTest `100% tests passed out of 68` in Release and Debug (62 sealed at b6afb49
plus `planner_c_api_smoke`, `planner_problem_smoke`, `powered_descent_free_time_transcription_smoke`,
`time_dilated_flow_smoke`, `device_time_dilated_test`, `spacepdhcg_plan_capabilities`); G2/G3
`status=PASS`.

**New topologies do not alter sealed ones.** Evidence, base 63271d5 vs candidate:

| Frozen artefact | Blob id (identical on both) |
| --- | --- |
| `cpp/include/spacepdhcg/transcription/hcw_rendezvous.hpp` | `1a0617974f882df7d3de3a4619e97cc29d594b78` |
| `cpp/include/spacepdhcg/transcription/powered_descent_3dof.hpp` | `c1771cc617741634cf03269684b53428703f5aaf` |
| `cpp/include/spacepdhcg/transcription/powered_descent_6dof.hpp` | `19817a9ac145bbc5cb4381fdca58f1d075747ee9` |
| `cpp/include/spacepdhcg/transcription/low_thrust.hpp` | `c77fb96870cc2561d973ab04fcdf5d954ddb7a2d` |
| `cpp/cuda/src/persistent_pdhcg.cu` | `8ce085181d7e1dc807c46ce44a1dcef11b936001` |
| `cpp/cuda/tests/device_scvx_integration_test.cu` | `af788b0bf60e7c92f124147d0324908e71b8b616` (release merge; was `32ba2649…` at d7ca28f) |
| `cpp/cuda/tests/recovery_test.cu` | `124afd35df935041d62d13b17697ef763555b6bd` (release merge; was `317ccdff…` at d7ca28f) |
| `benchmarks/g4_policy.json` (sha256 `9ab3b444…`) | `07755ed79d77700ab2641259b414ddffcadc6cef` |
| `benchmarks/g4_applicability.json` (sha256 `1c4e0d51…`) | `a5b4b0d751e11cd76c348800b39c4fffb2f83a73` |
| `benchmarks/g4_h5_h6_claim_core.json` (sha256 `40dc2174…`) | `b16685b57f6aa2cd63d81e9ce2604daece6b394e` |
| `benchmarks/paper2_instances.json` | `5825c94a1fe3f393f67d05d4938eeda2b238ecd3` |

Release merge (`release/single-gpu-v1-merge`, 2026-09-05): two of these files legitimately moved
after 63271d5 and the table now records the blobs of the merged tree. `device_scvx_integration_test.cu`
carries the `integration/single-gpu-v1` claim-core amendment commits 26def2b, 8930817, 2ef27e1, 857f99a
and aca6500 (GPU-validated on that line: CUDA CTest, executor selftests and capability probes recorded
in `docs/G4_GATE_REPORT.md`); `recovery_test.cu` carries `proposal/g3-sanitizer-recovery-cap` 9fafee8 (a
20,000-iteration budget for the cancellation solve only under `--sanitizer`; host build only, not yet
executed on the GPU). Every other blob is unchanged from base 63271d5.

Changed CUDA sources: `device_scvx.cu` +370/−0 (pd3_fft/pd6_fft branches only; the 9a4cbea
workspace-wait fix retained), `device_scvx_c_api.h` +49 with CRLF→LF normalisation only
(`git diff --ignore-cr-at-eol` shows no deletions), `cpp/cuda/CMakeLists.txt` (new tool/tests),
plus the new `device_time_dilated_test.cu` and `tools/spacepdhcg_plan.cu`. The G4 capability
`e546583b…` is bound to source 9a4cbea and `libspacepdhcg_cuda.so` `84d98bcd…`; the candidate's
Release library is `af835ad8…`. No G4 claim-core evidence may be produced from the candidate until
it has its own G0–G3 seal and capability.

## Promotion once the campaign releases the worktree

```bash
cd /home/angus/worktrees/spacepdhcg-single-gpu-integration \
  && test -z "$(git status --porcelain=v1)" \
  && test "$(git rev-parse HEAD)" = 63271d58b78df343c1ae694525e460db31696f5d \
  && git merge --ff-only integration/single-gpu-v2-candidate
```

The candidate strictly descends from `integration/single-gpu-v1` at 63271d5, so this is a pure
fast-forward (no merge commit). If `integration/single-gpu-v1` has moved, merge it into the
candidate first and re-run the CPU matrix before promoting.
