#!/bin/bash
# v2 candidate: GPU-deferred validation manifest (benchmarks/gpu_deferred_validation_v2.json) on the H100.
# Serialised GPU use: refuses to start while the v1 reseal or any G4 process is alive.
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export V2="$HOME/spacepdhcg/v2"
cd "$V2"
export PATH="$V2/.venv/bin:/usr/local/cuda-12.8/bin:$HOME/.local/node/bin:$PATH"
export PYTHONPATH="$V2/src"
export CUDA_VISIBLE_DEVICES=0
export SPACEPDHCG_NATIVE_LIBRARY="$V2/build-v2-relwithdebinfo/libspacepdhcg.so"
export SPACEPDHCG_QOCO_LIBRARY="$HOME/spacepdhcg/v1/build-current-head-qoco/libqoco.so"
export LD_LIBRARY_PATH="$HOME/spacepdhcg/v1/build-current-head-qoco-cudss-lib:$V2/.venv/lib/python3.12/site-packages/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SPACEPDHCG_GTOC12_DATA="$HOME/spacepdhcg/gtoc12/benchmarks/gtoc12/data"
head=$(git rev-parse HEAD)
out="$V2/results/gpu/h100-deferred-${head:0:7}"
mkdir -p "$out" build-v2-gpu-deferred
: > "$out/commands.txt"
: > "$out/items.tsv"
printf 'status=RUNNING\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" > "$out/status.txt"

item() { # id expected_exit command...
  local id=$1 expected=$2; shift 2
  printf '%s\t%q' "$id" "$1" >> "$out/commands.txt"; shift; for a in "$@"; do printf ' %q' "$a" >> "$out/commands.txt"; done; printf '\n' >> "$out/commands.txt"
  set -- "${@}"
  log "STEP $id"
  local t0=$SECONDS
  "$@" > "$out/$id.log" 2>&1
  local rc=$?
  local verdict=PASS; [ "$rc" -eq "$expected" ] || verdict=FAIL
  printf '%s\t%s\texit=%s\texpected=%s\t%ss\n' "$id" "$verdict" "$rc" "$expected" "$((SECONDS-t0))" >> "$out/items.tsv"
  log "$verdict  $id exit=$rc (expected $expected) $((SECONDS-t0))s"
}
# item() consumed the command as "$@" after shifting id/expected; re-implement plainly:
item() {
  local id=$1 expected=$2; shift 2
  { printf '%s\t' "$id"; printf '%q ' "$@"; printf '\n'; } >> "$out/commands.txt"
  log "STEP $id"
  local t0=$SECONDS
  "$@" > "$out/$id.log" 2>&1
  local rc=$?
  local verdict=PASS; [ "$rc" -eq "$expected" ] || verdict=FAIL
  printf '%s\t%s\texit=%s\texpected=%s\t%ss\n' "$id" "$verdict" "$rc" "$expected" "$((SECONDS-t0))" >> "$out/items.tsv"
  log "$verdict  $id exit=$rc (expected $expected) $((SECONDS-t0))s"
}

log "== gate"
test -z "$(pgrep -f 'run_g4_campaign.py|--g4-server|--g4-session')" || { log "G4 process alive: refuse"; exit 3; }
test -z "$(pgrep -f 'reseal_all.sh')" || { log "v1 reseal still running: refuse"; exit 3; }
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { log "GPU busy: refuse"; exit 3; }
test -z "$(git status --porcelain=v1)" || { log "v2 tree dirty: refuse"; git status --short; exit 4; }
{
  printf 'source_commit=%s\nsource_tree=%s\nbranch=%s\n' "$head" "$(git rev-parse 'HEAD^{tree}')" "$(git branch --show-current)"
  printf 'hardware_id=lambda-h100-80gb-hbm3\ncuda_architectures=90\n'
  printf 'qoco_library=%s\n' "$SPACEPDHCG_QOCO_LIBRARY"
  sha256sum "$SPACEPDHCG_QOCO_LIBRARY" "$SPACEPDHCG_NATIVE_LIBRARY" build-v2-cuda-release/cuda/libspacepdhcg_cuda.so build-v2-cuda-debug/cuda/libspacepdhcg_cuda.so \
    build-v2-cuda-release/cuda-tests/device_scvx_integration_test build-v2-cuda-release/cuda-tools/spacepdhcg_plan build-v2-cuda-release/cuda-tests/device_time_dilated_test
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader
  nvcc --version | tail -1; cmake --version | head -1; node --version; python --version
} > "$out/manifest.txt" 2>&1
item gate-gpu-preflight 0 spacepdhcg literature gpu-preflight

subset='spacepdhcg_plan_capabilities|device_variational_test|persistent_cw_test|pointer_contract_test|allocation_lifecycle_test|stream_lifetime_test|cone_inventory_test|dlpack_contract_test|recovery_test|persistent_soc_test'
item planner-cuda-ctest-subset-release 0 ctest --test-dir build-v2-cuda-release --output-on-failure -R "$subset"
item planner-cuda-ctest-subset-debug 0 ctest --test-dir build-v2-cuda-debug --output-on-failure -R "$subset"
item literature-device-time-dilated-ctest-release 0 ctest --test-dir build-v2-cuda-release --output-on-failure -R device_time_dilated_test
item literature-device-time-dilated-ctest-debug 0 ctest --test-dir build-v2-cuda-debug --output-on-failure -R device_time_dilated_test
item gtoc12-gpu-lambert-parity-ctest 0 ctest --test-dir build-v2-cuda-release --output-on-failure -R 'orbitweaver_gpu_test|orbitweaver_g3_adapter_test'
item gtoc12-gpu-lambert-parity-pytest 0 python -m pytest -q -p no:cacheprovider tests/test_gtoc12_pipeline.py -k lambert

item planner-gpu-pytest 0 env SPACEPDHCG_PLANNER_GPU_TESTS=1 SPACEPDHCG_PLAN_EXECUTABLE="$V2/build-v2-cuda-release/cuda-tools/spacepdhcg_plan" python -m pytest -q -p no:cacheprovider tests/test_planner_gpu.py

for f in hcw_rendezvous powered_descent_3dof powered_descent_6dof low_thrust; do
  item "planner-gpu-example-$f" 0 spacepdhcg plan "examples/planner/$f.json" --executable build-v2-cuda-release/cuda-tools/spacepdhcg_plan --output "build-v2-gpu-deferred/plan-$f" --export-viewer "build-v2-gpu-deferred/plan-$f/viewer"
  item "planner-gpu-example-$f-viewer-check" 0 bash -c "cd build-v2-gpu-deferred/plan-$f/viewer && node scripts/check.mjs"
done

item planner-memcheck-hcw 0 compute-sanitizer --tool memcheck --leak-check full build-v2-cuda-release/cuda-tools/spacepdhcg_plan examples/planner/hcw_rendezvous.json --output build-v2-gpu-deferred/memcheck-hcw.json --quiet
item planner-memcheck-pd3 0 compute-sanitizer --tool memcheck --leak-check full build-v2-cuda-release/cuda-tools/spacepdhcg_plan examples/planner/powered_descent_3dof.json --output build-v2-gpu-deferred/memcheck-pd3.json --quiet
for tool in initcheck synccheck racecheck; do
  item "planner-$tool-hcw" 0 compute-sanitizer --tool "$tool" build-v2-cuda-release/cuda-tools/spacepdhcg_plan examples/planner/hcw_rendezvous.json --output "build-v2-gpu-deferred/$tool-hcw.json" --quiet
done
item planner-memcheck-results 0 python -c "import json; [print(f, json.load(open(f))['status']['code'], json.load(open(f))['certificate']['certified'], json.load(open(f)).get('objective')) for f in ('build-v2-gpu-deferred/memcheck-hcw.json','build-v2-gpu-deferred/memcheck-pd3.json')]"

for tool in memcheck racecheck initcheck synccheck; do
  item "literature-time-dilated-$tool" 0 compute-sanitizer --tool "$tool" build-v2-cuda-release/cuda-tests/device_time_dilated_test
done

item literature-gpu-run 2 spacepdhcg literature gpu-run acikmese-ploen-2007-pd3 blackmore-2010-pd3-case1 chari-2024-pd6-monte-carlo
item literature-gpu-run-git-status 0 git status --porcelain -- results/literature docs/REFERENCE_REPRODUCTION_REPORT.md benchmarks/literature/reference_reproduction.json

item full-cuda-ctest-release 0 ctest --test-dir build-v2-cuda-release --output-on-failure --no-tests=error
item full-cuda-ctest-debug 0 ctest --test-dir build-v2-cuda-debug --output-on-failure --no-tests=error

log "== collect artefacts"
mkdir -p "$out/build-v2-gpu-deferred" "$out/literature"
rsync -a --exclude 'viewer/trajectories.json' build-v2-gpu-deferred/ "$out/build-v2-gpu-deferred/"
cp results/literature/acikmese-ploen-2007-pd3.json results/literature/blackmore-2010-pd3-case1.json results/literature/chari-2024-pd6-monte-carlo.json "$out/literature/" 2>/dev/null
cp docs/REFERENCE_REPRODUCTION_REPORT.md benchmarks/literature/reference_reproduction.json "$out/literature/" 2>/dev/null
git diff --stat -- docs/REFERENCE_REPRODUCTION_REPORT.md benchmarks/literature/reference_reproduction.json > "$out/literature/tracked-diffstat.txt"

python - "$out" "$head" <<'PY'
import json, re, sys, pathlib
out = pathlib.Path(sys.argv[1]); head = sys.argv[2]
rows = [l.split("\t") for l in (out / "items.tsv").read_text().splitlines() if l.strip()]
items = {r[0]: {"verdict": r[1], "exit": int(r[2].split("=")[1]), "expected_exit": int(r[3].split("=")[1]), "seconds": int(r[4][:-1])} for r in rows}
def text(name): 
    p = out / f"{name}.log"; return p.read_text(errors="replace") if p.exists() else ""
def ctest_count(name):
    m = re.search(r"100% tests passed, 0 tests failed out of (\d+)", text(name)); return int(m.group(1)) if m else None
def sanitizer_clean(name):
    t = text(name); return ("ERROR SUMMARY: 0 errors" in t) or ("RACECHECK SUMMARY: 0 hazards" in t)
def pytest_counts(name):
    m = re.search(r"(\d+) passed(?:, (\d+) skipped)?", text(name)); return {"passed": int(m.group(1)), "skipped": int(m.group(2) or 0)} if m else None
lit = {}
for tid in ("acikmese-ploen-2007-pd3", "blackmore-2010-pd3-case1", "chari-2024-pd6-monte-carlo"):
    p = out / "literature" / f"{tid}.json"
    if p.exists():
        d = json.loads(p.read_text()); m = d.get("measured", {}) or {}
        lit[tid] = {"status": d.get("status"), "scvx_qoco_gpu_status": m.get("scvx_qoco_gpu_status"), "scvx_qoco_gpu_fuel_used_kg": m.get("scvx_qoco_gpu_fuel_used_kg"),
                    "gpu_persistent_batch": (d.get("details", {}) or {}).get("gpu_persistent_batch")}
summary = {
  "schema_version": "h100-deferred-validation-1.0.0",
  "source_commit": head, "hardware_id": "lambda-h100-80gb-hbm3", "cuda_architecture": 90,
  "manifest": "benchmarks/gpu_deferred_validation_v2.json",
  "items": items,
  "all_pass": all(v["verdict"] == "PASS" for v in items.values()),
  "ctest_counts": {k: ctest_count(k) for k in items if "ctest" in k},
  "sanitizer_clean": {k: sanitizer_clean(k) for k in items if any(t in k for t in ("memcheck", "racecheck", "initcheck", "synccheck")) and not k.endswith("results")},
  "pytest": {k: pytest_counts(k) for k in items if "pytest" in k},
  "literature": lit,
  "local_only": True, "immutable_uri": None,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: summary[k] for k in ("all_pass", "ctest_counts", "sanitizer_clean", "pytest")}, indent=1))
print(json.dumps(lit, indent=1)[:3000])
PY
printf 'status=%s\ncompleted_utc=%s\n' "$(python -c "import json;print('PASS' if json.load(open('$out/summary.json'))['all_pass'] else 'FAIL')")" "$(date -u +%FT%TZ)" > "$out/status.txt"
cat "$out/status.txt"; cat "$out/items.tsv"
log "== done: $out"
