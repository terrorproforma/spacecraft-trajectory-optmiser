#!/usr/bin/env bash
set -Eeuo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-9e75b47-h100/g1"
tool="/home/ubuntu/spacepdhcg/v1/.venv/bin"
venv="${root}/build-current-head-g1-venv"
upstream="${root}/_upstream/pdhcg"
upstream_build="${root}/build-current-head-g1-pdhcg"
wheel="$(printf '%s\n' results/gpu/current-head-9e75b47-h100/g0/artifacts/spacepdhcg-*.whl)"

export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="/usr/local/cuda-12.8/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: >"${out}/commands.txt"
printf 'status=RUNNING\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" >"${out}/status.txt"
sampler_pid=""
cleanup() {
    if [[ -n "${sampler_pid}" ]]; then
        kill "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
}
trap 'status=$?; cleanup; printf "status=FAIL\nexit_code=%d\nfailed_utc=%s\n" "${status}" "$(date -u +%FT%TZ)" >"${out}/status.txt"; exit "${status}"' ERR

run_log() {
    local name="$1"
    shift
    printf '%q ' "$@" >>"${out}/commands.txt"
    printf '\n' >>"${out}/commands.txt"
    printf 'STEP %s %s\n' "$(date -u +%FT%TZ)" "${name}"
    "$@" >"${out}/${name}.log" 2>&1
    printf 'PASS %s %s\n' "$(date -u +%FT%TZ)" "${name}"
}

run_gpu_log() {
    local name="$1"
    shift
    printf '%q ' "$@" >>"${out}/commands.txt"
    printf '\n' >>"${out}/commands.txt"
    printf 'STEP %s %s\n' "$(date -u +%FT%TZ)" "${name}"
    nvidia-smi dmon -s pucm -d 1 -o DT >"${out}/${name}.nvidia-dmon.log" 2>&1 &
    sampler_pid=$!
    "$@" >"${out}/${name}.log" 2>&1
    local status=$?
    cleanup
    sampler_pid=""
    test "${status}" -eq 0
    printf 'PASS %s %s\n' "$(date -u +%FT%TZ)" "${name}"
}

run_expected_failure() {
    local name="$1"
    shift
    printf '%q ' "$@" >>"${out}/commands.txt"
    printf '\n' >>"${out}/commands.txt"
    trap - ERR
    set +e
    "$@" >"${out}/${name}.log" 2>&1
    local status=$?
    set -e
    trap 'status=$?; cleanup; printf "status=FAIL\nexit_code=%d\nfailed_utc=%s\n" "${status}" "$(date -u +%FT%TZ)" >"${out}/status.txt"; exit "${status}"' ERR
    printf '\nexit_code=%d\n' "${status}" >>"${out}/${name}.log"
    test "${status}" -ne 0
}

test "$(git rev-parse HEAD)" = "9e75b470fd5378d9b20f6f13892ac909c43757cd"
test -z "$(git status --porcelain=v1)"
test "$(git -C "${upstream}" rev-parse HEAD)" = \
    "167c8b72b4b96d2f94d405b8763e485514192b81"
test "$(git -C "${upstream}" rev-parse "HEAD^{tree}")" = \
    "62b05e6c1bedd385f6c267af3645ae4aae0421b4"
test -z "$(git -C "${upstream}" status --porcelain=v1)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"

run_log upstream-configure "${tool}/cmake" -S "${upstream}" -B "${upstream_build}" \
    -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DPDHCG_COMPILE_DISTRIBUTED=OFF \
    -DPDHCG_BUILD_STATIC_LIB=ON \
    -DPDHCG_BUILD_CLI=ON
run_log upstream-build "${tool}/cmake" --build "${upstream_build}" --clean-first --parallel 8
run_log upstream-cli-help "${upstream_build}/pdhcg" --help

run_log g1-venv /home/ubuntu/.local/bin/uv venv --python 3.12 --clear "${venv}"
run_log install-spacepdhcg /home/ubuntu/.local/bin/uv pip install \
    --python "${venv}/bin/python" "${wheel}"
run_log install-pdhcg env \
    CUDACXX=/usr/local/cuda-12.8/bin/nvcc \
    SKBUILD_CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=90 \
    /home/ubuntu/.local/bin/uv pip install \
    --python "${venv}/bin/python" --no-deps --reinstall "${upstream}"

for intervals in 8 32 128; do
    run_gpu_log "automated-banded-n${intervals}" \
        "${venv}/bin/spacepdhcg-banded-correctness" \
        --pdhcg \
        --seeds 17 29 41 53 71 \
        --intervals "${intervals}" \
        --tolerance 1e-6
done

run_gpu_log declared-expansion \
    "${venv}/bin/spacepdhcg-g1-correctness" \
    --intervals 20 50 100 500 \
    --tolerances 1e-3 1e-4 1e-6 1e-8 \
    --start-modes cold primal primal-dual \
    --include-updates \
    --output "${out}/declared-expansion.json"

run_expected_failure no-device-negative-control \
    env CUDA_VISIBLE_DEVICES= \
    "${venv}/bin/spacepdhcg-g1-correctness" \
    --intervals 8 \
    --tolerances 1e-6 \
    --start-modes cold \
    --output "${out}/no-device-unexpected.json"

{
    printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'source_tree=%s\n' "$(git rev-parse "HEAD^{tree}")"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'upstream_commit=%s\n' "$(git -C "${upstream}" rev-parse HEAD)"
    printf 'upstream_tree=%s\n' "$(git -C "${upstream}" rev-parse "HEAD^{tree}")"
    printf 'upstream_status=%q\n' "$(git -C "${upstream}" status --porcelain=v1)"
    "${venv}/bin/python" -c \
        'import json,pdhcg,spacepdhcg; print(json.dumps({"pdhcg":pdhcg.__version__,"spacepdhcg":spacepdhcg.__version__},sort_keys=True))'
    sha256sum \
        third_party/pdhcg.lock.json \
        "${upstream_build}/pdhcg" \
        "${venv}/lib/python3.12/site-packages/pdhcg/_pdhcg_core"*.so
    nvidia-smi
} >"${out}/manifest.txt" 2>&1

"${venv}/bin/python" - "${out}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
payload = json.loads((out / "declared-expansion.json").read_text(encoding="utf-8"))
cases = payload["cases"]
updates = payload["update_cases"]
expected_cases = (
    len(payload["intervals"])
    * 2
    * len(payload["tolerances"])
    * len(payload["start_modes"])
)
expected_updates = len(payload["intervals"]) * 2 * 2
assert len(cases) == expected_cases
assert len(updates) == expected_updates
assert all(row["qualified"] for row in cases + updates)
quality = [row["quality"] for row in cases + updates]
keys = (
    "objective_gap_relative",
    "scalar_primal_violation_inf",
    "variable_primal_violation_inf",
    "cone_primal_violation_inf",
    "stationarity_residual_inf",
    "natural_residual_inf",
    "relative_natural_residual_inf",
    "initial_error_inf",
    "terminal_error_inf",
    "dynamics_defect_inf",
    "control_violation_inf",
)
summary = {
    "schema_version": "current-head-g1-1.0.0",
    "decision": "PASS",
    "source_commit": "9e75b470fd5378d9b20f6f13892ac909c43757cd",
    "pdhcg_commit": "167c8b72b4b96d2f94d405b8763e485514192b81",
    "pdhcg_tree": "62b05e6c1bedd385f6c267af3645ae4aae0421b4",
    "automated_banded_cases": 15,
    "declared_cases": len(cases),
    "update_cases": len(updates),
    "box_cases": sum(row["thrust_constraint"] == "box" for row in cases),
    "soc_cases": sum(row["thrust_constraint"] == "soc" for row in cases),
    "maximum": {key: max(float(row[key]) for row in quality) for key in keys},
    "no_device_negative_control": "PASS",
    "local_only": True,
    "immutable_uri": None,
}
(out / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY

test -z "$(git status --porcelain=v1)"
printf 'status=PASS\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${out}/status.txt"
