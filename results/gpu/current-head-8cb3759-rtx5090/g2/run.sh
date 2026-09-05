#!/usr/bin/env bash
# Gate G2 evidence runner for the main 8cb3759 reseal on the WSL RTX 5090 (sm_120).
# Procedure = the sealed b6afb49 (RTX 5090) / 9e75b47 (H100) G2 template, unchanged in every
# test, tolerance and sanitizer target. Two operational additions for the shared host:
#   * gpu_guard: before every GPU step, wait while a foreign CUDA context (WSL nvidia-smi
#     compute apps, or a Windows-side python/torch/cuda workload seen by nvidia-smi.exe) holds
#     the GPU, and record every check/wait in foreign-gpu-waits.log;
#   * builds run at absolute nice 10 with --parallel 8; the runner itself is launched at nice 5.
set -Eeuo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-8cb3759-rtx5090/g2"
tool="/home/angus/spacecraft-trajectory-optmiser/.venv/bin"
gpu_py="/home/angus/spacecraft-trajectory-optmiser/.venv/bin/python"
gpu_site="/home/angus/spacecraft-trajectory-optmiser/.venv/lib/python3.12/site-packages"
qoco="${root}/build-current-head-qoco/libqoco.so"
cudss="${gpu_site}/nvidia/cu12"
shim="${root}/build-current-head-qoco-cudss-lib"

export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="/usr/local/cuda-12.8/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export SPACEPDHCG_QOCO_LIBRARY="${qoco}"
export LD_LIBRARY_PATH="${shim}:${cudss}/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${root}/src:${gpu_site}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

current_nice="$(nice)"
build_nice_delta=$((10 - current_nice))
if ((build_nice_delta < 0)); then build_nice_delta=0; fi

: >"${out}/commands.txt"
: >"${out}/foreign-gpu-waits.log"
printf 'status=RUNNING\nstarted_utc=%s\n' "$(date -u +%FT%TZ)" >"${out}/status.txt"
trap 'status=$?; printf "status=FAIL\nexit_code=%d\nfailed_utc=%s\n" "${status}" "$(date -u +%FT%TZ)" >"${out}/status.txt"; exit "${status}"' ERR

run_log() {
    local name="$1"
    shift
    printf '%q ' "$@" >>"${out}/commands.txt"
    printf '\n' >>"${out}/commands.txt"
    printf 'STEP %s %s\n' "$(date -u +%FT%TZ)" "${name}"
    "$@" >"${out}/${name}.log" 2>&1
    printf 'PASS %s %s\n' "$(date -u +%FT%TZ)" "${name}"
}

gpu_guard() {
    local step="$1"
    local waited=0
    local wsl_apps win_apps win_compute
    while :; do
        wsl_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr -d '\r' || true)"
        win_apps="$(/mnt/c/Windows/System32/nvidia-smi.exe --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr -d '\r' || true)"
        win_compute="$(printf '%s\n' "${win_apps}" | grep -i -E 'python|torch|cuda|jax|cupy|nvcc|ollama|blender|nsys|ncu' || true)"
        if [[ -z "${wsl_apps}" && -z "${win_compute}" ]]; then
            printf '%s step=%s clear waited_seconds=%d gpu_util_pct=%s mem_used_mib=%s\n' \
                "$(date -u +%FT%TZ)" "${step}" "${waited}" \
                "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' \r')" \
                "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')" \
                >>"${out}/foreign-gpu-waits.log"
            return 0
        fi
        if ((waited == 0)); then
            printf '%s step=%s WAITING foreign GPU process; wsl=[%s] windows_compute=[%s]\n' \
                "$(date -u +%FT%TZ)" "${step}" "${wsl_apps//$'\n'/;}" "${win_compute//$'\n'/;}" \
                >>"${out}/foreign-gpu-waits.log"
        fi
        sleep 30
        waited=$((waited + 30))
    done
}

run_gpu() {
    gpu_guard "$1"
    run_log "$@"
}

test "$(git rev-parse HEAD)" = "8cb3759b29ea8c7d843322a940a7ebcabfd9ff21"
test -z "$(git status --porcelain=v1)"
gpu_guard start

for kind in debug release; do
    if [[ "${kind}" == debug ]]; then
        build_type=Debug
    else
        build_type=RelWithDebInfo
    fi
    build="build-current-head-g2-${kind}"
    run_log "${kind}-configure" "${tool}/cmake" -S cpp -B "${build}" -G Ninja \
        "-DCMAKE_BUILD_TYPE=${build_type}" \
        -DSPACEPDHCG_BUILD_CUDA=ON \
        -DSPACEPDHCG_BUILD_DISTRIBUTED=OFF \
        -DSPACEPDHCG_BUILD_C_API=ON \
        -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
        -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
        -DCMAKE_CUDA_ARCHITECTURES=120
    run_log "${kind}-build" nice -n "${build_nice_delta}" "${tool}/cmake" --build "${build}" --clean-first --parallel 8
    run_gpu "${kind}-ctest" "${tool}/ctest" --test-dir "${build}" \
        --output-on-failure --no-tests=error
done

debug="build-current-head-g2-debug"
for test_name in \
    persistent_cw_test \
    persistent_soc_test \
    dlpack_contract_test \
    cone_inventory_test \
    allocation_lifecycle_test \
    pointer_contract_test \
    stream_lifetime_test \
    recovery_test; do
    if [[ "${test_name}" == persistent_soc_test ]]; then
        run_gpu "${test_name}" "${debug}/cuda-tests/${test_name}" --with-one-shot
    else
        run_gpu "${test_name}" "${debug}/cuda-tests/${test_name}"
    fi
done

for producer in cupy torch jax; do
    run_gpu "dlpack-${producer}" "${gpu_py}" scripts/gpu/dlpack_producer_compat.py \
        --producer "${producer}" \
        --library "${root}/${debug}/cuda/libspacepdhcg_cuda.so"
done

sanitizer=/usr/local/cuda-12.8/bin/compute-sanitizer
run_gpu sanitizer-memcheck "${sanitizer}" --tool memcheck --leak-check full \
    --error-exitcode 91 "${debug}/cuda-tests/persistent_cw_test"
run_gpu sanitizer-dlpack-memcheck "${sanitizer}" --tool memcheck --leak-check full \
    --error-exitcode 92 "${debug}/cuda-tests/dlpack_contract_test"
run_gpu sanitizer-racecheck "${sanitizer}" --tool racecheck --error-exitcode 93 \
    "${debug}/cuda-tests/persistent_cw_test"
run_gpu sanitizer-initcheck "${sanitizer}" --tool initcheck --track-unused-memory \
    --error-exitcode 94 "${debug}/cuda-tests/persistent_soc_test"
run_gpu sanitizer-synccheck "${sanitizer}" --tool synccheck --error-exitcode 95 \
    "${debug}/cuda-tests/persistent_cw_test"

{
    printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'source_tree=%s\n' "$(git rev-parse "HEAD^{tree}")"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'cuda_architectures=120\n'
    printf 'hardware_id=local-rtx-5090\n'
    printf 'upstream_commit=%s\n' "$(git -C _upstream/pdhcg rev-parse HEAD)"
    printf 'upstream_tree=%s\n' "$(git -C _upstream/pdhcg rev-parse "HEAD^{tree}")"
    sha256sum \
        third_party/pdhcg.lock.json \
        third_party/patches/pdhcg/0001-free-quadratic-state.patch \
        "${debug}/cuda/libspacepdhcg_cuda.so" \
        build-current-head-g2-release/cuda/libspacepdhcg_cuda.so
    printf 'build_nice=%d parallel=8 runner_nice=%d\n' "$((current_nice + build_nice_delta))" "${current_nice}"
    nvidia-smi
    "${gpu_py}" -m pip show cupy-cuda12x torch jax jaxlib jax-cuda12-plugin
} >"${out}/manifest.txt" 2>&1

test -z "$(git status --porcelain=v1)"
printf 'status=PASS\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${out}/status.txt"
