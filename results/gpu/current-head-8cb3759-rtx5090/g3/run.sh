#!/usr/bin/env bash
# Gate G3 evidence runner for the main 8cb3759 reseal on the WSL RTX 5090 (sm_120).
# Procedure = the sealed b6afb49 (RTX 5090) / 9e75b47 (H100) G3 template, unchanged in every
# test, tolerance, timeout and sanitizer target. Two operational additions for the shared host:
#   * gpu_guard: before every GPU step, wait while a foreign CUDA context (WSL nvidia-smi
#     compute apps, or a Windows-side python/torch/cuda workload seen by nvidia-smi.exe) holds
#     the GPU, and record every check/wait in foreign-gpu-waits.log;
#   * builds run at absolute nice 10 with --parallel 8; the runner itself is launched at nice 5.
set -Eeuo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-8cb3759-rtx5090/g3"
tool="/home/angus/spacecraft-trajectory-optmiser/.venv/bin"
py="${root}/.venv-current-head/bin/python"
gpu_site="/home/angus/spacecraft-trajectory-optmiser/.venv/lib/python3.12/site-packages"
qoco="${root}/build-current-head-qoco/libqoco.so"
cudss="${gpu_site}/nvidia/cu12"
shim="${root}/build-current-head-qoco-cudss-lib"
release="build-single-gpu-cuda-release"
debug="build-current-head-g3-debug"

export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="/usr/local/cuda-12.8/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export SPACEPDHCG_HARDWARE_ID=local-rtx-5090
export SPACEPDHCG_CUDA_ARCHITECTURES=120
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
    trap 'status=$?; printf "status=FAIL\nexit_code=%d\nfailed_utc=%s\n" "${status}" "$(date -u +%FT%TZ)" >"${out}/status.txt"; exit "${status}"' ERR
    printf '\nexit_code=%d\n' "${status}" >>"${out}/${name}.log"
    test "${status}" -ne 0
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

run_log release-configure "${tool}/cmake" -S cpp -B "${release}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPACEPDHCG_BUILD_CUDA=ON \
    -DSPACEPDHCG_BUILD_DISTRIBUTED=OFF \
    -DSPACEPDHCG_BUILD_C_API=ON \
    -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
    -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120
run_log release-build nice -n "${build_nice_delta}" "${tool}/cmake" --build "${release}" --clean-first --parallel 8
run_gpu release-ctest "${tool}/ctest" --test-dir "${release}" \
    --output-on-failure --no-tests=error

run_log debug-configure "${tool}/cmake" -S cpp -B "${debug}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSPACEPDHCG_BUILD_CUDA=ON \
    -DSPACEPDHCG_BUILD_DISTRIBUTED=OFF \
    -DSPACEPDHCG_BUILD_C_API=ON \
    -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
    -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120
run_log debug-build nice -n "${build_nice_delta}" "${tool}/cmake" --build "${debug}" --clean-first --parallel 8
run_gpu debug-ctest "${tool}/ctest" --test-dir "${debug}" \
    --output-on-failure --no-tests=error

release_exe="${release}/cuda-tests/device_scvx_integration_test"
debug_exe="${debug}/cuda-tests/device_scvx_integration_test"

run_gpu device-variational "${debug}/cuda-tests/device_variational_test"
run_gpu device-integration "${debug_exe}"
run_gpu tight-all "${release_exe}" --tight-all
run_gpu production-outer "${release_exe}" --production-outer
run_gpu p1d-path-audit "${release_exe}" --p1d-path-audit
run_gpu qoco-handback "${release_exe}" --qoco-handback
run_gpu recovery "${release}/cuda-tests/recovery_test"
run_gpu displaced-regressions "${py}" "${out}/run_displaced_regressions.py" \
    --executable "${release_exe}" \
    --output "${out}/displaced"

run_expected_failure no-device-negative-control env CUDA_VISIBLE_DEVICES= \
    "${release_exe}" --production-outer

run_gpu h1 "${py}" scripts/gpu/run_g3_h1.py \
    --repository . \
    --executable "${release_exe}" \
    --output "${out}/h1"

sanitizer=/usr/local/cuda-12.8/bin/compute-sanitizer
for tool_name in memcheck racecheck synccheck; do
    run_gpu "sanitizer-variational-${tool_name}" "${sanitizer}" \
        --tool "${tool_name}" --error-exitcode 91 \
        "${debug}/cuda-tests/device_variational_test"
    run_gpu "sanitizer-integration-${tool_name}" "${sanitizer}" \
        --tool "${tool_name}" --error-exitcode 92 \
        "${debug_exe}" --sanitizer
    run_gpu "sanitizer-recovery-${tool_name}" "${sanitizer}" \
        --tool "${tool_name}" --error-exitcode 93 \
        "${debug}/cuda-tests/recovery_test" --sanitizer
    run_gpu "sanitizer-production-${tool_name}" "${sanitizer}" \
        --tool "${tool_name}" --error-exitcode 94 \
        "${debug_exe}" --production-outer-sanitizer
done
run_gpu sanitizer-variational-initcheck "${sanitizer}" \
    --tool initcheck --track-unused-memory --error-exitcode 95 \
    "${debug}/cuda-tests/device_variational_test"
run_gpu sanitizer-integration-initcheck "${sanitizer}" \
    --tool initcheck --track-unused-memory --error-exitcode 96 \
    "${debug_exe}" --sanitizer
run_gpu sanitizer-recovery-initcheck "${sanitizer}" \
    --tool initcheck --track-unused-memory --error-exitcode 97 \
    "${debug}/cuda-tests/recovery_test" --sanitizer
run_gpu sanitizer-production-initcheck "${sanitizer}" \
    --tool initcheck --track-unused-memory --error-exitcode 98 \
    "${debug_exe}" --production-outer-sanitizer

profile="${out}/device-scvx"
run_gpu nsys-profile nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
    --force-overwrite=true --output="${profile}" "${debug_exe}"
run_log nsys-stats nsys stats --force-export=true --report \
    cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "${profile}.nsys-rep"

{
    printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'source_tree=%s\n' "$(git rev-parse "HEAD^{tree}")"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'cuda_architectures=120\n'
    printf 'hardware_id=local-rtx-5090\n'
    printf 'upstream_commit=%s\n' "$(git -C _upstream/pdhcg rev-parse HEAD)"
    printf 'upstream_tree=%s\n' "$(git -C _upstream/pdhcg rev-parse "HEAD^{tree}")"
    sha256sum \
        benchmarks/g4_policy.json \
        benchmarks/g4_policy.sha256 \
        benchmarks/paper1_matrix.json \
        "${out}/displaced/execution-context.json" \
        "${release_exe}" \
        "${debug_exe}" \
        "${release}/cuda/libspacepdhcg_cuda.so" \
        "${debug}/cuda/libspacepdhcg_cuda.so" \
        "${qoco}"
    printf 'build_nice=%d parallel=8 runner_nice=%d\n' "$((current_nice + build_nice_delta))" "${current_nice}"
    nvidia-smi
    nsys --version
} >"${out}/manifest.txt" 2>&1

test -z "$(git status --porcelain=v1)"
printf 'status=PASS\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${out}/status.txt"
