#!/usr/bin/env bash
set -Eeuo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-9e75b47-h100/g3"
tool="/home/ubuntu/spacepdhcg/v1/.venv/bin"
py="${root}/.venv/bin/python"
gpu_site="/home/ubuntu/spacepdhcg/v1/.venv/lib/python3.12/site-packages"
qoco="${root}/build-current-head-qoco/libqoco.so"
cudss="${gpu_site}/nvidia/cu12"
shim="${root}/build-current-head-qoco-cudss-lib"
release="build-single-gpu-cuda-release"
debug="build-current-head-g3-debug"

export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="/usr/local/cuda-12.8/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export SPACEPDHCG_HARDWARE_ID=lambda-h100-80gb-hbm3
export SPACEPDHCG_CUDA_ARCHITECTURES=90
export SPACEPDHCG_QOCO_LIBRARY="${qoco}"
export LD_LIBRARY_PATH="${shim}:${cudss}/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${root}/src:${gpu_site}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: >"${out}/commands.txt"
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

test "$(git rev-parse HEAD)" = "9e75b470fd5378d9b20f6f13892ac909c43757cd"
test -z "$(git status --porcelain=v1)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"

run_log release-configure "${tool}/cmake" -S cpp -B "${release}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPACEPDHCG_BUILD_CUDA=ON \
    -DSPACEPDHCG_BUILD_DISTRIBUTED=OFF \
    -DSPACEPDHCG_BUILD_C_API=ON \
    -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
    -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90
run_log release-build "${tool}/cmake" --build "${release}" --clean-first --parallel 8
run_log release-ctest "${tool}/ctest" --test-dir "${release}" \
    --output-on-failure --no-tests=error

run_log debug-configure "${tool}/cmake" -S cpp -B "${debug}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSPACEPDHCG_BUILD_CUDA=ON \
    -DSPACEPDHCG_BUILD_DISTRIBUTED=OFF \
    -DSPACEPDHCG_BUILD_C_API=ON \
    -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
    -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90
run_log debug-build "${tool}/cmake" --build "${debug}" --clean-first --parallel 8
run_log debug-ctest "${tool}/ctest" --test-dir "${debug}" \
    --output-on-failure --no-tests=error

release_exe="${release}/cuda-tests/device_scvx_integration_test"
debug_exe="${debug}/cuda-tests/device_scvx_integration_test"

run_log device-variational "${debug}/cuda-tests/device_variational_test"
run_log device-integration "${debug_exe}"
run_log tight-all "${release_exe}" --tight-all
run_log production-outer "${release_exe}" --production-outer
run_log p1d-path-audit "${release_exe}" --p1d-path-audit
run_log qoco-handback "${release_exe}" --qoco-handback
run_log recovery "${release}/cuda-tests/recovery_test"
run_log displaced-regressions "${py}" "${out}/run_displaced_regressions.py" \
    --executable "${release_exe}" \
    --output "${out}/displaced"

run_expected_failure no-device-negative-control env CUDA_VISIBLE_DEVICES= \
    "${release_exe}" --production-outer

run_log h1 "${py}" scripts/gpu/run_g3_h1.py \
    --repository . \
    --executable "${release_exe}" \
    --output "${out}/h1"

sanitizer=/usr/local/cuda-12.8/bin/compute-sanitizer
for tool_name in memcheck racecheck synccheck; do
    run_log "sanitizer-variational-${tool_name}" "${sanitizer}" \
        --tool "${tool_name}" --error-exitcode 91 \
        "${debug}/cuda-tests/device_variational_test"
    run_log "sanitizer-integration-${tool_name}" "${sanitizer}" \
        --tool "${tool_name}" --error-exitcode 92 \
        "${debug_exe}" --sanitizer
    run_log "sanitizer-recovery-${tool_name}" "${sanitizer}" \
        --tool "${tool_name}" --error-exitcode 93 \
        "${debug}/cuda-tests/recovery_test" --sanitizer
    run_log "sanitizer-production-${tool_name}" "${sanitizer}" \
        --tool "${tool_name}" --error-exitcode 94 \
        "${debug_exe}" --production-outer-sanitizer
done
run_log sanitizer-variational-initcheck "${sanitizer}" \
    --tool initcheck --track-unused-memory --error-exitcode 95 \
    "${debug}/cuda-tests/device_variational_test"
run_log sanitizer-integration-initcheck "${sanitizer}" \
    --tool initcheck --track-unused-memory --error-exitcode 96 \
    "${debug_exe}" --sanitizer
run_log sanitizer-recovery-initcheck "${sanitizer}" \
    --tool initcheck --track-unused-memory --error-exitcode 97 \
    "${debug}/cuda-tests/recovery_test" --sanitizer
run_log sanitizer-production-initcheck "${sanitizer}" \
    --tool initcheck --track-unused-memory --error-exitcode 98 \
    "${debug_exe}" --production-outer-sanitizer

profile="${out}/device-scvx"
run_log nsys-profile nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
    --force-overwrite=true --output="${profile}" "${debug_exe}"
run_log nsys-stats nsys stats --force-export=true --report \
    cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "${profile}.nsys-rep"

{
    printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'source_tree=%s\n' "$(git rev-parse "HEAD^{tree}")"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'upstream_commit=%s\n' "$(git -C _upstream/pdhcg rev-parse HEAD)"
    printf 'upstream_tree=%s\n' "$(git -C _upstream/pdhcg rev-parse "HEAD^{tree}")"
    sha256sum \
        benchmarks/g4_policy.json \
        benchmarks/g4_policy.sha256 \
        benchmarks/paper1_matrix.json \
        "${out}/displaced/execution-context.json" \
        "${release_exe}" \
        "${debug_exe}" \
        "${qoco}"
    nvidia-smi
    nsys --version
} >"${out}/manifest.txt" 2>&1

test -z "$(git status --porcelain=v1)"
printf 'status=PASS\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${out}/status.txt"
