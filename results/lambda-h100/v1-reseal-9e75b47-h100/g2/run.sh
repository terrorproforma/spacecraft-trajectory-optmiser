#!/usr/bin/env bash
set -Eeuo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-9e75b47-h100/g2"
tool="/home/ubuntu/spacepdhcg/v1/.venv/bin"
gpu_py="/home/ubuntu/spacepdhcg/v1/.venv/bin/python"
gpu_site="/home/ubuntu/spacepdhcg/v1/.venv/lib/python3.12/site-packages"
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

test "$(git rev-parse HEAD)" = "9e75b470fd5378d9b20f6f13892ac909c43757cd"
test -z "$(git status --porcelain=v1)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"

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
        -DCMAKE_CUDA_ARCHITECTURES=90
    run_log "${kind}-build" "${tool}/cmake" --build "${build}" --clean-first --parallel 8
    run_log "${kind}-ctest" "${tool}/ctest" --test-dir "${build}" \
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
        run_log "${test_name}" "${debug}/cuda-tests/${test_name}" --with-one-shot
    else
        run_log "${test_name}" "${debug}/cuda-tests/${test_name}"
    fi
done

for producer in cupy torch jax; do
    run_log "dlpack-${producer}" "${gpu_py}" scripts/gpu/dlpack_producer_compat.py \
        --producer "${producer}" \
        --library "${root}/${debug}/cuda/libspacepdhcg_cuda.so"
done

sanitizer=/usr/local/cuda-12.8/bin/compute-sanitizer
run_log sanitizer-memcheck "${sanitizer}" --tool memcheck --leak-check full \
    --error-exitcode 91 "${debug}/cuda-tests/persistent_cw_test"
run_log sanitizer-dlpack-memcheck "${sanitizer}" --tool memcheck --leak-check full \
    --error-exitcode 92 "${debug}/cuda-tests/dlpack_contract_test"
run_log sanitizer-racecheck "${sanitizer}" --tool racecheck --error-exitcode 93 \
    "${debug}/cuda-tests/persistent_cw_test"
run_log sanitizer-initcheck "${sanitizer}" --tool initcheck --track-unused-memory \
    --error-exitcode 94 "${debug}/cuda-tests/persistent_soc_test"
run_log sanitizer-synccheck "${sanitizer}" --tool synccheck --error-exitcode 95 \
    "${debug}/cuda-tests/persistent_cw_test"

{
    printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'source_tree=%s\n' "$(git rev-parse "HEAD^{tree}")"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'upstream_commit=%s\n' "$(git -C _upstream/pdhcg rev-parse HEAD)"
    printf 'upstream_tree=%s\n' "$(git -C _upstream/pdhcg rev-parse "HEAD^{tree}")"
    sha256sum \
        third_party/pdhcg.lock.json \
        third_party/patches/pdhcg/0001-free-quadratic-state.patch \
        "${debug}/cuda/libspacepdhcg_cuda.so" \
        build-current-head-g2-release/cuda/libspacepdhcg_cuda.so
    nvidia-smi
    "${gpu_py}" -m pip show cupy-cuda12x torch jax jaxlib jax-cuda12-plugin
} >"${out}/manifest.txt" 2>&1

test -z "$(git status --porcelain=v1)"
printf 'status=PASS\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${out}/status.txt"
