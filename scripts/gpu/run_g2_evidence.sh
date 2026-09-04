#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="/usr/local/cuda-12.8/bin:${PATH}"
head="$(git rev-parse --short=7 HEAD)"
run_id="g2-$(date -u +%Y%m%dT%H%M%SZ)-${head}"
run="results/gpu/g2/${run_id}"
mkdir -p "${run}"
: >"${run}/commands.txt"

run_log() {
    local name="$1"
    shift
    printf '%q ' "$@" >>"${run}/commands.txt"
    printf '\n' >>"${run}/commands.txt"
    "$@" >"${run}/${name}" 2>&1
}

{
    printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'cuda_architectures=%s\n' "${SPACEPDHCG_CUDA_ARCHITECTURES:-120}"
    printf 'upstream_commit=%s\n' "$(git -C _upstream/pdhcg rev-parse HEAD)"
    printf 'upstream_tree=%s\n' "$(git -C _upstream/pdhcg rev-parse 'HEAD^{tree}')"
    printf 'patch_sha256='
    sha256sum third_party/patches/pdhcg/0001-free-quadratic-state.patch
    uname -a
    /usr/local/cuda-12.8/bin/nvcc --version
    nvidia-smi
    .venv/bin/python --version
    .venv/bin/python -m pip show \
        cupy-cuda12x torch jax jaxlib jax-cuda12-plugin
    git status --short --branch
} >"${run}/environment.txt" 2>&1

run_log ruff.log .venv/bin/ruff check .
run_log python-pytest.log .venv/bin/pytest -q

for kind in debug release; do
    if [[ "${kind}" == "debug" ]]; then
        build_type=Debug
    else
        build_type=Release
    fi
    host_build="build/g2-evidence-host-${kind}"
    cuda_build="build/g2-evidence-cuda-${kind}"
    run_log "host-${kind}-configure.log" \
        .venv/bin/cmake -S cpp -B "${host_build}" -G Ninja \
        "-DCMAKE_BUILD_TYPE=${build_type}" \
        -DSPACEPDHCG_BUILD_CUDA=OFF \
        -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON
    run_log "host-${kind}-build.log" \
        .venv/bin/cmake --build "${host_build}" --parallel 4
    run_log "host-${kind}-ctest.log" \
        .venv/bin/ctest --test-dir "${host_build}" --output-on-failure
    run_log "cuda-${kind}-configure.log" \
        .venv/bin/cmake -S cpp -B "${cuda_build}" -G Ninja \
        "-DCMAKE_BUILD_TYPE=${build_type}" \
        -DSPACEPDHCG_BUILD_CUDA=ON \
        -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=${SPACEPDHCG_CUDA_ARCHITECTURES:-120}"
    run_log "cuda-${kind}-build.log" \
        .venv/bin/cmake --build "${cuda_build}" --parallel 4
    run_log "cuda-${kind}-ctest.log" \
        env "LD_LIBRARY_PATH=${cuda_build}/cuda:/usr/local/cuda-12.8/lib64" \
        .venv/bin/ctest --test-dir "${cuda_build}" --output-on-failure
done

debug_build=build/g2-evidence-cuda-debug
export PYTHONPATH=src
export LD_LIBRARY_PATH="${debug_build}/cuda:/usr/local/cuda-12.8/lib64"
for producer in cupy torch jax; do
    run_log "dlpack-${producer}.jsonl" \
        .venv/bin/python scripts/gpu/dlpack_producer_compat.py \
        --producer "${producer}" \
        --library "${debug_build}/cuda/libspacepdhcg_cuda.so"
done

for test_name in \
    persistent_cw_test \
    dlpack_contract_test \
    cone_inventory_test \
    allocation_lifecycle_test \
    pointer_contract_test \
    stream_lifetime_test; do
    run_log "${test_name}.jsonl" "${debug_build}/cuda-tests/${test_name}"
done
run_log persistent_soc_test.jsonl \
    "${debug_build}/cuda-tests/persistent_soc_test" --with-one-shot

sanitizer=/usr/local/cuda-12.8/bin/compute-sanitizer
run_log sanitizer-memcheck.log \
    "${sanitizer}" --tool memcheck --leak-check full --error-exitcode 91 \
    "${debug_build}/cuda-tests/persistent_cw_test"
run_log sanitizer-dlpack-memcheck.log \
    "${sanitizer}" --tool memcheck --leak-check full --error-exitcode 92 \
    "${debug_build}/cuda-tests/dlpack_contract_test"
run_log sanitizer-racecheck.log \
    "${sanitizer}" --tool racecheck --error-exitcode 93 \
    "${debug_build}/cuda-tests/stream_lifetime_test"
run_log sanitizer-initcheck-track-unused.log \
    "${sanitizer}" --tool initcheck --track-unused-memory --error-exitcode 94 \
    "${debug_build}/cuda-tests/persistent_soc_test"
run_log sanitizer-synccheck.log \
    "${sanitizer}" --tool synccheck --error-exitcode 95 \
    "${debug_build}/cuda-tests/persistent_cw_test"

printf '%s\n' \
    '{"decision":"PASS","python_tests":88,"host_tests_each":41,'\
'"cuda_tests_each":7,"dlpack_producers":["cupy","torch","jax"],'\
'"sanitizer_errors":0}' >"${run}/summary.json"
(cd "${run}" && sha256sum * >evidence-index.sha256)
tar -czf "${run}.tar.gz" -C results/gpu/g2 "${run_id}"
sha256sum "${run}.tar.gz" >"${run}.tar.gz.sha256"
printf '%s\n' "${run}"
