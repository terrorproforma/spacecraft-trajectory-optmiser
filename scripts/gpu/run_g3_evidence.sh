#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="/usr/local/cuda-12.8/bin:${PATH}"

head="$(git rev-parse --short=7 HEAD)"
run_id="g3-$(date -u +%Y%m%dT%H%M%SZ)-${head}"
run="results/gpu/g3/${run_id}"
mkdir -p "${run}"
: >"${run}/commands.txt"

run_log() {
    local name="$1"
    shift
    printf '%q ' "$@" >>"${run}/commands.txt"
    printf '\n' >>"${run}/commands.txt"
    "$@" >"${run}/${name}" 2>&1
}

run_expected_failure() {
    local name="$1"
    shift
    printf '%q ' "$@" >>"${run}/commands.txt"
    printf '\n' >>"${run}/commands.txt"
    set +e
    "$@" >"${run}/${name}" 2>&1
    local status=$?
    set -e
    printf '\nexit_code=%d\n' "${status}" >>"${run}/${name}"
    if [[ "${status}" -eq 0 ]]; then
        printf 'expected negative qualification unexpectedly passed\n' >&2
        return 1
    fi
}

{
    printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'cuda_architectures=%s\n' "${SPACEPDHCG_CUDA_ARCHITECTURES:-120}"
    printf 'source_tree=%s\n' "$(git rev-parse 'HEAD^{tree}')"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'upstream_commit=%s\n' "$(git -C _upstream/pdhcg rev-parse HEAD)"
    printf 'upstream_tree=%s\n' "$(git -C _upstream/pdhcg rev-parse 'HEAD^{tree}')"
    printf 'patch_sha256='
    sha256sum third_party/patches/pdhcg/0001-free-quadratic-state.patch
    printf 'paper1_matrix_sha256='
    sha256sum benchmarks/paper1_matrix.json
    uname -a
    /usr/local/cuda-12.8/bin/nvcc --version
    nvidia-smi
    nsys --version
    .venv/bin/python --version
    git status --short --branch
} >"${run}/environment.txt" 2>&1

cp benchmarks/paper1_matrix.json "${run}/paper1_matrix.json"
run_log ruff.log .venv/bin/ruff check .
run_log python-pytest.log .venv/bin/pytest -q

for kind in debug release; do
    if [[ "${kind}" == "debug" ]]; then
        build_type=Debug
    else
        build_type=Release
    fi
    build="build/g3-evidence-cuda-${kind}"
    run_log "${kind}-configure.log" \
        .venv/bin/cmake -S cpp -B "${build}" -G Ninja \
        "-DCMAKE_BUILD_TYPE=${build_type}" \
        -DSPACEPDHCG_BUILD_CUDA=ON \
        -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=${SPACEPDHCG_CUDA_ARCHITECTURES:-120}"
    run_log "${kind}-build.log" \
        .venv/bin/cmake --build "${build}" --parallel 4
    run_log "${kind}-ctest.log" \
        env "LD_LIBRARY_PATH=${build}/cuda:/usr/local/cuda-12.8/lib64" \
        .venv/bin/ctest --test-dir "${build}" --output-on-failure
done

debug_build=build/g3-evidence-cuda-debug
export LD_LIBRARY_PATH="${debug_build}/cuda:/usr/local/cuda-12.8/lib64"
run_log device-variational.jsonl \
    "${debug_build}/cuda-tests/device_variational_test"
run_log device-scvx-integration.jsonl \
    "${debug_build}/cuda-tests/device_scvx_integration_test"
run_log pd3-cqp.txt \
    "${debug_build}/cuda-tests/device_scvx_integration_test" --dump-pd3
run_log pd3-cpu-independent.json \
    .venv/bin/python scripts/gpu/diagnose_g3_pd3.py \
    --dump "${run}/pd3-cqp.txt"
run_log pd3-upstream-independent.json \
    .venv/bin/python scripts/gpu/diagnose_g3_pd3.py \
    --dump "${run}/pd3-cqp.txt" \
    --upstream-variant default
run_log pd3-upstream-warm-convention.json \
    .venv/bin/python scripts/gpu/diagnose_g3_pd3.py \
    --dump "${run}/pd3-cqp.txt" \
    --upstream-variant default \
    --upstream-start cpu-primal-dual \
    --iteration-limit 1000
release_build=build/g3-evidence-cuda-release
export LD_LIBRARY_PATH="${release_build}/cuda:/usr/local/cuda-12.8/lib64"
run_log tight-all.jsonl \
    "${release_build}/cuda-tests/device_scvx_integration_test" --tight-all
run_log production-outer.jsonl \
    "${release_build}/cuda-tests/device_scvx_integration_test" --production-outer
run_log recovery.jsonl \
    "${release_build}/cuda-tests/recovery_test"
run_expected_failure no-device-negative-control.log \
    env CUDA_VISIBLE_DEVICES= \
    "${release_build}/cuda-tests/device_scvx_integration_test" --production-outer
run_log h1.log \
    .venv/bin/python scripts/gpu/run_g3_h1.py \
    --repository . \
    --executable "${release_build}/cuda-tests/device_scvx_integration_test" \
    --output "${run}/h1"

sanitizer=/usr/local/cuda-12.8/bin/compute-sanitizer
export LD_LIBRARY_PATH="${debug_build}/cuda:/usr/local/cuda-12.8/lib64"
for tool in memcheck racecheck synccheck; do
    run_log "sanitizer-variational-${tool}.log" \
        "${sanitizer}" --tool "${tool}" --error-exitcode 91 \
        "${debug_build}/cuda-tests/device_variational_test"
    run_log "sanitizer-integration-${tool}.log" \
        "${sanitizer}" --tool "${tool}" --error-exitcode 92 \
        "${debug_build}/cuda-tests/device_scvx_integration_test" --sanitizer
    run_log "sanitizer-recovery-${tool}.log" \
        "${sanitizer}" --tool "${tool}" --error-exitcode 95 \
        "${debug_build}/cuda-tests/recovery_test" --sanitizer
    run_log "sanitizer-production-${tool}.log" \
        "${sanitizer}" --tool "${tool}" --error-exitcode 96 \
        "${debug_build}/cuda-tests/device_scvx_integration_test" \
        --production-outer-sanitizer
done
run_log sanitizer-variational-initcheck.log \
    "${sanitizer}" --tool initcheck --track-unused-memory --error-exitcode 93 \
    "${debug_build}/cuda-tests/device_variational_test"
run_log sanitizer-integration-initcheck.log \
    "${sanitizer}" --tool initcheck --track-unused-memory --error-exitcode 94 \
    "${debug_build}/cuda-tests/device_scvx_integration_test" --sanitizer
run_log sanitizer-recovery-initcheck.log \
    "${sanitizer}" --tool initcheck --track-unused-memory --error-exitcode 97 \
    "${debug_build}/cuda-tests/recovery_test" --sanitizer
run_log sanitizer-production-initcheck.log \
    "${sanitizer}" --tool initcheck --track-unused-memory --error-exitcode 98 \
    "${debug_build}/cuda-tests/device_scvx_integration_test" \
    --production-outer-sanitizer

profile="${run}/device-scvx"
run_log nsys-profile.log \
    nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
    --force-overwrite=true --output="${profile}" \
    "${debug_build}/cuda-tests/device_scvx_integration_test"
run_log nsys-stats.log \
    nsys stats --force-export=true \
    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    "${profile}.nsys-rep"

run_log summary-command.log \
    .venv/bin/python scripts/gpu/summarize_g3_run.py "${run}"
printf '%q ' .venv/bin/python scripts/gpu/archive_run.py \
    "${run}" --repository . --require-clean-repository \
    --archive "${run}.tar.gz" >>"${run}/commands.txt"
printf '\n' >>"${run}/commands.txt"
.venv/bin/python scripts/gpu/archive_run.py \
    "${run}" \
    --repository . \
    --require-clean-repository \
    --archive "${run}.tar.gz" >"${run}.archive.log" 2>&1
sha256sum "${run}.tar.gz" >"${run}.tar.gz.sha256"
printf '%s\n' "${run}"
