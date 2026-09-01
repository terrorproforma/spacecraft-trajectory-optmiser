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
        -DCMAKE_CUDA_ARCHITECTURES=120
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
run_expected_failure tight-final-residual.jsonl \
    "${debug_build}/cuda-tests/device_scvx_integration_test" --tight-pd3

sanitizer=/usr/local/cuda-12.8/bin/compute-sanitizer
for tool in memcheck racecheck synccheck; do
    run_log "sanitizer-variational-${tool}.log" \
        "${sanitizer}" --tool "${tool}" --error-exitcode 91 \
        "${debug_build}/cuda-tests/device_variational_test"
    run_log "sanitizer-integration-${tool}.log" \
        "${sanitizer}" --tool "${tool}" --error-exitcode 92 \
        "${debug_build}/cuda-tests/device_scvx_integration_test" --sanitizer
done
run_log sanitizer-variational-initcheck.log \
    "${sanitizer}" --tool initcheck --track-unused-memory --error-exitcode 93 \
    "${debug_build}/cuda-tests/device_variational_test"
run_log sanitizer-integration-initcheck.log \
    "${sanitizer}" --tool initcheck --track-unused-memory --error-exitcode 94 \
    "${debug_build}/cuda-tests/device_scvx_integration_test" --sanitizer

profile="${run}/device-scvx"
run_log nsys-profile.log \
    nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
    --force-overwrite=true --output="${profile}" \
    "${debug_build}/cuda-tests/device_scvx_integration_test"
run_log nsys-stats.log \
    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    "${profile}.nsys-rep"

cat >"${run}/summary.json" <<'EOF'
{
  "decision": "FAIL",
  "g4_authorized": false,
  "families_exercised": 4,
  "production_outer_loop_complete": false,
  "maximum_loose_repair_natural_residual": 0.00242428348,
  "tight_requested_residual": 1e-6,
  "tight_achieved_natural_residual": 0.000576465191,
  "tight_backend_termination": "iteration_limit",
  "tight_backend_iterations": 1000000,
  "tight_relative_primal_residual": 1.38560229e-8,
  "tight_relative_dual_residual": 7.55280608e-5,
  "topology_allocation_delta": 0,
  "topology_index_copy_delta": 0,
  "update_allocation_delta": 0,
  "hidden_cpu_fallback": false,
  "sanitizer_errors": 0,
  "nsys_cuda_api_records": true,
  "nsys_cuda_kernel_records": false,
  "nsys_cuda_memory_records": false,
  "h1_decision": "unresolved_not_qualified",
  "h1_scale_boundary": null,
  "h1_reason": "No matched-quality production SCvx outer loop because the final forcing residual gate failed."
}
EOF

(cd "${run}" && sha256sum * >evidence-index.sha256)
tar -czf "${run}.tar.gz" -C results/gpu/g3 "${run_id}"
sha256sum "${run}.tar.gz" >"${run}.tar.gz.sha256"
printf '%s\n' "${run}"
