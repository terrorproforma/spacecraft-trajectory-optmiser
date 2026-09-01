#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cmake_bin="${CMAKE_BIN:-cmake}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"
build_type="${BUILD_TYPE:-Debug}"
build_dir="${BUILD_DIR:-${repository}/build/g5-${build_type,,}}"
jobs="${BUILD_JOBS:-2}"

"${cmake_bin}" -S "${repository}/cpp" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-120}" \
  -DCMAKE_CUDA_COMPILER="${cuda_home}/bin/nvcc" \
  -DSPACEPDHCG_BUILD_CUDA=ON \
  -DSPACEPDHCG_BUILD_DISTRIBUTED=ON \
  -DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF \
  -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
  -DSPACEPDHCG_NATIVE_ENABLE_SANITIZERS="${ENABLE_HOST_SANITIZERS:-OFF}"

"${cmake_bin}" --build "${build_dir}" \
  --target g5_logical_rank_test g5_one_rank_runtime_test \
  --parallel "${jobs}"

"${cmake_bin}" --build "${build_dir}" --target g5_logical_checks

if [[ "${RUN_ONE_RANK_GPU:-0}" == "1" ]]; then
  active_compute="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
  )"
  if [[ -n "${active_compute//[[:space:]]/}" ]]; then
    printf 'G5 one-rank GPU correctness deferred: GPU is contended by PID(s): %s\n' \
      "${active_compute//$'\n'/, }"
  else
    "${cmake_bin}" --build "${build_dir}" --target g5_one_rank_checks
  fi
fi
