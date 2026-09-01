#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SOURCE="${ROOT}/_upstream/qoco-g4"
BUILD="${ROOT}/build/qoco-g4"
VENV="${ROOT}/.venv"
COMMIT="09f049597deef2a7ead15b3da19a9456ff7d4e53"
TREE="c85fe82f71a67921868fc761c242de11ac46f4a2"
CUDSS_VERSION="0.7.1.6"
CUDSS_ROOT="${VENV}/lib/python3.12/site-packages/nvidia/cu12"
SHIM="${ROOT}/build/qoco-cudss-lib"

if [[ ! -x "${VENV}/bin/python" || ! -x "${VENV}/bin/cmake" ]]; then
  echo "G4 QOCO build requires the repository .venv with Python and CMake" >&2
  exit 2
fi

if [[ ! -d "${SOURCE}/.git" ]]; then
  git clone --filter=blob:none https://github.com/qoco-org/qoco.git "${SOURCE}"
fi
git -C "${SOURCE}" fetch --quiet origin "${COMMIT}"
git -C "${SOURCE}" checkout --quiet --detach "${COMMIT}"
git -C "${SOURCE}" submodule update --init --recursive

actual_commit="$(git -C "${SOURCE}" rev-parse HEAD)"
actual_tree="$(git -C "${SOURCE}" rev-parse 'HEAD^{tree}')"
dirty="$(git -C "${SOURCE}" status --porcelain=v1)"
[[ "${actual_commit}" == "${COMMIT}" ]] || {
  echo "QOCO commit mismatch: ${actual_commit}" >&2
  exit 3
}
[[ "${actual_tree}" == "${TREE}" ]] || {
  echo "QOCO tree mismatch: ${actual_tree}" >&2
  exit 4
}
[[ -z "${dirty}" ]] || {
  echo "QOCO checkout is dirty" >&2
  exit 5
}

installed_cudss="$("${VENV}/bin/python" -c \
  'from importlib.metadata import version; print(version("nvidia-cudss-cu12"))' \
  2>/dev/null || true)"
if [[ "${installed_cudss}" != "${CUDSS_VERSION}" ]]; then
  echo "Install nvidia-cudss-cu12==${CUDSS_VERSION} in .venv before building" >&2
  exit 6
fi
if [[ ! -f "${CUDSS_ROOT}/include/cudss.h" \
      || ! -f "${CUDSS_ROOT}/lib/libcudss.so.0" ]]; then
  echo "Pinned cuDSS headers or runtime library are missing" >&2
  exit 7
fi

mkdir -p "${SHIM}"
ln -sfn "${CUDSS_ROOT}/lib/libcudss.so.0" "${SHIM}/libcudss.so"

"${VENV}/bin/cmake" -S "${SOURCE}" -B "${BUILD}" -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_LIBRARY_PATH="${SHIM}" \
  -DCMAKE_C_FLAGS="-I${CUDSS_ROOT}/include" \
  -DCMAKE_CUDA_FLAGS="-I${CUDSS_ROOT}/include" \
  -DQOCO_ALGEBRA_BACKEND=cuda \
  -DQOCO_BUILD_TYPE=Release \
  -DQOCO_SINGLE_PRECISION=OFF \
  -DENABLE_TESTING=OFF \
  -DBUILD_QOCO_DEMO=ON
"${VENV}/bin/cmake" --build "${BUILD}" --parallel

export LD_LIBRARY_PATH="${SHIM}:${CUDSS_ROOT}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
"${BUILD}/qoco_demo"
