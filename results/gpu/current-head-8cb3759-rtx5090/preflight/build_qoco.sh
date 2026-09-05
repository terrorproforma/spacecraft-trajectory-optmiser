#!/usr/bin/env bash
set -Eeuo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-8cb3759-rtx5090/preflight"
venv="/home/angus/spacecraft-trajectory-optmiser/.venv"
source_dir="${root}/_upstream/qoco-current-head"
build_dir="${root}/build-current-head-qoco"
commit="09f049597deef2a7ead15b3da19a9456ff7d4e53"
tree="c85fe82f71a67921868fc761c242de11ac46f4a2"
cudss_root="${venv}/lib/python3.12/site-packages/nvidia/cu12"
shim="${root}/build-current-head-qoco-cudss-lib"
patch="${root}/scripts/gpu/qoco_absolute_kkt_stopping.patch"

# Shared host: builds run at absolute nice 10 with at most 8 jobs.
current_nice="$(nice)"
build_nice_delta=$((10 - current_nice))
if ((build_nice_delta < 0)); then build_nice_delta=0; fi

trap 'status=$?; printf "status=FAIL\nexit_code=%d\n" "${status}" >"${out}/qoco-build.status"; exit "${status}"' ERR

test -z "$(git status --porcelain=v1)"
if [[ ! -d "${source_dir}/.git" ]]; then
    git clone --filter=blob:none https://github.com/qoco-org/qoco.git "${source_dir}"
fi
git -C "${source_dir}" fetch --quiet origin "${commit}"
git -C "${source_dir}" checkout --quiet --detach "${commit}"
git -C "${source_dir}" submodule update --init --recursive
test "$(git -C "${source_dir}" rev-parse HEAD)" = "${commit}"
test "$(git -C "${source_dir}" rev-parse "HEAD^{tree}")" = "${tree}"

if git -C "${source_dir}" apply --reverse --check "${patch}" 2>/dev/null; then
    :
elif [[ -z "$(git -C "${source_dir}" status --porcelain=v1)" ]]; then
    git -C "${source_dir}" apply --check "${patch}"
    git -C "${source_dir}" apply "${patch}"
else
    echo "QOCO checkout has undeclared changes" >&2
    exit 5
fi

test "$("${venv}/bin/python" -c \
    'from importlib.metadata import version; print(version("nvidia-cudss-cu12"))')" = "0.7.1.6"
test -f "${cudss_root}/include/cudss.h"
test -f "${cudss_root}/lib/libcudss.so.0"
mkdir -p "${shim}"
ln -sfn "${cudss_root}/lib/libcudss.so.0" "${shim}/libcudss.so"

"${venv}/bin/cmake" -S "${source_dir}" -B "${build_dir}" -G Ninja \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_LIBRARY_PATH="${shim}" \
    -DCMAKE_C_FLAGS="-I${cudss_root}/include" \
    -DCMAKE_CUDA_FLAGS="-I${cudss_root}/include" \
    -DQOCO_ALGEBRA_BACKEND=cuda \
    -DQOCO_BUILD_TYPE=Release \
    -DQOCO_SINGLE_PRECISION=OFF \
    -DENABLE_TESTING=OFF \
    -DBUILD_QOCO_DEMO=ON
nice -n "${build_nice_delta}" "${venv}/bin/cmake" --build "${build_dir}" --clean-first --parallel 8

{
    printf 'qoco_commit=%s\n' "$(git -C "${source_dir}" rev-parse HEAD)"
    printf 'qoco_tree=%s\n' "$(git -C "${source_dir}" rev-parse "HEAD^{tree}")"
    printf 'qoco_status_porcelain=%q\n' "$(git -C "${source_dir}" status --porcelain=v1)"
    printf 'declared_patch_sha256='
    sha256sum "${patch}"
    printf 'library_sha256='
    sha256sum "${build_dir}/libqoco.so"
    printf 'demo_sha256='
    sha256sum "${build_dir}/qoco_demo"
    printf 'cudss_sha256='
    sha256sum "${cudss_root}/lib/libcudss.so.0"
    printf 'build_nice=%d parallel=8\n' "$((current_nice + build_nice_delta))"
} >"${out}/qoco-build-manifest.txt"
printf 'status=PASS\n' >"${out}/qoco-build.status"
