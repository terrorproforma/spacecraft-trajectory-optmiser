#!/usr/bin/env bash
set -Eeuo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-9e75b47-h100/g0"
py="${root}/.venv/bin/python"
tool="/home/ubuntu/spacepdhcg/v1/.venv/bin"
site="$("${py}" -c 'import site; print(site.getsitepackages()[0])')"
gpu_site="/home/ubuntu/spacepdhcg/v1/.venv/lib/python3.12/site-packages"
qoco="${root}/build-current-head-qoco/libqoco.so"
cudss="${gpu_site}/nvidia/cu12"
shim="${root}/build-current-head-qoco-cudss-lib"
install="${root}/build-current-head-g0-install"

export PYTHONPATH="${root}/src:${site}"
export SPACEPDHCG_QOCO_LIBRARY="${qoco}"
export LD_LIBRARY_PATH="${shim}:${cudss}/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
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

rel="build-current-head-g0-relwithdebinfo"
run_log rel-configure "${tool}/cmake" -S cpp -B "${rel}" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="${install}" \
    -DSPACEPDHCG_BUILD_CUDA=OFF \
    -DSPACEPDHCG_BUILD_C_API=ON \
    -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
    -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON
run_log rel-build "${tool}/cmake" --build "${rel}" --clean-first --parallel 8
run_log rel-ctest "${tool}/ctest" --test-dir "${rel}" --output-on-failure --no-tests=error

native_rel="build-current-head-g0-native-relwithdebinfo"
run_log native-rel-configure "${tool}/cmake" -S cpp/native -B "${native_rel}" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DSPACEPDHCG_NATIVE_BUILD_TESTS=ON
run_log native-rel-build "${tool}/cmake" --build "${native_rel}" --clean-first --parallel 8
run_log native-rel-ctest "${tool}/ctest" --test-dir "${native_rel}" \
    --output-on-failure --no-tests=error

debug="build-current-head-g0-debug"
run_log debug-configure "${tool}/cmake" -S cpp -B "${debug}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSPACEPDHCG_BUILD_CUDA=OFF \
    -DSPACEPDHCG_BUILD_C_API=ON \
    -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
    -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON
run_log debug-build "${tool}/cmake" --build "${debug}" --clean-first --parallel 8
run_log debug-ctest "${tool}/ctest" --test-dir "${debug}" --output-on-failure --no-tests=error

native_debug="build-current-head-g0-native-debug"
run_log native-debug-configure "${tool}/cmake" -S cpp/native -B "${native_debug}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSPACEPDHCG_NATIVE_BUILD_TESTS=ON
run_log native-debug-build "${tool}/cmake" --build "${native_debug}" --clean-first --parallel 8
run_log native-debug-ctest "${tool}/ctest" --test-dir "${native_debug}" \
    --output-on-failure --no-tests=error

asan="build-current-head-g0-asan-ubsan"
run_log asan-configure "${tool}/cmake" -S cpp -B "${asan}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSPACEPDHCG_BUILD_CUDA=OFF \
    -DSPACEPDHCG_BUILD_C_API=ON \
    -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
    -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON \
    -DSPACEPDHCG_NATIVE_ENABLE_SANITIZERS=ON
run_log asan-build "${tool}/cmake" --build "${asan}" --clean-first --parallel 8
run_log asan-ctest env \
    ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "${tool}/ctest" --test-dir "${asan}" --output-on-failure --no-tests=error

native_asan="build-current-head-g0-native-asan-ubsan"
run_log native-asan-configure "${tool}/cmake" -S cpp/native -B "${native_asan}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSPACEPDHCG_NATIVE_BUILD_TESTS=ON \
    -DSPACEPDHCG_NATIVE_ENABLE_SANITIZERS=ON
run_log native-asan-build "${tool}/cmake" --build "${native_asan}" --clean-first --parallel 8
run_log native-asan-ctest env \
    ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    "${tool}/ctest" --test-dir "${native_asan}" --output-on-failure --no-tests=error

export SPACEPDHCG_NATIVE_LIBRARY="${root}/${rel}/libspacepdhcg.so"
run_log ruff-check "${tool}/ruff" check .
run_log ruff-format "${tool}/ruff" format --check .
run_log full-pytest "${py}" -S -m pytest -q

artifacts="${out}/artifacts"
mkdir -p "${artifacts}"
run_log package-build "${py}" -S -m build --wheel --sdist \
    --outdir "${artifacts}" .

wheel="$(printf '%s\n' "${artifacts}"/spacepdhcg-*.whl)"
sdist="$(printf '%s\n' "${artifacts}"/spacepdhcg-*.tar.gz)"
test -f "${wheel}"
test -f "${sdist}"

consumer="${root}/build-current-head-g0-wheel-consumer"
run_log consumer-venv /home/ubuntu/.local/bin/uv venv --python 3.12 --clear "${consumer}"
run_log consumer-install /home/ubuntu/.local/bin/uv pip install \
    --python "${consumer}/bin/python" "${wheel}"
run_log consumer-import "${consumer}/bin/python" -c \
    'import json, spacepdhcg; from spacepdhcg.native import c_api_version, native_available, native_version, packaged_library_path; assert native_available(); assert c_api_version() == 1; assert native_version() == spacepdhcg.__version__; print(json.dumps({"version": spacepdhcg.__version__, "c_api_version": c_api_version(), "native_version": native_version(), "library": str(packaged_library_path())}, sort_keys=True))'

run_log install-package "${tool}/cmake" --install "${rel}"
consumer_cmake="build-current-head-g0-cmake-consumer"
run_log cmake-consumer-configure "${tool}/cmake" -S cpp/package-smoke -B "${consumer_cmake}" \
    -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH="${install}"
run_log cmake-consumer-build "${tool}/cmake" --build "${consumer_cmake}" --clean-first --parallel 8
run_log cmake-consumer-ctest "${tool}/ctest" --test-dir "${consumer_cmake}" \
    --output-on-failure --no-tests=error

{
    sha256sum "${wheel}" "${sdist}"
    sha256sum \
        "${rel}/libspacepdhcg.so.0.1.0" \
        "${native_rel}/libspacepdhcg_native_core.so.0.1.0" \
        "${consumer_cmake}/spacepdhcg_package_consumer"
} >"${out}/artifact-sha256.txt"

{
    printf 'commit=%s\n' "$(git rev-parse HEAD)"
    printf 'tree=%s\n' "$(git rev-parse "HEAD^{tree}")"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'python_environment=%s\n' "$(readlink -f "${py}")"
    "${py}" --version
    "${tool}/cmake" --version
    gcc --version
    g++ --version
    ninja --version
} >"${out}/versions.txt" 2>&1

test -z "$(git status --porcelain=v1)"
printf 'status=PASS\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${out}/status.txt"
