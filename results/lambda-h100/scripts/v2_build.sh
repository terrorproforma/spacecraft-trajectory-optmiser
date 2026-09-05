#!/bin/bash
# Build the v2 candidate trees (CPU only; no GPU execution) pinned away from the reseal's cores.
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export V2="$HOME/spacepdhcg/v2"
cd "$V2"
export PATH="$V2/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""
[ -e .venv-v2 ] || ln -s .venv .venv-v2
log "v2 HEAD $(git rev-parse HEAD) dirty=$(git status --porcelain=v1 | wc -l)"
build() { # dir configure-args...
  local dir=$1; shift
  log "== configure $dir"
  cmake -S cpp -B "$dir" -G Ninja "$@" > "$HOME/logs/v2-$dir-configure.log" 2>&1 || { tail -20 "$HOME/logs/v2-$dir-configure.log"; return 1; }
  log "== build $dir"
  cmake --build "$dir" --parallel 12 > "$HOME/logs/v2-$dir-build.log" 2>&1 || { tail -30 "$HOME/logs/v2-$dir-build.log"; return 1; }
  log "== built $dir"
}
build build-v2-relwithdebinfo -DCMAKE_BUILD_TYPE=RelWithDebInfo -DSPACEPDHCG_BUILD_CUDA=OFF -DSPACEPDHCG_BUILD_C_API=ON -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON || exit 1
build build-v2-cuda-release -DCMAKE_BUILD_TYPE=Release -DSPACEPDHCG_BUILD_CUDA=ON -DSPACEPDHCG_BUILD_DISTRIBUTED=OFF -DSPACEPDHCG_BUILD_C_API=ON -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON -DCMAKE_CUDA_ARCHITECTURES=90 || exit 1
build build-v2-cuda-debug -DCMAKE_BUILD_TYPE=Debug -DSPACEPDHCG_BUILD_CUDA=ON -DSPACEPDHCG_BUILD_DISTRIBUTED=OFF -DSPACEPDHCG_BUILD_C_API=ON -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON -DCMAKE_CUDA_ARCHITECTURES=90 || exit 1
sha256sum build-v2-cuda-release/cuda/libspacepdhcg_cuda.so build-v2-cuda-release/cuda-tests/device_scvx_integration_test build-v2-cuda-release/cuda-tools/spacepdhcg_plan build-v2-cuda-release/cuda-tests/device_time_dilated_test build-v2-relwithdebinfo/libspacepdhcg.so
cuobjdump --list-elf build-v2-cuda-release/cuda/libspacepdhcg_cuda.so | head -3
log "== native ctest (CPU) on build-v2-relwithdebinfo"
ctest --test-dir build-v2-relwithdebinfo --output-on-failure --no-tests=error > "$HOME/logs/v2-native-ctest.log" 2>&1; echo "native ctest exit=$?"; tail -3 "$HOME/logs/v2-native-ctest.log"
log "== done v2 build"
