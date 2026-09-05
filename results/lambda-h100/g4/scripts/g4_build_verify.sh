#!/usr/bin/env bash
# Stage 2 of the H100 G4 resume: pin v1 at 1dbcae0, fresh sm_90 CUDA Release + Debug configure/build
# (SPACEPDHCG_SOURCE_COMMIT is baked at configure time), full CUDA CTest on both, the 13-case GPU
# deadline matrix, and one ordinal-73 600 s reproduction. Everything is logged under ~/g4/fix-verification-1dbcae0.
set -uo pipefail
source /home/ubuntu/s/g4env-h100.sh
out=/home/ubuntu/g4/fix-verification-${head7}
mkdir -p "$out" "$g4logs"
statusf="$out/status.txt"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
: > "$out/items.tsv"
step() {
  local name="$1" expected="${2:-0}"; shift 2
  echo "[$(ts)] START $name"; echo "status=RUNNING step=$name" > "$statusf"
  { printf '%s\t' "$name"; printf '%q ' "$@"; printf '\n'; } >> "$out/commands.txt"
  local t0=$SECONDS
  "$@" > "$out/$name.log" 2>&1; local rc=$?
  local verdict=PASS; [ "$rc" -eq "$expected" ] || verdict=FAIL
  printf '%s\t%s\texit=%s\t%ss\n' "$name" "$verdict" "$rc" "$((SECONDS-t0))" >> "$out/items.tsv"
  echo "[$(ts)] $verdict  $name exit=$rc $((SECONDS-t0))s"
  [ "$verdict" = PASS ] || { echo "status=FAIL step=$name" > "$statusf"; tail -30 "$out/$name.log"; exit 1; }
}

# --- 1. pin the tree at 1dbcae0 -------------------------------------------------------------
test -z "$(git status --porcelain=v1)" || { echo "dirty tree, refuse"; exit 2; }
if [ "$(git rev-parse HEAD)" != "$head_sha" ]; then
  # 1dbcae0 sits on addac2b beside 9e75b47 (not a descendant), so integration/single-gpu-v1 cannot be
  # fast-forwarded without a non-ff branch move. Check the fix commit out on its own branch instead and
  # leave integration/single-gpu-v1 at 9e75b47 untouched.
  git rev-parse -q --verify "$head_sha^{commit}" >/dev/null || { echo "1dbcae0 not fetched"; exit 2; }
  git checkout -q -B "g4/h100-${head7}" "$head_sha" || exit 2
fi
echo "HEAD=$(git rev-parse HEAD) branch=$(git branch --show-current) dirty=$(git status --porcelain=v1 | wc -l)" | tee "$out/source.txt"
test "$(git rev-parse HEAD)" = "$head_sha" || exit 2
test -z "$(git status --porcelain=v1)" || exit 2
git ls-tree -r --name-only HEAD | grep -E 'cpp/cuda/tests/cancellation_deadline_test.cu|tests/test_g4_pdhcg_deadline_gpu.py' | tee -a "$out/source.txt"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo "GPU busy, refuse"; exit 3; }

# --- 2. fresh configure + build (Release, Debug) for sm_90 ------------------------------------
common=(-G Ninja -DSPACEPDHCG_BUILD_CUDA=ON -DSPACEPDHCG_BUILD_DISTRIBUTED=OFF -DSPACEPDHCG_BUILD_C_API=ON
        -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON -DCMAKE_CUDA_ARCHITECTURES=90)
rm -rf "$release" "$debug"
step release-configure 0 "$tool/cmake" -S cpp -B "$release" -DCMAKE_BUILD_TYPE=Release "${common[@]}"
step release-build 0 "$tool/cmake" --build "$release" --parallel 12
step debug-configure 0 "$tool/cmake" -S cpp -B "$debug" -DCMAKE_BUILD_TYPE=Debug "${common[@]}"
step debug-build 0 "$tool/cmake" --build "$debug" --parallel 12
grep -E 'SPACEPDHCG_SOURCE_COMMIT|CMAKE_CUDA_ARCHITECTURES' "$release/CMakeCache.txt" | tee "$out/release-cache.txt"
"$exe" --g4-capabilities | python3 -c 'import json,sys; d=json.load(sys.stdin); print("compiled_source_commit", d.get("compiled_source_commit"))' | tee -a "$out/source.txt"

# --- 3. full CUDA CTest on both trees ---------------------------------------------------------
step release-ctest 0 "$tool/ctest" --test-dir "$release" --output-on-failure --no-tests=error
step debug-ctest 0 "$tool/ctest" --test-dir "$debug" --output-on-failure --no-tests=error
grep -h -E 'tests passed|Total Test time' "$out/release-ctest.log" "$out/debug-ctest.log" | tee "$out/ctest-totals.txt"
grep -h -i 'cancellation_deadline' "$out/release-ctest.log" "$out/debug-ctest.log" | tee -a "$out/ctest-totals.txt"

# --- 4. the 13-case GPU deadline matrix on the Release executor -----------------------------
export SPACEPDHCG_G4_EXECUTOR="$exe"
step deadline-matrix-pytest 0 "$py" -m pytest -p no:cacheprovider -v -rA --durations=0 \
  --junitxml="$out/deadline-matrix-junit.xml" tests/test_g4_pdhcg_deadline_gpu.py
grep -E '^(PASSED|FAILED|ERROR|SKIPPED)|passed|failed' "$out/deadline-matrix-pytest.log" | tail -20

# --- 5. ordinal-73 600 s reproduction (twin stratum: 600 s / 1,000,000 cap) ------------------
export G4_ROOT="$root"
step ordinal73-600s-repro 0 "$py" /home/ubuntu/s/ordinal73_repro.py "$out/ordinal73-600s"

echo "status=PASS step=all completed_utc=$(ts)" > "$statusf"
echo "[$(ts)] BUILD+VERIFY COMPLETE"; cat "$out/items.tsv"
