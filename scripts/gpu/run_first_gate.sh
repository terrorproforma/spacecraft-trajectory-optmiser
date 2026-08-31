#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/gpu/run_first_gate.sh [options]

Run the first real, pinned, single-GPU PDHCG correctness gate and seal its evidence.
The repository must be clean and the current commit is recorded in every output directory.

Options:
  --cuda PATH        CUDA root, e.g. /usr/local/cuda-12.6.
  --venv PATH        Existing virtual environment (default: .venv-gpu).
  --gpu INDEX        CUDA_VISIBLE_DEVICES value (default: 0).
  --results PATH     Results root (default: results/gpu/first-gate).
  --skip-build       Reuse existing upstream C++/Python builds.
  --intervals LIST   Space-separated interval values (default: "8 32 128").
  --tolerance VALUE  Requested solver tolerance (default: 1e-6).
  -h, --help         Show this help.
EOF
}

CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-}}"
VENV_PATH=".venv-gpu"
GPU_INDEX="0"
RESULTS_ROOT="results/gpu/first-gate"
SKIP_BUILD=0
INTERVALS="8 32 128"
TOLERANCE="1e-6"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA_ROOT="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --gpu)
      GPU_INDEX="$2"
      shift 2
      ;;
    --results)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --intervals)
      INTERVALS="$2"
      shift 2
      ;;
    --tolerance)
      TOLERANCE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPOSITORY_ROOT}"
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "Refusing to benchmark a dirty repository" >&2
  git status --short >&2
  exit 3
fi

COMMIT="$(git rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="first-gate-${TIMESTAMP}-${COMMIT:0:12}"
RUN_DIRECTORY="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIRECTORY}"
exec > >(tee "${RUN_DIRECTORY}/driver.stdout.log") \
     2> >(tee "${RUN_DIRECTORY}/driver.stderr.log" >&2)

if [[ -n "${CUDA_ROOT}" ]]; then
  CUDA_ROOT="$(realpath "${CUDA_ROOT}")"
  export CUDA_HOME="${CUDA_ROOT}"
  export CUDA_PATH="${CUDA_ROOT}"
  export CUDACXX="${CUDA_ROOT}/bin/nvcc"
  export PATH="${CUDA_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
fi
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "Virtual environment not found: ${VENV_PATH}" >&2
  exit 4
fi
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

python scripts/gpu/verify_environment.py \
  --repository "${REPOSITORY_ROOT}" \
  --output "${RUN_DIRECTORY}/environment.json"
cp third_party/pdhcg.lock.json "${RUN_DIRECTORY}/pdhcg.lock.json"
git show --no-patch --format=fuller HEAD > "${RUN_DIRECTORY}/spacepdhcg-commit.txt"

scripts/gpu/checkout_pinned_pdhcg.sh "${REPOSITORY_ROOT}/_upstream/pdhcg" \
  | tee "${RUN_DIRECTORY}/pdhcg-checkout.txt"
PDHCG_COMMIT="$(git -C _upstream/pdhcg rev-parse HEAD)"

if [[ ${SKIP_BUILD} -eq 0 ]]; then
  cmake -S _upstream/pdhcg -B build/pdhcg-one-shot -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPDHCG_COMPILE_DISTRIBUTED=OFF
  cmake --build build/pdhcg-one-shot --clean-first --parallel
  test -x build/pdhcg-one-shot/pdhcg
  build/pdhcg-one-shot/pdhcg --help \
    > "${RUN_DIRECTORY}/pdhcg-cli-help.txt" 2>&1 || true

  export SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=${CUDACXX:-$(command -v nvcc)}"
  python -m pip install --no-deps --force-reinstall ./_upstream/pdhcg
fi

python - <<PY | tee "${RUN_DIRECTORY}/versions.json"
import json
import pdhcg
import spacepdhcg
from spacepdhcg.native import c_api_version, native_version

print(json.dumps({
    "spacepdhcg": spacepdhcg.__version__,
    "spacepdhcg_c_api": c_api_version(),
    "spacepdhcg_native": native_version(),
    "pdhcg": getattr(pdhcg, "__version__", None),
    "pdhcg_commit": "${PDHCG_COMMIT}",
}, indent=2, sort_keys=True))
PY

IFS=' ' read -r -a INTERVAL_VALUES <<< "${INTERVALS}"
for N in "${INTERVAL_VALUES[@]}"; do
  OUTPUT="${RUN_DIRECTORY}/banded-n${N}.json"
  echo "Running pinned PDHCG exact-optimum gate: intervals=${N}, tolerance=${TOLERANCE}"
  spacepdhcg-banded-correctness \
    --pdhcg \
    --seeds 17 29 41 53 71 \
    --intervals "${N}" \
    --tolerance "${TOLERANCE}" \
    > "${OUTPUT}"
  python -m json.tool "${OUTPUT}" > /dev/null
done

python - "${RUN_DIRECTORY}" "${COMMIT}" "${PDHCG_COMMIT}" "${TOLERANCE}" <<'PY'
import json
import pathlib
import sys

run_directory = pathlib.Path(sys.argv[1])
summary = {
    "schema_version": "1.0.0",
    "gate": "pinned one-shot PDHCG exact-optimum correctness",
    "spacepdhcg_commit": sys.argv[2],
    "pdhcg_commit": sys.argv[3],
    "tolerance": float(sys.argv[4]),
    "case_files": [],
}
for path in sorted(run_directory.glob("banded-n*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary["case_files"].append({
        "path": path.name,
        "cases": len(payload["cases"]),
        "pdhcg_version": payload.get("pdhcg_version"),
    })
(run_directory / "gate-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
PY

python scripts/gpu/archive_run.py \
  "${RUN_DIRECTORY}" \
  --repository "${REPOSITORY_ROOT}" \
  --require-clean-repository \
  --archive "${RUN_DIRECTORY}.tar.gz"

echo "First GPU correctness gate completed: ${RUN_DIRECTORY}"
