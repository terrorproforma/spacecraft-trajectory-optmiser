#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/gpu/bootstrap_ubuntu.sh [options]

This script installs ordinary development tools, creates an isolated Python environment,
checks out the pinned PDHCG revision, builds the host-native SpacePDHCG package, and records
the machine environment. It deliberately does NOT install or replace the NVIDIA driver or
CUDA toolkit.

Options:
  --install-system       Install Ubuntu build packages with apt.
  --install-distributed  Also install OpenMPI; verify NCCL is already available.
  --venv PATH            Virtual environment path (default: .venv-gpu).
  --cuda PATH            CUDA root, e.g. /usr/local/cuda-12.6.
  --skip-python          Do not create/install the Python environment.
  --skip-native-build    Do not configure/build/test the host-native C++ tree.
  --skip-upstream        Do not check out the pinned PDHCG revision.
  --allow-no-gpu         Permit CPU-only preparation; environment verification will not require GPU.
  -h, --help             Show this help.
EOF
}

INSTALL_SYSTEM=0
INSTALL_DISTRIBUTED=0
SKIP_PYTHON=0
SKIP_NATIVE=0
SKIP_UPSTREAM=0
ALLOW_NO_GPU=0
VENV_PATH=".venv-gpu"
CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-system)
      INSTALL_SYSTEM=1
      shift
      ;;
    --install-distributed)
      INSTALL_DISTRIBUTED=1
      shift
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --cuda)
      CUDA_ROOT="$2"
      shift 2
      ;;
    --skip-python)
      SKIP_PYTHON=1
      shift
      ;;
    --skip-native-build)
      SKIP_NATIVE=1
      shift
      ;;
    --skip-upstream)
      SKIP_UPSTREAM=1
      shift
      ;;
    --allow-no-gpu)
      ALLOW_NO_GPU=1
      shift
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

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "bootstrap_ubuntu.sh supports Linux only" >&2
  exit 3
fi
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "Expected Ubuntu; detected ${ID:-unknown}. Use the manual roadmap instead." >&2
    exit 4
  fi
fi

if [[ ${INSTALL_SYSTEM} -eq 1 ]]; then
  sudo apt-get update
  PACKAGES=(
    build-essential
    ca-certificates
    cmake
    curl
    git
    jq
    ninja-build
    numactl
    pkg-config
    python3
    python3-dev
    python3-venv
  )
  if [[ ${INSTALL_DISTRIBUTED} -eq 1 ]]; then
    PACKAGES+=(libopenmpi-dev openmpi-bin)
  fi
  sudo apt-get install -y --no-install-recommends "${PACKAGES[@]}"
fi

if [[ -n "${CUDA_ROOT}" ]]; then
  CUDA_ROOT="$(realpath "${CUDA_ROOT}")"
  if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
    echo "CUDA root does not contain bin/nvcc: ${CUDA_ROOT}" >&2
    exit 5
  fi
  export CUDA_HOME="${CUDA_ROOT}"
  export CUDA_PATH="${CUDA_ROOT}"
  export CUDACXX="${CUDA_ROOT}/bin/nvcc"
  export PATH="${CUDA_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ ${ALLOW_NO_GPU} -eq 0 ]]; then
  command -v nvidia-smi >/dev/null
  command -v nvcc >/dev/null
  GPU_COUNT="$(nvidia-smi -L | grep -c '^GPU ' || true)"
  if [[ "${GPU_COUNT}" -lt 1 ]]; then
    echo "No NVIDIA GPU detected" >&2
    exit 6
  fi
fi

if [[ ${INSTALL_DISTRIBUTED} -eq 1 ]]; then
  command -v mpirun >/dev/null
  if ! ldconfig -p 2>/dev/null | grep -qi libnccl; then
    cat >&2 <<'EOF'
NCCL was not found. Install libnccl2 and libnccl-dev from NVIDIA's repository that matches
your CUDA/Ubuntu installation, then re-run this script. The script intentionally does not
add third-party apt repositories or guess a CUDA/NCCL combination.
EOF
    exit 7
  fi
fi

if [[ ${SKIP_PYTHON} -eq 0 ]]; then
  python3 -m venv "${VENV_PATH}"
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
  python -m pip install --upgrade pip build
  python -m pip install -e '.[dev]'
  python - <<'PY'
import spacepdhcg
from spacepdhcg.native import c_api_version, native_available, native_version

assert native_available()
assert c_api_version() == 1
assert native_version() == spacepdhcg.__version__
print(f"SpacePDHCG Python/native package: {spacepdhcg.__version__}")
PY
fi

if [[ ${SKIP_NATIVE} -eq 0 ]]; then
  cmake -S cpp -B build/native-host -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DSPACEPDHCG_BUILD_C_API=ON \
    -DSPACEPDHCG_BUILD_NATIVE_TESTS=ON \
    -DSPACEPDHCG_NATIVE_WARNINGS_AS_ERRORS=ON
  cmake --build build/native-host --parallel
  ctest --test-dir build/native-host --output-on-failure
fi

if [[ ${SKIP_UPSTREAM} -eq 0 ]]; then
  scripts/gpu/checkout_pinned_pdhcg.sh
fi

VERIFY_ARGS=(
  --repository "${REPOSITORY_ROOT}"
  --output "${REPOSITORY_ROOT}/results/gpu/environment.json"
)
if [[ ${ALLOW_NO_GPU} -eq 1 ]]; then
  VERIFY_ARGS+=(--allow-no-gpu)
fi
python3 scripts/gpu/verify_environment.py "${VERIFY_ARGS[@]}"

echo
echo "Bootstrap complete."
echo "Repository: ${REPOSITORY_ROOT}"
echo "Virtual environment: ${VENV_PATH}"
echo "Environment record: results/gpu/environment.json"
