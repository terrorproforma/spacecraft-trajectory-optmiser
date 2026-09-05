#!/bin/bash
# Python 3.12 venvs (uv) for v1 and v2 mirroring the WSL tool venv's pinned GPU stack.
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export UV_LINK_MODE=copy
for wt in v1 v2; do
  root="$HOME/spacepdhcg/$wt"
  cd "$root"
  log "== $wt: venv"
  [ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
  uv pip install -q -p .venv/bin/python --upgrade pip build "cmake==4.4.3" "ninja==1.13.2" "ruff==0.16.5" "pytest==9.1.1" pytest-cov jsonschema 2>&1 | tail -2
  log "== $wt: pip install -e .[dev] (scikit-build-core native build, cmake from venv)"
  PATH="$root/.venv/bin:$PATH" .venv/bin/python -m pip install -q -e '.[dev]' 2>&1 | tail -3
  .venv/bin/python - <<'PY'
import spacepdhcg
from spacepdhcg.native import c_api_version, native_available, native_version
print("spacepdhcg", spacepdhcg.__version__, "native", native_available(), c_api_version(), native_version())
PY
  log "== $wt: GPU stack (cupy/torch/jax/cudss pinned to the WSL tool venv versions)"
  uv pip install -q -p .venv/bin/python "cupy-cuda12x==14.2.0" "nvidia-cudss-cu12==0.7.1.6" 2>&1 | tail -2
  uv pip install -q -p .venv/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
  uv pip install -q -p .venv/bin/python "jax[cuda12]==0.11.1" 2>&1 | tail -2
  .venv/bin/python -m pip list 2>/dev/null | grep -iE '^(cupy|torch|jax|jaxlib|jax-cuda12|nvidia-cudss|cmake|ninja|ruff|pytest|numpy|scipy|clarabel|osqp|matplotlib|jsonschema|scikit)' | tr -s ' ' | paste -sd, -
  .venv/bin/cmake --version | head -1
done
log "== done venvs"
