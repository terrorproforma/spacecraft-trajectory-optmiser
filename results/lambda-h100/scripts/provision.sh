#!/bin/bash
# Lambda H100 toolkit-only provisioning (no driver, no `cuda`/`cuda-drivers` metapackage).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
log() { echo "[$(date -u +%FT%TZ)] $*"; }

log "== apt: base tools"
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
  build-essential ca-certificates curl wget git jq ninja-build pkg-config numactl \
  python3 python3-dev python3-venv rsync tmux xz-utils unzip zip gdb sysstat \
  libopenmpi-dev openmpi-bin 2>&1 | tail -3
log "mpicc: $(command -v mpicc || echo MISSING)  mpirun: $(mpirun --version 2>/dev/null | head -1)"
log "nccl: $(ldconfig -p | grep -c libnccl) entries; $(dpkg -l libnccl-dev | tail -1 | awk '{print $2, $3}')"

log "== apt: NVIDIA CUDA repo (toolkit-only: cuda-toolkit-12-8)"
if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
  cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && sudo dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null && sudo apt-get update -qq
fi
log "simulate cuda-toolkit-12-8 (checking for any driver package):"
sudo apt-get install -s cuda-toolkit-12-8 2>/dev/null | grep -E '^Inst ' | awk '{print $2}' > /tmp/cuda-toolkit-12-8.would-install.txt
wc -l /tmp/cuda-toolkit-12-8.would-install.txt
if grep -qiE 'nvidia-driver|nvidia-kernel|nvidia-dkms|cuda-drivers|nvidia-fabricmanager|libnvidia-(compute|gl|decode|encode|cfg|extra|fbc)' /tmp/cuda-toolkit-12-8.would-install.txt; then
  log "REFUSING: cuda-toolkit-12-8 would pull driver packages:"; grep -iE 'nvidia-driver|nvidia-kernel|nvidia-dkms|cuda-drivers|libnvidia' /tmp/cuda-toolkit-12-8.would-install.txt
else
  if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
    sudo apt-get install -y -qq --no-install-recommends cuda-toolkit-12-8 2>&1 | tail -3
  fi
  log "cuda-12.8: nvcc=$(/usr/local/cuda-12.8/bin/nvcc --version | tail -1)"
  log "nsys: $(ls /usr/local/cuda-12.8/bin/nsys 2>/dev/null || echo MISSING)  compute-sanitizer: $(ls /usr/local/cuda-12.8/bin/compute-sanitizer 2>/dev/null || echo MISSING)"
  ls -la /usr/local/cuda /usr/local/cuda-12 2>/dev/null
  ls /usr/local/cuda-12.8/lib64 | grep -E '^lib(cusparse|cublas|cusolver|cudart)\.so$'
fi

log "== uv + Python 3.12"
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.12 2>&1 | tail -1
uv python find 3.12

log "== node 20 LTS (user-local)"
if [ ! -x "$HOME/.local/node/bin/node" ]; then
  mkdir -p "$HOME/.local"
  cd /tmp
  fname=$(curl -s https://nodejs.org/dist/latest-v20.x/SHASUMS256.txt | grep -oE 'node-v20\.[0-9.]+-linux-x64\.tar\.xz' | head -1)
  curl -sO "https://nodejs.org/dist/latest-v20.x/$fname"
  mkdir -p "$HOME/.local/node" && tar -xJf "$fname" -C "$HOME/.local/node" --strip-components=1
fi
"$HOME/.local/node/bin/node" --version; "$HOME/.local/node/bin/npm" --version

log "== git identity (env-based, no global config)"
git config --global --get user.name || true

log "== shell env file"
cat > "$HOME/spacepdhcg/env.sh" <<'EOF'
# H100 environment for SpacePDHCG (CUDA 12.8 toolkit at /usr/local/cuda-12.8, driver already installed)
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="$HOME/.local/bin:$HOME/.local/node/bin:/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export SPACEPDHCG_CUDA_ARCHITECTURES=90
export CUDA_VISIBLE_DEVICES=0
EOF
log "== done provision"
