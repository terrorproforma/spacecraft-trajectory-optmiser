#!/bin/bash
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log "== apt base tools (no distro MPI: Lambda ships OpenMPI 4.1.7 at /usr/mpi/gcc/openmpi-4.1.7rc1)"
sudo apt-get install -y -qq --no-install-recommends build-essential ca-certificates curl wget git jq ninja-build pkg-config numactl python3 python3-dev python3-venv rsync tmux xz-utils unzip zip gdb sysstat 2>&1 | tail -2
for t in ninja jq; do printf '%s=%s ' "$t" "$(command -v $t || echo MISSING)"; done; echo
cat >> "$HOME/spacepdhcg/env.sh" <<'EOF'
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7rc1
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
EOF
source "$HOME/spacepdhcg/env.sh"
log "nvcc: $(nvcc --version | tail -1); nsys: $(nsys --version 2>&1 | head -1); sanitizer: $(compute-sanitizer --version | head -1); mpicc: $(command -v mpicc)"

cd "$HOME/spacepdhcg"
log "== verify bundles"
mkdir -p /tmp/bv && git -C /tmp/bv init -q 2>/dev/null
for b in v1 v2 gtoc12 pdhcg qoco; do printf '%s: ' "$b"; git -C /tmp/bv bundle verify "$HOME/bundles/$b.bundle" 2>&1 | tail -1; done

clone_branch() { # name bundle branch expected_sha
  local name=$1 bundle=$2 branch=$3 expect=$4
  if [ ! -d "$name/.git" ]; then
    git clone -q --branch "$branch" "$HOME/bundles/$bundle" "$name" || { log "clone $name FAILED"; return 1; }
    git -C "$name" remote rename origin bundle
  fi
  local head; head=$(git -C "$name" rev-parse HEAD)
  local dirty; dirty=$(git -C "$name" status --porcelain=v1 | wc -l)
  log "$name: branch=$(git -C "$name" rev-parse --abbrev-ref HEAD) HEAD=$head expected=$expect match=$([ "$head" = "$expect" ] && echo yes || echo NO) dirty_lines=$dirty"
}
clone_branch v1 v1.bundle integration/single-gpu-v1 addac2b655d4e06dd17a2d2c2a7e0b1c354e720e
clone_branch v2 v2.bundle integration/single-gpu-v2-candidate 3373988057a251a4df11b9ac125583ed2a5ca65b
clone_branch gtoc12 gtoc12.bundle feat/gtoc12-asteroid-mining c495dc0f81955ba9d6b7a49ac922548005d2f08e
git -C v1 fsck --no-dangling 2>&1 | tail -1

log "== pinned upstream PDHCG from bundle into v1/_upstream/pdhcg and v2/_upstream/pdhcg (origin URL set to the lock's repository so checkout_pinned_pdhcg.sh accepts it)"
for wt in v1 v2; do
  dest="$HOME/spacepdhcg/$wt/_upstream/pdhcg"
  if [ ! -d "$dest/.git" ]; then
    mkdir -p "$(dirname "$dest")"
    git clone -q --no-checkout "$HOME/bundles/pdhcg.bundle" "$dest"
    git -C "$dest" remote set-url origin https://github.com/Lhongpei/PDHCG
    git -C "$dest" checkout -q --detach 167c8b72b4b96d2f94d405b8763e485514192b81
  fi
  echo "$wt pdhcg: $(git -C "$dest" rev-parse HEAD) tree=$(git -C "$dest" rev-parse 'HEAD^{tree}') dirty=$(git -C "$dest" status --porcelain=v1 | wc -l)"
done
log "== pinned QOCO from bundle into v1/_upstream/qoco-g4 (script path) (origin URL set to the lock's repository)"
for wt in v1 v2; do
  dest="$HOME/spacepdhcg/$wt/_upstream/qoco-g4"
  if [ ! -d "$dest/.git" ]; then
    git clone -q --no-checkout "$HOME/bundles/qoco.bundle" "$dest"
    git -C "$dest" remote set-url origin https://github.com/qoco-org/qoco.git
    git -C "$dest" checkout -q --detach 09f049597deef2a7ead15b3da19a9456ff7d4e53
  fi
  echo "$wt qoco: $(git -C "$dest" rev-parse HEAD) tree=$(git -C "$dest" rev-parse 'HEAD^{tree}') dirty=$(git -C "$dest" status --porcelain=v1 | wc -l)"
done
log "== lock expectations"; jq -r '.commit, .tree' v1/third_party/pdhcg.lock.json; jq -r '.commit, .tree' v1/third_party/qoco_gpu.lock.json
log "== done"
