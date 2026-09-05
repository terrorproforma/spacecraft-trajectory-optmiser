#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-9e75b47-h100/preflight"
venv="/home/ubuntu/spacepdhcg/v1/.venv"
expected_commit="9e75b470fd5378d9b20f6f13892ac909c43757cd"
expected_branch="integration/single-gpu-v1"

test "$(git rev-parse HEAD)" = "${expected_commit}"
test "$(git branch --show-current)" = "${expected_branch}"
test -z "$(git status --porcelain=v1)"

{
    git status --short --branch
    git rev-parse HEAD
    git rev-parse "HEAD^{tree}"
    git branch --show-current
    git log -1 --format=fuller
} >"${out}/source.txt"

{
    uname -a
    cat /etc/os-release
    free -h
    df -h . /tmp
    ulimit -a
} >"${out}/system.txt"

{
    /usr/local/cuda-12.8/bin/nvcc --version
    nvidia-smi
    nvidia-smi \
        --query-gpu=index,name,uuid,compute_cap,driver_version,memory.total,memory.free,memory.used,temperature.gpu,power.draw \
        --format=csv,noheader
    nvidia-smi pmon -c 1
} >"${out}/gpu.txt" 2>&1

{
    printf 'pdhcg_commit='
    git -C _upstream/pdhcg rev-parse HEAD
    printf 'pdhcg_tree='
    git -C _upstream/pdhcg rev-parse "HEAD^{tree}"
    git -C _upstream/pdhcg status --short --branch
    if [[ -d _upstream/qoco-g4/.git ]]; then
        printf 'qoco_commit='
        git -C _upstream/qoco-g4 rev-parse HEAD
        printf 'qoco_tree='
        git -C _upstream/qoco-g4 rev-parse "HEAD^{tree}"
        git -C _upstream/qoco-g4 status --short --branch
    fi
    sha256sum \
        third_party/pdhcg.lock.json \
        third_party/patches/pdhcg/0001-free-quadratic-state.patch \
        third_party/qoco_gpu.lock.json \
        scripts/gpu/qoco_absolute_kkt_stopping.patch
} >"${out}/dependencies.txt"

{
    gcc --version
    g++ --version
    /usr/local/cuda-12.8/bin/nvcc --version
    ninja --version
    "${venv}/bin/python" --version
    "${venv}/bin/cmake" --version
    "${venv}/bin/ruff" --version
    "${venv}/bin/pytest" --version
    "${venv}/bin/python" -m pip show \
        nvidia-cudss-cu12 cupy-cuda12x torch jax jaxlib
    git --version
    nsys --version || true
    /usr/local/cuda-12.8/bin/compute-sanitizer --version
} >"${out}/toolchain.txt" 2>&1

python3 - "${out}/source-policy-schema.sha256" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

root = Path(".")
patterns = (
    "benchmarks/*.json",
    "benchmarks/campaign_scopes/*.json",
    "experiments/schema/*.json",
    "third_party/*.json",
    "scripts/gpu/*.patch",
)
paths = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
Path(sys.argv[1]).write_text(
    "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}\n"
        for path in paths
    ),
    encoding="utf-8",
)
PY

{
    printf 'preflight_complete_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'commit=%s\n' "$(git rev-parse HEAD)"
    printf 'tree=%s\n' "$(git rev-parse "HEAD^{tree}")"
    printf 'branch=%s\n' "$(git branch --show-current)"
    printf 'repository_clean=true\n'
    printf 'local_only=true\n'
    printf 'immutable_uri=unavailable\n'
} >"${out}/status.txt"
