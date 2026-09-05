#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-8cb3759-rtx5090/preflight"
venv="/home/angus/spacecraft-trajectory-optmiser/.venv"
expected_commit="8cb3759b29ea8c7d843322a940a7ebcabfd9ff21"
expected_branch="chore/g2g3-reseal-8cb3759"

test "$(git rev-parse HEAD)" = "${expected_commit}"
test "$(git branch --show-current)" = "${expected_branch}"
test -z "$(git status --porcelain=v1)"

{
    git status --short --branch
    git rev-parse HEAD
    git rev-parse "HEAD^{tree}"
    git branch --show-current
    git log -1 --format=fuller
    printf 'main_ref=%s\n' "$(git rev-parse main)"
    printf 'branch_is_at_main=%s\n' "$([[ "$(git rev-parse HEAD)" == "$(git rev-parse main)" ]] && echo true || echo false)"
} >"${out}/source.txt"

{
    uname -a
    cat /etc/os-release
    free -h
    df -h . /tmp
    ulimit -a
    nproc
    uptime
    printf 'shared_host_note=GTOC12 v9 CPU workers (nice 19) run concurrently; evidence at nice 5, builds at nice 10, -j8\n'
    ps -eo pid,ni,pcpu,etime,comm --sort=-pcpu | head -12
} >"${out}/system.txt"

{
    /usr/local/cuda-12.8/bin/nvcc --version
    nvidia-smi
    nvidia-smi \
        --query-gpu=index,name,uuid,compute_cap,driver_version,memory.total,memory.free,memory.used,temperature.gpu,power.draw \
        --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    nvidia-smi pmon -c 1
    printf 'windows_nvidia_smi_compute_apps:\n'
    /mnt/c/Windows/System32/nvidia-smi.exe --query-compute-apps=pid,process_name,used_memory --format=csv | tr -d '\r'
} >"${out}/gpu.txt" 2>&1

{
    printf 'pdhcg_commit='
    git -C _upstream/pdhcg rev-parse HEAD
    printf 'pdhcg_tree='
    git -C _upstream/pdhcg rev-parse "HEAD^{tree}"
    git -C _upstream/pdhcg status --short --branch
    if [[ -d _upstream/qoco-current-head/.git ]]; then
        printf 'qoco_commit='
        git -C _upstream/qoco-current-head rev-parse HEAD
        printf 'qoco_tree='
        git -C _upstream/qoco-current-head rev-parse "HEAD^{tree}"
        git -C _upstream/qoco-current-head status --short --branch
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
    "${venv}/bin/ninja" --version
    "${venv}/bin/python" --version
    "${venv}/bin/cmake" --version
    "${root}/.venv-current-head/bin/python" --version
    "${root}/.venv-current-head/bin/python" -m pip list 2>/dev/null || "${root}/.venv-current-head/bin/python" -c 'import importlib.metadata as m; print(sorted((d.metadata["Name"], d.version) for d in m.distributions()))'
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
    printf 'scope=G2,G3\n'
    printf 'hardware_id=local-rtx-5090\n'
    printf 'cuda_architecture=120\n'
    printf 'local_only=true\n'
    printf 'immutable_uri=unavailable\n'
} >"${out}/status.txt"
