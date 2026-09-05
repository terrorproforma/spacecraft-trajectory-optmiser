#!/bin/bash
set -euo pipefail
source "$HOME/spacepdhcg/env.sh"
root="$HOME/spacepdhcg/v1"
cd "$root"
test -z "$(git status --porcelain=v1)"
python3 "$HOME/s/patch_sm90.py" "$root"
git diff --stat
bash -n scripts/gpu/run_g2_evidence.sh && bash -n scripts/gpu/run_g3_evidence.sh && bash -n scripts/gpu/checkout_build_qoco_gpu.sh && echo bash-syntax-ok
.venv/bin/ruff check scripts/gpu/run_g3_h1.py && .venv/bin/ruff format --check scripts/gpu/run_g3_h1.py
git diff --check
export GIT_AUTHOR_NAME=SpacePDHCG-Integration GIT_AUTHOR_EMAIL=integration@spacepdhcg.local
export GIT_COMMITTER_NAME=SpacePDHCG-Integration GIT_COMMITTER_EMAIL=integration@spacepdhcg.local
git add scripts/gpu/run_g2_evidence.sh scripts/gpu/run_g3_evidence.sh scripts/gpu/checkout_build_qoco_gpu.sh scripts/gpu/run_g3_h1.py
git commit -q -F - <<'EOF'
build(gpu): make the evidence scripts target the local CUDA architecture (H100 sm_90)

The G2/G3 evidence scripts and the pinned QOCO build hard-coded
-DCMAKE_CUDA_ARCHITECTURES=120 (RTX 5090). They now read
SPACEPDHCG_CUDA_ARCHITECTURES (default unchanged: 120) and record the value in
environment.txt, so the same scripts reseal on an H100 (sm_90) without edits.
The QOCO build honours CUDACXX; run_g3_h1.py takes the Paper 1 hardware_id from
SPACEPDHCG_HARDWARE_ID (default unchanged: local-rtx-5090); nsys stats passes
--force-export=true so a stale .sqlite can never be reused.
EOF
git log -1 --format='%H %an <%ae> %s'
git status --porcelain=v1 | wc -l
