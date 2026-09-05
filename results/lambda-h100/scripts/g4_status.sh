#!/bin/bash
# Record the G4 step state on the H100 host: capability regeneration pending the executor deadline fix.
mkdir -p "$HOME/g4"
cd "$HOME/spacepdhcg/v1" || exit 1
cat > "$HOME/g4/STATUS.txt" <<EOF
g4_state=capability regeneration pending fix
recorded_utc=$(date -u +%FT%TZ)
host=lambda-h100-80gb-hbm3
v1_head=$(git rev-parse HEAD) ($(git branch --show-current))
v1_tree_clean=$([ -z "$(git status --porcelain=v1)" ] && echo yes || echo no)
reseal=results/gpu/current-head-9e75b47-h100 (G0-G3 PASS; evidence-index $(cut -c1-16 results/gpu/current-head-9e75b47-h100/evidence-index.json.sha256))
blocker=adaptive PDHCG attempts ignore the attempt deadline (run to the 1,000,000-iteration cap); fix in development on integration/single-gpu-v1 in WSL; bundle expected at /home/angus/bundles/single-gpu-v1-<sha>.bundle
wsl_bundle_present_at_check=no (checked $(date -u +%FT%TZ); /home/angus/bundles/ absent; WSL worktree HEAD addac2b with uncommitted edits to persistent_pdhcg.cu, device_scvx.cu, cancellation_deadline_test.cu, test_g4_pdhcg_deadline_gpu.py)
capability_generated=no
checkpoint_initialised=no
worker_launched=no
foreign_gpu_processes_now=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | wc -l)
next_steps=1) scp the fix bundle to ~/bundles and 'git -C ~/spacepdhcg/v1 fetch <bundle> integration/single-gpu-v1 && git merge --ff-only FETCH_HEAD' 2) rebuild CUDA Release for sm_90 3) regenerate capability with the real --g4-session probe: IPM probe must create QOCO workspaces AND a PDHCG session probe with SPACEPDHCG_G4_ATTEMPT_DEADLINE_SECONDS=20 must terminate within the deadline 4) init a NEW claim-core checkpoint (hardware lambda-h100-80gb-hbm3; do not import RTX 5090 rows) with amendments v1.1+v1.2 5) launch worker + status/observer/finish scripts; verify zero foreign processes
EOF
cat "$HOME/g4/STATUS.txt"
