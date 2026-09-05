#!/usr/bin/env bash
# Orchestrator for the main 8cb3759 G2/G3 reseal on the WSL RTX 5090.
# preflight -> G2 -> G3 -> summarize -> validate -> seal -> verify.
# Launch with: nice -n 5 setsid nohup bash run_all.sh > logs/run_all.nohup.log 2>&1 &
set -uo pipefail
export GIT_AUTHOR_NAME="SpacePDHCG-Integration"
export GIT_AUTHOR_EMAIL="integration@spacepdhcg.local"
export GIT_COMMITTER_NAME="${GIT_AUTHOR_NAME}"
export GIT_COMMITTER_EMAIL="${GIT_AUTHOR_EMAIL}"
root=/home/angus/worktrees/spacepdhcg-reseal-8cb3759
tool=/home/angus/spacecraft-trajectory-optmiser/.venv/bin
export PATH="${tool}:/usr/local/cuda-12.8/bin:${PATH}"
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
commit=8cb3759b29ea8c7d843322a940a7ebcabfd9ff21
ev="results/gpu/current-head-8cb3759-rtx5090"
logdir=/home/angus/reseal8cb/logs
export RESEAL_LOGDIR="${logdir}"
mkdir -p "${logdir}"
cd "${root}" || exit 1
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
START_FROM="${START_FROM:-}"
stamp() { date -u +%FT%TZ; }
t0=$(date +%s)
step() { # name command...
  local name=$1; shift
  if [[ -n "${START_FROM}" && "${name}" != "${START_FROM}" ]]; then
    echo "[$(stamp)] SKIP  ${name} (resuming from ${START_FROM})"; return 0
  fi
  START_FROM=""
  echo "[$(stamp)] START ${name}"
  echo "status=RUNNING step=${name} started=$(stamp)" > "${logdir}/RESULT"
  local s=$(date +%s)
  if "$@" > "${logdir}/${name}.log" 2>&1; then
    echo "[$(stamp)] PASS  ${name} ($(( $(date +%s) - s )) s)"
  else
    local rc=$?
    echo "[$(stamp)] FAIL  ${name} (exit ${rc}, $(( $(date +%s) - s )) s) -- see ${logdir}/${name}.log"
    tail -40 "${logdir}/${name}.log"
    echo "status=FAIL step=${name} exit=${rc} failed=$(stamp)" > "${logdir}/RESULT"
    exit 1
  fi
}
gpu_wait() { # wait for a free GPU before launching a gate; record any wait
  local step_name=$1 waited=0
  while :; do
    local wsl win
    wsl="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr -d '\r')"
    win="$(/mnt/c/Windows/System32/nvidia-smi.exe --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr -d '\r' | grep -i -E 'python|torch|cuda|jax|cupy|nvcc|ollama|blender|nsys|ncu')"
    if [[ -z "${wsl}" && -z "${win}" ]]; then
      echo "[$(stamp)] GPU clear before ${step_name} (waited ${waited} s)" | tee -a "${logdir}/orchestrator-gpu-waits.log"
      return 0
    fi
    if (( waited == 0 )); then
      echo "[$(stamp)] GPU BUSY before ${step_name}: wsl=[${wsl//$'\n'/;}] windows=[${win//$'\n'/;}] -- waiting" | tee -a "${logdir}/orchestrator-gpu-waits.log"
    fi
    sleep 30; waited=$((waited+30))
  done
}
echo "[$(stamp)] orchestrator pid $$ nice $(nice) host $(hostname) root ${root}"
echo "status=RUNNING step=init started=$(stamp)" > "${logdir}/RESULT"
test "$(git rev-parse HEAD)" = "${commit}" || { echo "HEAD is not ${commit}"; echo "status=FAIL step=init" > "${logdir}/RESULT"; exit 2; }
test -z "$(git status --porcelain=v1)" || { echo "dirty tree"; git status --short; echo "status=FAIL step=init" > "${logdir}/RESULT"; exit 2; }
echo "[$(stamp)] HEAD ${commit} on $(git branch --show-current), clean"
if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  step preflight-capture bash "${ev}/preflight/capture.sh"
  step preflight-build-qoco bash "${ev}/preflight/build_qoco.sh"
else
  echo "[$(stamp)] preflight skipped (status: $(head -1 "${ev}/preflight/status.txt"), qoco: $(cat "${ev}/preflight/qoco-build.status"))"
fi
gpu_wait g2
step g2 bash "${ev}/g2/run.sh"
gpu_wait g3
step g3 bash "${ev}/g3/run.sh"
step seals-summarize .venv-current-head/bin/python "${ev}/seals/summarize.py"
step seals-validate bash "${ev}/seals/validate.sh"
step seals-seal bash "${ev}/seals/seal.sh"
step seals-verify .venv-current-head/bin/python "${ev}/seals/verify_seals.py"
echo "status=PASS step=all completed=$(stamp) wall_seconds=$(( $(date +%s) - t0 ))" > "${logdir}/RESULT"
echo "[$(stamp)] RESEAL COMPLETE in $(( $(date +%s) - t0 )) s: ${ev}"
cat "${ev}/current-head-summary.json"
cat "${ev}/evidence-index.json.sha256"
cat "${ev}/seals/"*.tar.gz.sha256
