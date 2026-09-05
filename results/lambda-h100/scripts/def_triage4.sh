#!/bin/bash
cd "$HOME/spacepdhcg/v2" || exit 1
echo "== sweep"; tail -3 "$HOME/logs/v2_deferred.sh.log"; cat results/gpu/h100-deferred-3373988/status.txt
echo "== viewer_export file lists"; sed -n 20,40p src/spacepdhcg/planner/viewer_export.py
echo "== spacepdhcg_plan usage"; build-v2-cuda-release/cuda-tools/spacepdhcg_plan --help 2>&1 | head -30
echo "== native-request.json head"; head -c 600 build-v2-gpu-deferred/plan-powered_descent_3dof/native-request.json; echo
echo "== gdb?"; which gdb cuda-gdb
echo "== git log for check.mjs vs viewer_export"; git log --oneline -3 -- web/trajectory-viewer/scripts/check.mjs; git log --oneline -3 -- src/spacepdhcg/planner/viewer_export.py
