#!/bin/bash
cd "$HOME/spacepdhcg/v2/results/gpu/h100-deferred-3373988/supplement" || exit 1
for f in pd3-memcheck-normalised-request pd3-memcheck-normalised-status cpu-plan-hcw_rendezvous-status cpu-plan-powered_descent_3dof-status cpu-plan-powered_descent_6dof-status cpu-plan-low_thrust-status; do
  echo "--- $f"; cat "$f.log"
done
echo "--- gdb"; grep -v '^\[\|^Thread\|^Using\|warning\|^Download\|^Debuginfod\|^This GDB\|^Enable' pd6-parity-magnitude-gdb.log | head -40
echo "--- cpu pytest"; grep -E '^FAILED|passed|failed|^E  ' cpu-planner-pytest.log | head -20
