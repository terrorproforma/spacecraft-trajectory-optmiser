#!/bin/bash
cd "$HOME/spacepdhcg/v2" || exit 1
out=results/gpu/h100-deferred-3373988
echo "== pytest short summary"; grep -n 'short test summary' -A 12 "$out/planner-gpu-pytest.log"
echo "== pytest assertion lines"; grep -n '^E  ' "$out/planner-gpu-pytest.log" | cut -c1-300 | head -30
echo "== device_time_dilated_test source 330-395"; sed -n 330,395p cpp/cuda/tests/device_time_dilated_test.cu
echo "== pd3 certificate"; python3 -c "
import json; d=json.load(open('build-v2-gpu-deferred/plan-powered_descent_3dof/plan-result.json'))
print(json.dumps(d['certificate'], indent=1)[:1800])
print('objective', d.get('objective'))
"
echo "== hcw certificate"; python3 -c "
import json; d=json.load(open('build-v2-gpu-deferred/plan-hcw_rendezvous/plan-result.json'))
print(json.dumps(d['certificate'], indent=1)[:1800])
"
echo "== sweep progress"; tail -4 "$HOME/logs/v2_deferred.sh.log"
