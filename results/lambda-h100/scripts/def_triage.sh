#!/bin/bash
cd "$HOME/spacepdhcg/v2" || exit 1
echo "== manifest doc"; sed -n 1,80p docs/GPU_DEFERRED_VALIDATION_V2.md
echo "== manifest json"; python3 -c "
import json; m=json.load(open('benchmarks/gpu_deferred_validation_v2.json'))
for k,v in m.items():
    if k!='items': print(k, json.dumps(v)[:300])
for it in m.get('items', []):
    print('-', json.dumps(it)[:400])
"
echo "== examples"; ls examples/planner/
echo "== maximum_tilt in pd3 example"; grep -n 'maximum_tilt' examples/planner/powered_descent_3dof.json src/spacepdhcg/planner/*.py 2>/dev/null | head
echo "== viewer export file list"; grep -rn 'gtoc12.js\|camera.js\|check.mjs' src/spacepdhcg --include=*.py | head
echo "== check.mjs reads"; sed -n 20,30p web/trajectory-viewer/scripts/check.mjs
echo "== qoco lock v1 vs v2"; for w in v1 v2; do (cd "$HOME/spacepdhcg/$w" && echo "$w: $(cat third_party/qoco/*.lock 2>/dev/null | head -3 | tr '\n' ' ')"; ls third_party/qoco 2>/dev/null | head); done
echo "== test_planner_gpu env/markers"; sed -n 1,60p tests/test_planner_gpu.py
echo "== device_time_dilated_test pd6_fft compare"; grep -n 'pd6_fft\|differ from CPU\|tolerance\|1e-' cpp/cuda/tests/device_time_dilated_test.cu | head -30
