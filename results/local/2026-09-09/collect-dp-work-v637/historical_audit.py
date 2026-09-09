"""Count existing DP scheduling/storage work; no solver or GPU execution."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'cpp/cuda/src/gtoc12_collect_dp.cu'
EXPECTED = 'e0122e96b3a39c0e48bdc49c377ed01fae1848dbe745096b1aeb51647ad5a3bc'
OUT = ROOT / 'build/performance/collect-dp-work-audit-v637.json'

def require(ok, why):
    if not ok:
        raise ValueError(why)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

require(sha(SOURCE) == EXPECTED, 'reviewed implementation changed')
require(not OUT.exists(), 'fresh output required')
rows = []
for k in range(1, 17):
    dense_states = k * (1 << k)
    active_by_layer = [math.comb(k, c) * (k-c) for c in range(k)]
    active = sum(active_by_layer)
    require(active == k * (1 << (k-1)), 'active state identity')
    exhaustive = k <= 10
    if exhaustive:
        # Independent definition versus a candidate compact index. This checks
        # integer topology only, not a changed numerical implementation.
        expected_states = {(s, j) for s in range(1 << k) for j in range(k) if not (s >> j & 1)}
        mapped = set()
        for mask in range(1 << (k-1)):
            for j in range(k):
                low = (1 << j) - 1
                subset = (mask & low) | ((mask & ~low) << 1)
                mapped.add((subset, j))
        require(mapped == expected_states and len(mapped) == active, 'compact bijection')
    rows.append(dict(k=k, dense_states=dense_states, usable_states=active,
                     dense_transition_coordinates_per_epoch=k*dense_states,
                     active_transition_coordinates_per_epoch=active-1,
                     active_layer_coordinates_including_skipped_camp=active_by_layer,
                     nominal_transition_coordinate_reduction=2*k,
                     storage_reduction_for_state_arrays=2,
                     compact_bijection_exhaustively_checked=exhaustive))

profile_path = ROOT / 'results/lambda/2026-09-09/gpu-shared-return-options-v846/local/profile/summary.json'
profile = json.loads(profile_path.read_text())
hotspots = {}
for ship, value in profile.items():
    hotspots[ship] = [row for row in value['top_cumulative']
                      if row['file'].endswith('/gpu_collect_dp.py')
                      and row['function'] in ('__init__', 'solve')]
report = dict(passed=True, source_sha256=EXPECTED, audit_source_sha256=sha(Path(__file__)),
              scope='Static scheduling/storage arithmetic and previously recorded Python call profile',
              solver_calls=0, gpu_calls=0, model_evaluations=0, rows=rows,
              profile_sha256=sha(profile_path), saved_profile_hotspots=hotspots,
              interpretation=[
                  'The dense transition grid is launched once for every cardinality; most coordinates immediately return.',
                  'Only states (subset,j) with j absent from subset are usable; all initial storage remains dense.',
                  'At cardinality zero the camp destination is additionally skipped, hence active-1.',
                  'Counts exclude CUDA block padding and do not measure instruction cost or time.',
                  'Python native-call timings include synchronized native execution and readback; they are not CPU arithmetic attribution.',
                  'Proposed compact scheduling preserves predecessor and epsilon tie order; numerical parity and end-to-end speed must be tested.',
                  'No production source changed and no speedup is claimed.'
              ])
with OUT.open('x', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
print(json.dumps(dict(passed=True, output=str(OUT), output_sha256=sha(OUT), k10=rows[9], solver_calls=0, gpu_calls=0)))
