"""Read-only stdlib fixture/provenance checks; no project imports or native calls."""
from pathlib import Path
import ast
import hashlib
import json
import re
import struct

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
fixture = ROOT / 'build/performance/mass-native-oracle-v622/fixtures.json'
assert sha(fixture.read_bytes()) == '7312df3f38291175e76df66459da86c68fdb71cb8c1895a67b68d72790291169'
source = json.loads(fixture.read_text())
header_path = ROOT / 'cpp/cuda/tests/persistent_mass_fixture.hpp'
header = header_path.read_text()
expected = {key: source['original'][key] for key in ('A', 'F', 'c', 'scalar_upper', 'affine_offset')}
for prefix, key in (('initial', 'initial_point'), ('seed', 'qualified_near_zero_seed')):
    expected[prefix + '_x'] = source[key]['x']
    expected[prefix + '_y'] = source[key]['scalar_dual'] + source[key]['affine_dual']
expected['seed_c'] = source['qualified_near_zero_seed']['c']
expected['expected_x'] = [point['x'] for point in source['iterations']]
expected['expected_y'] = [point['scalar_dual'] + point['affine_dual'] for point in source['iterations']]
for key, origin in (('tau', 'tau'), ('sigma', 'sigma'), ('row_sums', 'row_abs_sums'), ('column_sums', 'column_abs_sums')):
    expected[key] = source['reduced'][origin]

def bits(value):
    if isinstance(value, list):
        return [bits(x) for x in value]
    return struct.pack('>d', float(value)).hex()

for name, values in expected.items():
    text = re.search(r'\b' + name + r'=(\{[^;]*\});', header).group(1)
    actual = ast.literal_eval(text.replace('{', '[').replace('}', ']'))
    assert bits(actual) == bits(values), name

paths = [
    'cpp/cuda/include/spacepdhcg/cuda/persistent_pdhcg_c_api.h',
    'cpp/cuda/include/spacepdhcg/cuda/mass_causal_arithmetic.hpp',
    'cpp/cuda/src/persistent_mass.cuh', 'cpp/cuda/src/persistent_mass_host.cuh',
    'cpp/cuda/src/persistent_pdhcg.cu', 'cpp/cuda/src/persistent_l1_host.cuh',
    'cpp/cuda/tests/persistent_mass_test.cu', 'cpp/cuda/tests/persistent_mass_fixture.hpp',
    'cpp/cuda/tests/persistent_mass_snapshot.hpp', 'cpp/cuda/tests/persistent_snapshot_replay.cu',
    'cpp/cuda/CMakeLists.txt', 'build/performance/build_mass_core_v622a.py',
    'build/performance/generate_mass_fixture_v622.py',
]
report = {
    'scope': 'Manual source/math/test review plus exact CPU fixture serialization check; no build, GPU or native loading.',
    'base_commit': 'fdf52ae31d259240ffebdcf79ed0965dce8b9298',
    'source_sha256': {name: sha((ROOT/name).read_bytes()) for name in paths},
    'oracle_fixture_sha256': sha(fixture.read_bytes()),
    'fixture_arrays_checked_bitwise': list(expected),
    'fixture_matches': True,
    'planned_solve_cases': [
        ['three_interval_oracle', 3, 3], ['disabled_to_L1', 1, 1],
        ['fresh_L1_control', 1, 1], ['original_seed_zero_step', 1, 0],
        ['cancel_before_initial', 1, 0], ['nonfinite_seed', 1, 0], ['metric_overflow', 1, 0],
    ],
    'planned_solve_APIs': 7, 'planned_iteration_caps': 9, 'expected_updates': 5,
    'actual_GPU_calls': 0,
    'decision': 'No provisional source/test blocker; GPU-hidden freeze/build cleared. Native launch requires final frozen build/resource review.',
}
(OUT/'findings.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'fixture_matches': True, 'arrays': len(expected), 'report_sha256': sha((OUT/'findings.json').read_bytes())}))
