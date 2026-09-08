"""Check encoded vectors against neutral points and selected QOCO repeats; CPU only."""
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parent.parent
rows = []
for label in ('conditioning', 'difficult'):
    path = root / 'inputs' / (label + '-point.json')
    neutral = json.loads(path.read_text(), parse_int=float)
    encoded_path = root / 'inputs' / (label + '-initial.txt')
    lines = encoded_path.read_text().splitlines()
    encoded = {line.split()[0]: np.asarray(line.split()[2:], dtype=np.float64) for line in lines[3:]}
    qoco_path = root / 'inputs' / (label + '-qoco.log')
    original = [json.loads(line.split(' ', 1)[1], parse_int=float)
                for line in qoco_path.read_text().splitlines() if line.startswith('QP_REPLAY ')]
    selected = next(x for x in original if x['repeat'] == neutral['reference_repeat'])
    assert selected['status'] == neutral['reference_status'] and selected['status'] in (1, 2)
    vector_checks = {}
    for key in ('x', 'y', 'z', 's'):
        neutral_key = 'x_translated' if key == 'x' else key + '_original'
        a = encoded[key]
        b = np.asarray(neutral[neutral_key], dtype=np.float64)
        c = np.asarray(selected[key], dtype=np.float64)
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
        assert np.array_equal(a, c)
        vector_checks[key] = {'encoded_matches_neutral_bits': True, 'encoded_matches_selected_qoco_values': True,
                             'qoco_signed_zero_bit_differences': int(np.count_nonzero(a.view(np.uint64) != c.view(np.uint64)))}
    rows.append({'label': label, 'selected_qoco_repeat': int(selected['repeat']), 'selected_qoco_status': int(selected['status']),
                 'encoded_sha256': hashlib.sha256(encoded_path.read_bytes()).hexdigest(),
                 'neutral_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'qoco_log_sha256': hashlib.sha256(qoco_path.read_bytes()).hexdigest(), 'vectors': vector_checks})
with (root / 'audit/source-chain.json').open('x') as output:
    json.dump({'complete': True, 'gpu_calls': 0, 'points': rows}, output, indent=2)
print(json.dumps({'complete': True, 'source_chains': len(rows), 'signed_zero_differences': sum(v['qoco_signed_zero_bit_differences'] for row in rows for v in row['vectors'].values())}))
