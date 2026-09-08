"""Attach CUDA-toolchain validation and quantified seed rescaling to v608 evidence."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

repo = Path(__file__).resolve().parents[2]
src = repo / 'build/performance/upstream-snapshot-v608'
dst = repo / 'results/local/2026-09-09/upstream-identical-capture-v608'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
index = json.loads((dst / 'sha256.json').read_text())
for name, expected in index.items():
    assert sha(dst / name) == expected, name
for folder in ('nvcc-check', 'nvcc-cmake-check'):
    target = dst / folder
    target.mkdir(exist_ok=False)
    for name in ('report.json', 'build.log', 'validate.log'):
        if (src / folder / name).exists():
            shutil.copyfile(src / folder / name, target / name)
shutil.copyfile(repo / 'build/performance/check_upstream_nvcc_v608.py', dst / 'check_nvcc.py')
changes = []
for name in ('conditioning', 'difficult'):
    point = (src / f'fixtures/{name}-initial.txt').read_text().splitlines()
    before = np.asarray(next(line for line in point if line.startswith('x ')).split()[2:], dtype=np.float64)
    prefix = 'UPSTREAM_REPLAY_RESULT '
    final = json.loads(next(line[len(prefix):] for line in (src / f'run/{name}-seeded.log').read_text().splitlines() if line.startswith(prefix)))
    after = np.asarray(final['x'], dtype=np.float64)
    changes.append({'capture': name, 'optimization_iterations': final['iterations'],
                    'changed_primal_bits': int(np.sum(before.view(np.uint64) != after.view(np.uint64))),
                    'maximum_primal_absolute_change': float(np.max(np.abs(before - after))),
                    'common_qualified': final['qualified']})
assert all(row['optimization_iterations'] == 0 and row['common_qualified'] for row in changes)
(dst / 'seed-roundtrip.json').write_text(json.dumps(changes, indent=2) + '\n')
readme = dst / 'README.md'
readme.write_text(readme.read_text() + """

The separate CUDA-toolchain compile and hidden-device input check also pass;
their logs are retained in `nvcc-cmake-check`. The initial `-Wpedantic` experiment
rejected nvcc-generated GCC line directives, not diagnostic C++ source; those
logs remain in `nvcc-check`. The source itself passed strict g++ warnings.
This compile validation did not add any GPU solver calls.

Iteration-zero upstream acceptance preserves accuracy but does not preserve
every primal bit: scaling/unscaling changes 1,756 and 2,354 entries by at most
4.440892098500626e-16. `seed-roundtrip.json` records that distinction. No
optimisation step was taken, and both common certificates remain qualified.
""", encoding='utf-8')
shutil.copyfile(__file__, dst / 'add_validation.py')
index = {str(p.relative_to(dst)).replace('\\', '/'): sha(p) for p in sorted(dst.rglob('*')) if p.is_file() and p.name != 'sha256.json'}
(dst / 'sha256.json').write_text(json.dumps(index, indent=2) + '\n')
print(json.dumps({'indexed_files': len(index), 'bytes': sum((dst / name).stat().st_size for name in index), 'index_sha256': sha(dst / 'sha256.json')}))
