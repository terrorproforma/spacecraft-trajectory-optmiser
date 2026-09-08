"""CPU-only resource and runner checks for the frozen v622a handoff."""
from pathlib import Path
import ast
import hashlib
import json
import re

BASE = Path(__file__).resolve().parent
OUT = BASE / 'mass-core-v622a'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = BASE / 'completion-fullcore-v621/resource-usage.log'
new = OUT / 'resources.log'
patterns = [
    '24cooperative_solve_kernelILb0E', '24cooperative_solve_kernelILb1E',
    '12solve_kernelILb0E', '12solve_kernelILb1E',
    '26cooperative_halpern_kernel', '21cooperative_l1_kernelILb0E',
    '21cooperative_l1_kernelILb1E', '32cooperative_l1_initialise_kernel',
    '29cooperative_initialise_kernel', '27cooperative_residual_kernel',
    '15residual_kernel', '15recovery_kernel',
]


def resource(path, pattern):
    lines = path.read_text().splitlines()
    matches = [lines[i + 1].strip() for i, line in enumerate(lines)
               if line.strip().startswith('Function ') and pattern in line]
    assert len(matches) == 1, (path, pattern, matches)
    return {k: int(v) for k, v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)', matches[0])}


result = {'passed': False, 'gpu_calls': 0, 'old_resource_sha256': sha(old),
          'new_resource_sha256': sha(new), 'existing_kernels': {}}
for pattern in patterns:
    before, after = resource(old, pattern), resource(new, pattern)
    assert before == after, (pattern, before, after)
    result['existing_kernels'][pattern] = after
result['new_kernels'] = {pattern: resource(new, pattern) for pattern in
                         ['23cooperative_mass_kernel', '34cooperative_mass_initialise_kernel',
                          '13mass_validate', '12mass_prepare']}
runner = BASE / 'run_mass_tiny_v622.py'
ast.parse(runner.read_text())
compile(runner.read_text(), str(runner), 'exec')
result['runner_sha256'] = sha(runner)
result['runner_syntax_passed'] = True
result['manifest_sha256'] = sha(OUT / 'manifest.json')
result['passed'] = True
(OUT / 'handoff-cpu-checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
