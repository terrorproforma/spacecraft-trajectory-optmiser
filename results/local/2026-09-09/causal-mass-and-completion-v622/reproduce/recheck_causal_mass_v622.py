"""Run only sequential saved-evidence CPU audits in a fresh portable tree."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from portable_causal_mass_v622 import materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Use CPython 3.12: the preserved admission audit relies on its float sum semantics.')
    out, verification = materialize(args.package, args.out)
    perf = out / 'build/performance'
    logs = out / 'portable-cpu-recheck'
    logs.mkdir()
    jobs = [
        ('mass-source-fixture', perf / 'mass-native-review-v622/review.py'),
        ('mass-tiny', perf / 'mass-native-review-v622/tiny_review.py'),
        ('mass-original-KKT-metric', perf / 'mass-real-review-v622/review_real.py'),
        ('compact-frozen-revisions', perf / 'completion-model-review-v622/review.py'),
        ('compact-readbacks', perf / 'completion-model-review-v622/readback_review.py'),
        ('admission-identities', perf / 'refinement-admission-v622/audit.py'),
        ('benchmark-readbacks', perf / 'completion-pack-review-v622/review_results.py'),
        ('combined-readbacks', perf / 'combined-core-review-v622/review_smoke_results.py'),
    ]
    report = {'passed': False, 'GPU_calls': 0, 'native_library_loads': 0, 'Git_calls': 0,
              'verification': verification, 'commands': [],
              'recheck_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'portable_helper_sha256': hashlib.sha256((Path(__file__).parent / 'portable_causal_mass_v622.py').read_bytes()).hexdigest(),
              'python_version': sys.version,
              'scope': 'Saved arrays, original equations, identities and source assertions only; no model/solver/trajectory reruns.'}
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
    # The isolated child guard prohibits accidental native loads or nested process
    # execution. Review scripts operate on saved data with standard-library math.
    guard = logs / 'guard.py'
    guard.write_text(
        'import ctypes,runpy,subprocess,sys\n'
        'def blocked(*a,**k):raise RuntimeError("portable audit forbids native/process execution")\n'
        'ctypes.CDLL=ctypes.PyDLL=blocked\n'
        'subprocess.Popen=subprocess.run=subprocess.check_call=subprocess.check_output=blocked\n'
        'runpy.run_path(sys.argv[1],run_name="__main__")\n')
    for name, script in jobs:
        assert script.is_file(), name
        expected_admission = None
        if name == 'admission-identities':
            prior = script.parent / 'readback-audit.json'
            assert prior.resolve().is_relative_to(out)
            expected_admission = json.loads(prior.read_text())
            # The original auditor deliberately creates its result exclusively.
            # Preserve the materialized original separately; package bytes stay untouched.
            prior.rename(logs / 'admission-original-audit.json')
        started = time.perf_counter()
        command = [sys.executable, '-B', str(guard), str(script)]
        result = subprocess.run(command, cwd=out, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        (logs / (name + '.log')).write_bytes(result.stdout)
        report['commands'].append({'name': name, 'returncode': result.returncode,
                                   'seconds': time.perf_counter() - started,
                                   'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
                                   'log_sha256': hashlib.sha256(result.stdout).hexdigest()})
        (logs / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        assert result.returncode == 0, name + ': inspect preserved CPU log'
        if expected_admission is not None:
            actual = json.loads((script.parent / 'readback-audit.json').read_text())
            assert actual == expected_admission, 'Regenerated admission audit differs'
    report['passed'] = True
    (logs / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
