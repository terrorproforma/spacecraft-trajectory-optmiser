"""Freeze the terminal v622 packaging plan; no Git/native/GPU calls."""
from pathlib import Path
import hashlib
import json

from prepare_causal_mass_package_v622 import ROOT, PERF


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((ROOT / (PERF + 'v622-package-terminal-plan-b.json')).read_text())
    for item in plan['components']:
        if item['name'] == 'benchmark':
            item['anchor'] = 'output/benchmark-report.json'
            item['root_terminal_pin'] = {
                'sha256': 'ccd74d20bec17954b72e731bc05341ce93168e80c4876cefd336d5ee8591483a',
                'scope': 'Root terminal bounded benchmark, 80 calls and 27,040 evaluations',
            }
    additions = [
        ('compact_failed_source', 'completion-model-v622a-source-recovery', 'completion/build/a-source-recovery', 'report.json', None),
        ('benchmark_analysis', 'completion-pack-analysis-v622', 'completion/analysis', 'summary.json', '91fea368ede02b101f45f016a821a282479403e669a97638f14e785f73b859ea'),
        ('benchmark_review', 'completion-pack-review-v622', 'reviews/benchmark', 'results-findings.json', '66ea3ce0c11b06185116fc976be6f0ae7047f84f1518a0228c64e6d59d6daf1a'),
        ('combined_build', 'combined-core-v622a', 'combined/build/a', 'manifest.json', '9bff6c54af1edca6323502908d440305b4edc8447550572a5d93b50a421e3730'),
        ('combined_smoke', 'combined-smoke-v622a', 'combined/smoke', 'report.json', 'aae7e90de93a7a13887712dea85167c2e5f9d8c5ef93cf22f3642b9f47e73a44'),
        ('combined_review', 'combined-core-review-v622', 'reviews/combined', 'smoke-results-findings.json', 'ff83b223d1285bea3bd33efa9a8379d0eb135799cc052b41e45cc26e13fbaec8'),
    ]
    for name, source, destination, anchor, expected in additions:
        actual = sha(ROOT / PERF / source / anchor)
        assert expected is None or expected == actual, name
        plan['components'].append({
            'name': name, 'source': PERF + source, 'destination': destination,
            'anchor': anchor, 'root_terminal_pin': {'sha256': actual, 'scope': 'Terminal owned evidence or stable independent review'},
        })
    for version in 'abcd':
        source = PERF + 'v622-portable-check-' + version + '/portable-cpu-recheck'
        plan['components'].append({
            'name': 'portable_attempt_' + version, 'source': source,
            'destination': 'package-validation/prior-' + version,
            'anchor': 'report.json', 'root_terminal_pin': {'sha256': sha(ROOT / source / 'report.json'), 'scope': 'Preserved earlier portable CPU audit attempt; no new evaluation'},
        })
    plan['dependencies'] += [PERF + name for name in (
        'build_combined_core_v622a.py', 'prepare_combined_builder_v622.py',
        'run_combined_smoke_v622.py', 'recover_compact_attempt_a_v622.py',
        'prepare_causal_mass_final_v622.py',
    )]
    plan['files'] = [
        {'source': 'docs/' + name, 'destination': 'docs/' + name} for name in (
            'GPU_PERSISTENT_CAPTURE_REPLAY.md', 'GPU_ROUTE_COMPLETION.md', 'SOTA_EXECUTION_PLAN_2026-09-09.md')
    ]
    plan['files'].append({'source': PERF + 'v622-evidence-stage-b/ASSEMBLY_FAILURE.json',
                          'destination': 'package-validation/assembly-b-failure.json'})
    # Preserve the exact earlier wrapper revisions without their duplicated source trees.
    for version in ('a', 'c'):
        for name in ('prepare_causal_mass_package_v622.py', 'assemble_causal_mass_package_v622.py',
                     'portable_causal_mass_v622.py', 'recheck_causal_mass_v622.py'):
            source = PERF + 'v622-evidence-stage-' + version + '/reproduce/' + name
            if (ROOT / source).is_file():
                plan['files'].append({'source': source, 'destination': 'package-validation/prior-wrapper-' + version + '/' + name})
    plan['terminal_scope'] = 'All native/GPU runs are complete. Final assembly and saved-data portability checks only; no further source changes or experiments.'
    target = ROOT / PERF / 'v622-package-final-plan.json'
    with target.open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'plan': target.relative_to(ROOT).as_posix(), 'sha256': sha(target),
                      'components': len(plan['components']), 'files': len(plan['files'])}))


if __name__ == '__main__':
    main()
