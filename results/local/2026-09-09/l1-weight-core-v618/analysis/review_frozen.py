"""Read-only local archive/build/runner checks for the bounded v618b tiny run.

This does not execute the runner, compiler, tests, CUDA or external commands.
Linux binaries are rehashed by the reviewed root-controlled runner before launch;
this Windows-side check verifies their manifest bindings and compiled build logs.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import tarfile

BASE = Path('build/performance/l1-weight-v618b')
PRIOR = Path('build/performance/l1-weight-v618a')
RUNNER = Path('build/performance/run_l1_weight_tiny_v618.py')
PIN = {
    'manifest': '6ad3dfc586c457afbeaa0da13417f1208484f908fb22977fd086ea41a4f3a4c2',
    'source_tree_sha256': 'e476bbf3065173ed7b06fca49306537ffe1d47804c281bacd441e5a42104e1a9',
    'library_sha256': '1002c69e2418ab8b4ba1cdb7376f2959b8af455ae7ea4815db751508009fceec',
    'persistent_snapshot_replay_sha256': 'af5cb7bb7c9dff82038945a966952dbd15450f4c4c82a89c69b5c8b4fd78aaad',
    'persistent_l1_test_sha256': '2ed65a3ef07dc436e51d7ff6d5dd2e6a9d2aea6bf4401cdc6778f0deb067d8a6',
    'runner': '874a46be33fc063caecec1f2bb3c652f02659e9981e9a7d84e281643aa76098c',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict(data):
    def reject(value):
        raise ValueError(value)
    return json.loads(data, parse_constant=reject)


def archive(root):
    manifest = strict((root/'manifest.json').read_text())
    with tarfile.open(root/'source.tar.gz', 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        assert len({m.name for m in members}) == len(members)
        data = {m.name: tar.extractfile(m).read() for m in members}
    assert {k: sha(v) for k, v in data.items()} == manifest['source_sha256']
    tree = ''.join(k+':'+v+'\n' for k, v in manifest['source_sha256'].items())
    assert sha(tree.encode()) == manifest['source_tree_sha256']
    return manifest, data


def main():
    assert sha((BASE/'manifest.json').read_bytes()) == PIN['manifest']
    assert sha(RUNNER.read_bytes()) == PIN['runner']
    m, files = archive(BASE)
    prior, previous_files = archive(PRIOR)
    for key in ('source_tree_sha256', 'library_sha256', 'persistent_snapshot_replay_sha256', 'persistent_l1_test_sha256'):
        assert m[key] == PIN[key]
    assert m['complete'] and m['gpu_calls'] == m['git_mutations'] == 0
    assert m['source_identity_kind'] == 'sha256_tree_not_git_commit'
    assert m['frozen_commit'] is None and m['compiled_source_commit'] == 'uncommitted'
    assert m['base_commit'] == '32efbed13eae47d2c8352771e884a2ad5ff5d07e'
    assert m['base_manifest_sha256'] == sha(Path('build/performance/l1-v615c/manifest.json').read_bytes())
    assert files.keys() == previous_files.keys()
    changed = [k for k in files if files[k] != previous_files[k]]
    assert changed == ['cpp/cuda/CMakeLists.txt']
    assert m['library_sha256'] == prior['library_sha256']
    for name in m['owned_paths']:
        assert Path(name).read_bytes() == files[name]
    for stage in m['stages']:
        assert stage['returncode'] == (1 if stage['name'].startswith('reject-') else 0)
    records = 0
    for path in BASE.glob('validate-*.log'):
        for line in path.read_text().splitlines():
            tag, payload = line.split(' ', 1)
            obj = strict(payload)
            records += 1
            if tag == 'PERSISTENT_REPLAY_META':
                assert obj['source_commit'] == 'uncommitted' and obj['source_dirty'] is True
                assert obj['source_tree_sha256'] == m['source_tree_sha256'] and obj['base_commit'] == m['base_commit']
                assert obj['source_commit_scope'] == 'uncommitted_frozen_source_tree'
    assert len(list(BASE.glob('validate-*.log'))) == 12
    parsed = ast.parse(RUNNER.read_text())
    case_modes = next(ast.literal_eval(node.value) for node in parsed.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'expected_case_modes' for t in node.targets))
    assert len(case_modes['unit']) == 9 and len(case_modes['weighted']) == 13
    assert case_modes['weighted'][1:3] == [('weight_switch_unit', 1), ('fresh_unit_l1_control', 0)]
    test_source = files['cpp/cuda/tests/persistent_l1_test.cu'].decode()
    for sequence in case_modes.values():
        assert len(set(sequence)) == len(sequence)
        for name, _ in sequence:
            assert '"'+name+'"' in test_source
    expected_resources = {
        'cooperative_solve_kernelILb0': (80, 0, 7168),
        'cooperative_solve_kernelILb1': (96, 0, 7168),
        '12solve_kernelILb0': (148, 40, 1032),
        '12solve_kernelILb1': (204, 40, 1032),
        'cooperative_halpern_kernel': (96, 0, 7168),
        'cooperative_l1_kernelILb0': (94, 0, 7168),
        'cooperative_l1_kernelILb1': (94, 0, 7168),
        'cooperative_l1_initialise_kernel': (58, 0, 5120),
    }
    lines = (BASE/'resource-usage.log').read_text().splitlines()
    for name, expected in expected_resources.items():
        matches = [lines[i+1] for i, line in enumerate(lines) if line.startswith(' Function ') and name in line]
        assert len(matches) == 1
        found = re.search(r'REG:(\d+) STACK:(\d+) SHARED:(\d+)', matches[0])
        assert tuple(map(int, found.groups())) == expected
    result = dict(scope='CPU-only frozen source/build/log review; no binary or GPU execution',
                  script_sha256=sha(Path(__file__).read_bytes()), pins=PIN,
                  verified_archive_members_each=len(files), owned_live_files_matched=len(m['owned_paths']),
                  a_to_b_changed=changed, same_core=True, stages_verified=len(m['stages']),
                  hidden_variants=12, parsed_json_records=records,
                  resources={k: dict(registers=v[0], stack=v[1], shared=v[2]) for k, v in expected_resources.items()},
                  exact_case_modes=case_modes,
                  expected_budget=dict(processes=2, solve_calls=22, requested_updates=31, expected_actual_updates=22),
                  binary_check_scope='Manifest/build-log bindings here; reviewed runner rehashes Linux core and test immediately before launch.',
                  review_correction='An initial duplicate-label concern referred to provisional source. Frozen b already has distinct weighted and cancel-global weak-seed names; concern retracted before launch.',
                  disposition='Clear for root-controlled bounded tiny run; no real-capture qualification or speed claim.')
    output = Path(__file__).with_name('frozen-findings.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output.read_bytes()))))


if __name__ == '__main__':
    main()
