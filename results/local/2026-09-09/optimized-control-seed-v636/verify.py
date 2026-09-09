"""Portable saved-array/input/status audit; no project import or propagation."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

READY = '9dcec8241fb2c8f9f4cc5e2a1b0d0b39bfe27a2a9b33d081838856cfd16f9ad9'
RESULT = '1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def npz(value):
    out = {}
    with zipfile.ZipFile(io.BytesIO(value)) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        for name in names:
            assert '/' not in name and '\\' not in name and name.endswith('.npy')
            data = archive.read(name)
            assert data[:6] == b'\x93NUMPY' and data[6] in (1, 2)
            width = 2 if data[6] == 1 else 4
            n = int.from_bytes(data[8:8 + width], 'little')
            start = 8 + width + n
            header = ast.literal_eval(data[8 + width:start].decode())
            assert header['descr'] == '<f8' and not header['fortran_order']
            size = 1
            for dim in header['shape']:
                size *= dim
            assert len(data[start:]) == size * 8
            out[name[:-4]] = (header['shape'], data[start:])
    return out


def audit(read, names):
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    def load(name):
        return json.loads(read(name))
    assert sha(read('ready.json')) == READY
    ready = load('ready.json')
    for name, digest in ready['files'].items():
        assert sha(read(name)) == digest, name
    report, launch = load('output/report.json'), load('output/launch-report.json')
    assert report['complete'] and report['status'] == 'optimized_control_return_uncertified'
    assert launch['complete'] and launch['passed'] and launch['ready_sha256'] == READY
    assert launch['exit_code'] == launch['child_exit_after_cleanup'] == 0
    assert launch['owned_descendant_cleanup']['verified_empty']
    assert not launch['owned_descendant_cleanup']['survivors'] and not launch['failures']
    assert not launch['compute_processes_after_cleanup']
    assert report['native_solves_started'] == report['native_solves_returned'] == 1
    assert report['native_iterations'] == 27 and report['seeded_calls_started'] == 1
    zero = ('flight_certificate_calls', 'wait_certificate_calls', 'cold_calls_started',
            'full_fleet_CPU_checks', 'full_fleet_official_checks', 'standalone_inspection_calls',
            'whole_route_reruns', 'Lambert_calls', 'fresh_prefix_propagations',
            'ephemeris_batches', 'ephemeris_state_requests', 'CPU_boundary_ephemeris_calls')
    assert all(report[k] == 0 for k in zero)
    assert not report['incumbent_promoted'] and not report['mass_scaling_requested']
    assert not any(n.startswith(('output/candidate-fleet/', 'output/optimized_control_return/propagation/')) for n in names)
    binding, profile = load('fleet-binding.json'), load('profile.json')
    assert binding['result_sha256'] == RESULT and binding['ships'] == 24 and binding['asteroids'] == 208
    core = load('output/core-manifest.json')
    assert sha(read('output/core-manifest.json')) == profile['native_manifest']['sha256']
    assert core['outputs']['library']['sha256'] == profile['library']['sha256']
    assert core['source_archive_sha256'] == launch['core_source_archive_validation']['sha256']
    directory = 'output/optimized_control_return/'
    raw = load(directory + 'legs/00/native-raw.json')
    old = load('inputs/prior-native-result.json')
    init = load(directory + 'initialization.json')
    review = load('input-review.json')
    assert raw['status'] == 'infeasible' and raw['seed_backend'] == 'cuda_zoh_replay'
    assert raw['iterations'] == len(raw['history']) == len(raw['solver_reports']) == 27
    assert raw['accepted_iterations'] == sum(bool(x['accepted']) for x in raw['history']) == 3
    assert [x['iteration'] for x in raw['history'] if x['accepted']] == [1, 2, 3]
    assert raw['max_defect'] > load('inputs/settings.json')['defect_tolerance']
    assert raw['virtual_inf'] > load('inputs/settings.json')['defect_tolerance']
    assert raw['outer_transfer_bytes']['trajectory_upload_bytes'] == (7 + 226 * 3) * 8
    assert init['target_initial_mass_kg'] is None and init['mass_ratio'] == 1
    assert not init['Python_thrust_scaling_or_clipping'] and not init['GPU_mass_scaling_requested']
    assert init['old_state_nodes_uploaded'] == init['old_vinf_outputs_uploaded'] == 0
    source = npz(read('inputs/optimized-native-raw.npz'))
    upload = npz(read(directory + 'seed-upload.npz'))
    assert set(upload) == {'initial_state', 'thrust_n', 'node_epochs_mjd'}
    assert upload['node_epochs_mjd'] == source['epochs']
    assert upload['thrust_n'] == source['thrust_n']
    assert sha(upload['thrust_n'][1]) == review['thrust_payload_sha256'] == init['source_thrust_payload_sha256']
    boundary = load('inputs/failed-boundary.json')
    initial = boundary['departure_position'] + boundary['departure_velocity'] + [boundary['initial_mass']]
    assert upload['initial_state'] == ((7,), struct.pack('<7d', *initial))
    assert sha(upload['initial_state'][1]) == review['canonical_initial_payload_sha256']
    assert load(directory + 'boundary.json')['effective'] == boundary
    emitted_plan, input_plan = load(directory + 'prescribed-plan.json'), load('inputs/candidate-plan.json')
    # Frozen RoutePlan.from_summary rebuilds this redundant list from deploy-map
    # insertion order. Actual flight order, epochs, cargo and every other field stay exact.
    assert emitted_plan.keys() == input_plan.keys()
    assert all(emitted_plan[k] == input_plan[k] for k in input_plan if k != 'asteroids')
    assert len(emitted_plan['asteroids']) == len(set(emitted_plan['asteroids']))
    assert set(emitted_plan['asteroids']) == set(input_plan['asteroids'])
    assert emitted_plan['asteroids'] == [int(k) for k in input_plan['deploy_epochs']]
    assert not report['cases'][0]['certified']
    current_arrays = npz(read(directory + 'legs/00/native-raw.npz'))
    post_arrays = npz(read(directory + 'legs/00/post-clamp.npz'))
    assert current_arrays.keys() == post_arrays.keys()
    raw_to_post = {key: current_arrays[key] == post_arrays[key] for key in current_arrays}
    assert raw_to_post['epochs'] and raw_to_post['states_scaled']
    return {
        'passed': True, 'scope': 'saved-only exact input arrays, source pins and accounted terminal status',
        'prepared_files_rehashed': len(ready['files']), 'ready_sha256': READY,
        'source_Result_sha256': RESULT, 'report_sha256': sha(read('output/report.json')),
        'launch_sha256': sha(read('output/launch-report.json')),
        'library_sha256': profile['library']['sha256'],
        'saved_native_calls': 1, 'saved_native_updates': 27, 'saved_accepted_updates': 3,
        'saved_qualified_conic_reports': sum(x['qualified'] for x in raw['solver_reports']),
        'saved_return_certificates': 0, 'saved_fleet_checker_calls': 0,
        'source_thrust_and_epochs_uploaded_bit_exact': True, 'canonical_initial_state_uploaded_bit_exact': True,
        'actual_flight_order_epochs_cargo_and_other_plan_fields_exact': True,
        'redundant_asteroid_list_reordered_by_existing_from_summary': emitted_plan['asteroids'] != input_plan['asteroids'],
        'old_state_nodes_uploaded': 0, 'mass_scaling_requested': False,
        'native_status': raw['status'], 'diagnostic': raw['diagnostic'],
        'final_max_defect': raw['max_defect'], 'final_virtual_inf': raw['virtual_inf'],
        'prior_max_defect': old['max_defect'], 'max_defect_difference': raw['max_defect'] - old['max_defect'],
        'defect_multiple_of_original_gate': raw['max_defect'] / load('inputs/settings.json')['defect_tolerance'],
        'final_node_mass_kg': raw['final_mass_kg'], 'node_mass_is_certified': False,
        'raw_to_post_clamp_array_identity': raw_to_post,
        'worker_seconds': report['worker_seconds'], 'native_python_bridge_seconds': raw['solve_seconds'],
        'same_failed_interval_or_axis_proved_by_this_run': False,
        'infeasibility_proved': False, 'incumbent_unchanged': True,
        'new_GPU_calls': 0, 'new_optimizer_calls': 0, 'new_propagations': 0,
        'raw_output_sha256': {n: sha(read(n)) for n in sorted(names) if n.startswith('output/')},
    }


def main():
    import stat
    from pathlib import PurePosixPath
    parser = argparse.ArgumentParser(description="Portable saved v636 package audit; no native execution")
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--index-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    raw_index = (args.package / 'index.json').read_bytes()
    if args.index_sha256:
        assert sha(raw_index) == args.index_sha256
    index = json.loads(raw_index)
    for name, item in index['top_files'].items():
        assert '/' not in name and chr(92) not in name and name not in ('.', '..')
        value = (args.package / name).read_bytes()
        assert len(value) == item['bytes'] and sha(value) == item['sha256']
    assert sha((args.package / 'evidence.zip').read_bytes()) == index['archive_sha256']
    saved = {}
    with zipfile.ZipFile(args.package / 'evidence.zip') as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == index['file_count']
        assert set(names) == set(index['files'])
        for member in archive.infolist():
            name = member.filename
            path = PurePosixPath(name)
            assert not path.is_absolute() and all(p not in ('', '.', '..') for p in path.parts)
            assert chr(92) not in name and ':' not in name and not member.is_dir()
            assert not stat.S_ISLNK(member.external_attr >> 16)
            value = archive.read(member)
            assert not value.startswith((bytes([127])+b'ELF', b'MZ'))
            assert b'-----BEGIN PRIVATE KEY-----' not in value
            assert b'-----BEGIN OPENSSH PRIVATE KEY-----' not in value
            item = index['files'][name]
            assert len(value) == item['bytes'] and sha(value) == item['sha256']
            saved[name] = value
    assert sum(map(len, saved.values())) == index['expanded_bytes']
    assert saved['RESULTS.md'] == (args.package / 'README.md').read_bytes()
    result = audit(saved.__getitem__, saved.keys())
    assert result == json.loads(saved['saved-audit.json'])
    summary = {'passed': True, 'index_sha256': sha(raw_index),
               'archive_sha256': index['archive_sha256'], 'file_count': len(saved),
               'expanded_bytes': index['expanded_bytes'], 'actual_seed_inputs_exact': True,
               'saved_native_calls': 1, 'saved_updates': 27, 'saved_accepted': 3,
               'saved_certificates': 0, 'saved_fleet_checks': 0,
               'trajectory_qualified': False, 'incumbent_unchanged': True,
               'new_GPU_solver_propagation_calls': 0, 'archived_code_executed': False}
    text = json.dumps(summary, indent=2, sort_keys=True) + chr(10)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
