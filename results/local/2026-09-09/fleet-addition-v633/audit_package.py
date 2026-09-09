"""Portable stdlib saved-byte/hash/arithmetic audit; never execute archived code."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

A = 'build/performance/fleet-addition-v633/'
B = 'build/performance/fleet-addition-v633b/'
HOST = 'build/performance/mass-merit-return-v632/'
FINAL_SHA = '1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da'
FIRST_SHA = 'ebf7affef3b73cd883986d4db9a4d6dd0ffa4579c66e6a0f7b10b52cf7eb8796'
BASE_SHA = '765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(name, data):
    path = PurePosixPath(name)
    assert not path.is_absolute() and all(p not in ('.', '..', '') for p in path.parts)
    assert '\\' not in name and ':' not in name
    assert not any(p in ('.git', '__pycache__', '.pytest_cache', '.venv') for p in path.parts)
    assert not data.startswith((b'\x7fELF', b'MZ'))
    assert not re.search(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----', data)


def sections(data):
    parts = {}
    for line in data.splitlines(keepends=True):
        tokens = line.split()
        assert tokens
        parts.setdefault(int(tokens[0]), bytearray()).extend(line)
    return {k: bytes(v) for k, v in parts.items()}


def cargo(data):
    rows = [line.split() for line in data.splitlines() if int(line.split()[1]) != -1]
    assert len(rows) % 2 == 0
    found, events, used = {}, [], set()
    for before, after in zip(rows[::2], rows[1::2]):
        assert len(before) == len(after) == 10 and before[:3] == after[:3]
        body = int(before[1])
        gain = Decimal(after[9].decode()) - Decimal(before[9].decode())
        if body > 0:
            used.add(body)
            if gain > 0:
                assert body not in found
                found[body] = gain
        events.append((body, Decimal(before[2].decode())))
    return found, events, used


def audit(folder, expected_index=None):
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled; Python -O is not supported')
    folder = Path(folder)
    raw_index = (folder / 'index.json').read_bytes()
    if expected_index:
        assert sha(raw_index) == expected_index
    index = json.loads(raw_index)
    for name, record in index['top_files'].items():
        data = (folder / name).read_bytes()
        assert len(data) == record['bytes'] and sha(data) == record['sha256'], name
    archive = (folder / 'evidence.zip').read_bytes()
    assert sha(archive) == index['archive_sha256']
    with zipfile.ZipFile(folder / 'evidence.zip') as bundle:
        infos = bundle.infolist()
        assert len(infos) == len({m.filename for m in infos}) == index['file_count']
        assert set(index['files']) == {m.filename for m in infos}
        files = {}
        for entry in infos:
            assert not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
            assert not (entry.flag_bits & 1)
            data = bundle.read(entry)
            safe(entry.filename, data)
            pin = index['files'][entry.filename]
            assert len(data) == pin['bytes'] and sha(data) == pin['sha256'], entry.filename
            files[entry.filename] = data
    assert sum(len(v) for v in files.values()) == index['expanded_bytes']
    final = (folder / 'Result.txt').read_bytes()
    assert sha(final) == FINAL_SHA and not final.endswith((b'\r', b'\n'))
    assert sha(final + b'\r\n') == FIRST_SHA
    base = final[:9978319]
    assert sha(base) == BASE_SHA
    assert final == base + b'\n' + files[B + 'draft/renumbered-ship-24.txt']
    assert files[A + 'draft/renumbered-ship-24.txt'] == files[B + 'draft/renumbered-ship-24.txt'] + b'\r\n'
    source = files[A + 'draft/source-ship-09.txt']
    restored = b''.join(re.sub(rb'^([ \t]*)24(?=[ \t])', rb'\g<1>9', line, count=1)
                        for line in files[A + 'draft/renumbered-ship-24.txt'].splitlines(keepends=True))
    assert restored == source == files[B + 'draft/source-ship-09.txt']
    original_parts, final_parts = sections(base), sections(final)
    assert set(original_parts) == set(range(1, 24)) and set(final_parts) == set(range(1, 25))
    for ship, part in original_parts.items():
        assert final_parts[ship] == part + (b'\n' if ship == 23 else b'')
    get = lambda path: json.loads(files[path])
    original_plan = get(A + 'draft/plan.json')
    assert sha(source) == original_plan['source_ship_sha256']
    for ship, pin in original_plan['existing_section_sha256'].items():
        assert sha(original_parts[int(ship)]) == pin
    assert get(B + 'draft/plan.json')['only_final_two_CRLF_bytes_removed']
    assert files[A + 'output/independent-full.json'] == files[B + 'output/independent-full.json']
    first, last = get(A + 'output/report.json'), get(B + 'output/report.json')
    assert first['complete'] and first['independent']['ok'] and not first['official']['ok']
    assert not first['qualified'] and first['result_sha256'] == FIRST_SHA
    assert first['official']['message'] == 'ErrorA09. File Format Error on the line 100392: This line is empty!'
    assert last['complete'] and last['qualified'] and last['status'] == 'verified_24_ship_addition'
    assert last['result_sha256'] == FINAL_SHA
    for prefix, report, result_sha in ((A, first, FIRST_SHA), (B, last, FINAL_SHA)):
        launch = get(prefix + 'output/launch-report.json')
        assert launch['complete'] and launch['passed'] and launch['exit_code'] == 0
        assert launch['cleanup']['verified_empty'] and not launch['cleanup']['survivors']
        assert report['independent_calls'] == report['official_calls'] == 1
        assert all(report[k] == 0 for k in ('GPU_calls', 'native_solves', 'search_calls', 'extra_leg_certificates'))
        binding = get(prefix + 'output/checker-binding.json')
        assert binding['result_sha256'] == result_sha
        for key in ('independent', 'official'):
            assert binding[key] == report[key] == get(prefix + 'output/' + key + '.json')
        runtime = get(prefix + 'output/runtime.json')
        assert runtime['bonus_sha256'] == index['bonus_sha256']
        assert runtime['catalogue_sha256'] == index['catalogue_sha256']
        assert runtime['checker_plan_sha256'] == sha(files[prefix + 'checker-plan.json'])
    assert last['independent']['ok'] and last['official']['ok']
    assert last['independent']['ships'] == last['official']['ships'] == 24
    assert last['independent']['mined_asteroids'] == last['official']['mined_asteroids'] == 208
    assert not last['independent']['violations'] and last['independent']['ship_limit'] >= 24
    host_index = get(HOST + 'host-index.json')
    assert len(host_index) == 192
    for name, expected in host_index.items():
        assert sha(files[HOST + 'host/' + name]) == expected
    for prefix in (A, B):
        for name, expected in get(prefix + 'checker-plan.json')['source_sha256'].items():
            if name in files:
                assert sha(files[name]) == expected, name
            else:
                assert index['omitted_files'][name]['sha256'] == expected, name
    derived = {
        A + 'draft/Result.txt': final + b'\r\n', A + 'output/official/Result.txt': final + b'\r\n',
        B + 'draft/Result.txt': final, B + 'output/official/Result.txt': final,
        'results/local/2026-09-09/mass-budgeted-frontier-v629/Result.txt': base,
    }
    for name, data in derived.items():
        assert sha(data) == index['omitted_files'][name]['sha256']
    with localcontext() as ctx:
        ctx.prec = 65
        weights = {int(k): Decimal(v) for k, v in get('derived/used-bonus.json')['coefficient'].items()}
        all_cargo, all_used, events = {}, set(), 0
        base_cargo = {}
        for ship, part in final_parts.items():
            values, ship_events, used = cargo(part)
            assert not set(values) & set(all_cargo)
            if ship == 24:
                assert not used & all_used and len(used) == 9 and len(ship_events) == 20
                assert ship_events[0] == (0, Decimal(64508)) and ship_events[-1] == (-3, Decimal(69803))
            else:
                base_cargo.update(values)
            all_used.update(used)
            all_cargo.update(values)
            events += len(ship_events)
        assert len(all_used) == len(all_cargo) == len(weights) == 208
        raw = sum(all_cargo.values(), Decimal(0))
        weighted = sum((weights[k]*v for k, v in all_cargo.items()), Decimal(0))
        raw_threshold = Decimal(6000)*Decimal(12).ln()
        assert raw > raw_threshold
        independent = get(B + 'output/independent-full.json')
        for body, value in all_cargo.items():
            assert abs(Decimal(str(independent['mined'][str(body)]['collected_mass_kg'])) - value) < Decimal('1e-8')
        rows = files[B + 'output/official/ScoreData.txt'].split()
        assert int(rows[0]) == 208 and len(rows) == 417
        official_mass = {int(rows[i]): Decimal(rows[i+1].decode()) for i in range(1, len(rows), 2)}
        assert set(official_mass) == set(all_cargo)
        assert all(abs(official_mass[k]-v) < Decimal('1e-8') for k, v in all_cargo.items())
        assert abs(raw-Decimal(str(last['independent']['total_mass_kg']))) < Decimal('1e-8')
        assert abs(weighted-Decimal(str(last['independent']['weighted_score_fixed_bonus_kg']))) < Decimal('1e-8')
        base_raw = sum(base_cargo.values(), Decimal(0))
        base_weighted = sum((weights[k]*v for k, v in base_cargo.items()), Decimal(0))
        amounts = dict(raw_kg=str(raw), weighted_kg=str(weighted),
            raw_gain_kg=str(raw-base_raw), weighted_gain_kg=str(weighted-base_weighted),
            raw_24_ship_margin_kg=str(raw-raw_threshold))
    return dict(passed=True, archive_files=len(files), archive_expanded_bytes=index['expanded_bytes'],
        index_sha256=sha(raw_index), final_Result_sha256=FINAL_SHA,
        source_ship_identifier_only_plus_final_CRLF_removal=True, original_23_section_payloads_exact=True,
        original_independent_results_byte_identical=True, ships=24, asteroids=208, events=events,
        independent_fullfleet_calls=2, official_fullfleet_calls=2,
        complete_pair_worker_seconds=first['seconds']+last['seconds'],
        first_EOF_rejection_preserved=True, final_both_original_checkers_pass=True,
        saved_audit_GPU_calls=0, saved_audit_optimizer_calls=0, saved_audit_propagations=0,
        audited_frozen_host_files=192, amounts=amounts,
        external_dependencies=index['external_dependency_scope'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('package')
    parser.add_argument('--index-sha256')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = audit(args.package, args.index_sha256)
    if args.output:
        with Path(args.output).open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))
