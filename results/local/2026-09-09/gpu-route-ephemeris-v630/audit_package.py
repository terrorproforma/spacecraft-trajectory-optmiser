"""Standalone stdlib audit of hashes and SAVED state arrays; no native imports."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import statistics
import struct
import tarfile
import zipfile

B = 'build/performance/gpu-route-ephemeris-v630/'
A = 'build/performance/route-ephemeris-audit-v630/'
H = 'build/performance/route-ephemeris-h100-v630/'
MANIFEST = '7435b392a5faceff50db4361bab287fb3ce98b1b37aa93d2d5d633c0a9e68a14'
SOURCE = '9c63e12b55273b8789c8ac6f8a868a158f76d3145aedb24c0ea43cc084cbeec3'
REPORT = 'e4fecfe0a333a8ee52a313657a032fe6860d606bf41977eb05d03bc7f9d62d28'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(data):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(data, parse_constant=reject)


def safe_name(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in name
            and ':' not in name and name == str(path), 'Unsafe member: ' + name)
    require(not any(part in ('.git', '__pycache__', '.pytest_cache', '.ruff_cache')
                    for part in path.parts), 'Unexpected cache/Git member')


def scan(name, data, totals, depth=0):
    require(depth < 8, 'Archive nesting too deep')
    require(not data.startswith((b'\x7fELF', b'MZ')), 'Compiled binary: ' + name)
    require(not re.search(rb'-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----', data), 'Private key data')
    if name.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            seen = set()
            for member in archive:
                safe_name(member.name.rstrip('/'))
                require(member.name not in seen, 'Duplicate tar member')
                seen.add(member.name)
                require(member.isdir() or member.isfile(), 'Nonregular tar member')
                if member.isfile():
                    require(member.size <= 30_000_000, 'Excessive source member')
                    totals['nested_members'] += 1
                    scan(member.name, archive.extractfile(member).read(), totals, depth + 1)
            totals['nested_archives'] += 1
    elif name.endswith(('.zip', '.npz')):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            seen = set()
            for item in archive.infolist():
                safe_name(item.filename)
                require(item.filename not in seen, 'Duplicate ZIP member')
                seen.add(item.filename)
                kind = stat.S_IFMT(item.external_attr >> 16)
                require(kind in (0, stat.S_IFREG) and item.file_size <= 30_000_000,
                        'Nonregular or excessive ZIP member')
                totals['nested_members'] += 1
                scan(item.filename, archive.read(item), totals, depth + 1)
            totals['nested_archives'] += 1


def npy(data):
    require(data[:6] == b'\x93NUMPY', 'Invalid NPY')
    major = data[6]
    require(major in (1, 2), 'Unsupported NPY version')
    count = 2 if major == 1 else 4
    size = int.from_bytes(data[8:8+count], 'little')
    head = ast.literal_eval(data[8+count:8+count+size].decode('latin1'))
    require(not head['fortran_order'], 'Unexpected Fortran ordering')
    return head, data[8+count+size:]


def arrays(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name[:-4]: npy(archive.read(name)) for name in archive.namelist()}


def numeric(item, kind='<f8'):
    head, raw = item
    require(head['descr'] == kind, 'Unexpected array type')
    count = math.prod(head['shape'])
    require(len(raw) == count * 8, 'Unexpected array payload')
    values = struct.unpack('<' + ('d' if kind == '<f8' else 'q') * count, raw)
    require(all(math.isfinite(value) for value in values), 'Nonfinite saved state')
    return values, head['shape']


def rows(item):
    values, shape = numeric(item)
    require(len(shape) == 2 and shape[1] == 3, 'Expected vectors')
    return [values[i:i+3] for i in range(0, len(values), 3)]


def audit(package, expected_index=None):
    index_bytes = (package / 'index.json').read_bytes()
    if expected_index:
        require(digest(index_bytes) == expected_index, 'Index hash mismatch')
    index = decode(index_bytes)
    for name, item in index['top_files'].items():
        safe_name(name)
        blob = (package / name).read_bytes()
        require(len(blob) == item['bytes'] and digest(blob) == item['sha256'], 'Top-file mismatch')
    payload = (package / 'evidence.zip').read_bytes()
    require(digest(payload) == index['archive_sha256'], 'Archive hash mismatch')
    totals = {'nested_members': 0, 'nested_archives': 0}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate package member')
        require(set(archive.namelist()) == set(index['files']), 'Package names differ')
        data = {}
        for item in archive.infolist():
            name = item.filename
            safe_name(name)
            require(stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG), 'Package link')
            blob = archive.read(item)
            expected = index['files'][name]
            require(len(blob) == expected['bytes'] and digest(blob) == expected['sha256'], 'File mismatch: ' + name)
            scan(name, blob, totals)
            data[name] = blob
    require(len(data) == index['file_count'], 'File count mismatch')
    require(sum(map(len, data.values())) == index['bytes'], 'Byte count mismatch')
    require(digest(data[B + 'manifest.json']) == MANIFEST, 'Local manifest differs')
    manifest = decode(data[B + 'manifest.json'])
    require(manifest['complete'] and manifest['passed'] and manifest['GPU_calls'] == 0, 'Build status')
    for name in (B + 'source.tar.gz', A + 'inputs/native-source.tar.gz', H + 'source.tar.gz'):
        require(digest(data[name]) == SOURCE, 'Original source archive differs')
        with tarfile.open(fileobj=io.BytesIO(data[name]), mode='r:gz') as archive:
            found = {item.name: digest(archive.extractfile(item).read()) for item in archive if item.isfile()}
        require(found == manifest['sources'], 'Full original source map differs')
    for stage in manifest['stages']:
        require(digest(data[B + stage['name'] + '.log']) == stage['log_sha256'], 'Build log differs')
    gpu = decode(data[B + 'gpu-tests-a/report.json'])
    require(gpu['complete'] and gpu['passed'] and len(gpu['stages']) == 3, 'GPU tiny status')
    for stage in gpu['stages']:
        require(stage['exit_code'] == 0 and stage['cleanup']['verified_empty'] and not stage['compute_after'], 'GPU tiny cleanup')
        require(digest(data[B + 'gpu-tests-a/' + stage['name'] + '.log']) == stage['log_sha256'], 'GPU tiny log binding')
    require(b'ERROR SUMMARY: 0 errors' in data[B + 'gpu-tests-a/native-memcheck.log'], 'Memcheck status')
    ready = decode(data[A + 'ready.json'])
    for name, item in ready['files'].items():
        require(digest(data[A + name]) == item['sha256'], 'Prepared source/input binding')
    supervision = decode(data[A + 'execution-a/supervision.json'])
    require(supervision['passed'] and supervision['complete'] and supervision['cleanup']['verified_empty']
            and supervision['exit_code'] == 0 and not supervision['compute_after'], 'Route supervisor status')
    report_bytes = data[A + 'execution-a/worker/report.json']
    require(digest(report_bytes) == REPORT == supervision['worker_report_sha256'], 'Route report pin')
    report = decode(report_bytes)
    require(report['complete'] and report['passed'] and report['counts_match'], 'Route outcome')
    require(report['counts']['CUDA_batches'] == 20 and report['counts']['CUDA_state_requests'] == 410
            and report['counts']['CPU_body_calls'] == 720 and report['counts']['solver_calls'] == 0, 'Actual counts')
    for name, item in report['output_files'].items():
        require(digest(data[A + 'execution-a/worker/' + name]) == item['sha256'], 'Raw output binding')
    work = decode(data[A + 'inputs/workload.json'])
    routes = {route['id']: route for route in work['routes']}
    max_r, max_v, vector_restores, comparisons = 0.0, 0.0, 0, 0
    for pair in work['order']:
        route = routes[pair['route']]
        prefix = A + 'execution-a/worker/' + f"repeat-{pair['repeat']}-{route['id']}"
        cpu, cuda = arrays(data[prefix + '-cpu.npz']), arrays(data[prefix + '-cuda.npz'])
        cr, cv = rows(cpu['position']), rows(cpu['velocity'])
        gr, gv = rows(cuda['position']), rows(cuda['velocity'])
        ck = list(zip(numeric(cpu['body_ids'], '<i8')[0], numeric(cpu['epochs'])[0]))
        gk = list(zip(numeric(cuda['body_ids'], '<i8')[0], numeric(cuda['epochs'])[0]))
        require(ck == list(map(tuple, route['old_CPU_requests'])) and gk == list(map(tuple, route['CUDA_unique_requests'])), 'Saved request order')
        lookup = {key: i for i, key in enumerate(gk)}
        for i, key in enumerate(ck):
            j = lookup[key]
            dr = math.sqrt(math.fsum((a-b)**2 for a, b in zip(cr[i], gr[j])))
            dv = math.sqrt(math.fsum((a-b)**2 for a, b in zip(cv[i], gv[j])))
            require(dr < 1e-3 and dv < 1e-9, 'Saved physical-state parity')
            max_r, max_v = max(max_r, dr), max(max_v, dv)
        raw = arrays(data[prefix + '-cuda-raw.npz'])['results']
        require(raw[0]['shape'] == (len(gk),) and len(raw[1]) == 56 * len(gk), 'Raw result shape')
        for i, record in enumerate(struct.iter_unpack('<6d2i', raw[1])):
            require(record[6:] == (0, 0), 'Raw CUDA row status')
            require(struct.pack('<6d', *record[:6]) == struct.pack('<6d', *gr[i], *gv[i]), 'Raw CUDA/public vector bits')
        seed = arrays(data[A + 'inputs/' + route['first_seed']])
        seed_vectors = {'departure_position': numeric(seed['initial_state'])[0][:3],
            'departure_velocity': numeric(seed['departure_body_velocity'])[0],
            'arrival_position': numeric(seed['arrival_position'])[0], 'arrival_velocity': numeric(seed['arrival_velocity'])[0]}
        for r, v, keys in ((cr, cv, ck), (gr, gv, gk)):
            positions = {key: i for i, key in enumerate(keys)}
            a, b = positions[ck[0]], positions[ck[1]]
            generated = {'departure_position': r[a], 'departure_velocity': v[a],
                'arrival_position': r[b], 'arrival_velocity': v[b]}
            for name, values in generated.items():
                original = seed_vectors[name]
                require(digest(struct.pack('<3d', *original)) == route['archived_first_boundary'][name]['float64_sha256'], 'Exact archive representation')
                bound = 64 * 2.0**-52 * max(1.0, max(map(abs, values)))
                require(max(abs(x-y) for x, y in zip(original, values)) <= bound, 'Existing representation guard')
                vector_restores += 1
        comparisons += 1
    for route_id, summary in report['summary'].items():
        pair_records = [item for item in report['comparisons'] if item['route'] == route_id]
        cm = statistics.median(item['CPU_seconds'] for item in pair_records)
        gm = statistics.median(item['CUDA_seconds'] for item in pair_records)
        require(cm == summary['five_sample_CPU_median_seconds'] and gm == summary['five_sample_CUDA_median_seconds']
                and cm / gm == summary['ratio_of_five_sample_medians_CPU_over_CUDA'], 'Saved timing median')
    h100 = decode(data[H + 'upload-review-rejection.json'])
    require(not h100['uploaded'] and not h100['remote_build_started'] and not h100['GPU_tests_started'], 'H100 not-run distinction')
    return {'passed': True, 'file_count': len(data), 'bytes': index['bytes'], **totals,
        'local_source_members': len(manifest['sources']), 'saved_comparisons': comparisons,
        'archive_vector_restore_checks': vector_restores, 'max_saved_position_difference_km': max_r,
        'max_saved_velocity_difference_km_s': max_v, 'H100_not_run': True,
        'GPU_calls_by_this_auditor': 0, 'ephemeris_evaluations_by_this_auditor': 0,
        'solver_calls_by_this_auditor': 0, 'index_sha256': digest(index_bytes), 'archive_sha256': digest(payload)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    parser.add_argument('--index-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    findings = audit(args.package, args.index_sha256)
    text = json.dumps(findings, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text)


if __name__ == '__main__':
    main()
