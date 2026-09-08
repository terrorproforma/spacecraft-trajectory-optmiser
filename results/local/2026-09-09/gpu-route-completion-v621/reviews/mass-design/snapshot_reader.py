#!/usr/bin/env python3
"""CPU-only independent audit of immutable v608 outputs; never invokes a solver.

Run from repository root with Python's standard library. Decimal arithmetic
uses the exact represented FP64 coefficients/vectors and 65 decimal digits.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import struct
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dec(value):
    return Decimal.from_float(float(value))


def inf(values):
    return max(map(abs, values), default=Decimal(0))


def dot(a, b):
    assert len(a) == len(b)
    return sum((x * y for x, y in zip(a, b)), Decimal(0))


class Snapshot:
    def __init__(self, path):
        lines = path.read_text().splitlines()
        assert len(lines) == 15 and lines[0] == 'SPACEPDHCG_QOCO_QP_V1'
        self.n, self.p, self.m, nq, na, ng, self.l, ns, self.shift = map(int, lines[1].split())
        self.ptr = [[int(x) for x in lines[k].split()[1:]] for k in (4, 6, 8)]
        self.idx = [[int(x) for x in lines[k].split()[1:]] for k in (5, 7, 9)]
        self.soc = [int(x) for x in lines[10].split()[1:]]
        assert len(self.soc) == ns and self.l + sum(self.soc) == self.m
        data = [dec(x) for x in lines[11].split()[1:]]
        assert len(data) == nq + na + ng + self.n + self.p + self.m
        self.val = [data[:nq], data[nq:nq + na], data[nq + na:nq + na + ng]]
        at = nq + na + ng
        self.c = data[at:at + self.n]
        self.b = data[at + self.n:at + self.n + self.p]
        self.h = data[at + self.n + self.p:]
        self.origin = [float(x) for x in lines[13].split()[1:]]
        self.offset = dec(lines[14])
        self.q_all_zero = all(v == 0 for v in self.val[0])
        self.q_upper_entries = nq
        self.q_upper_offdiagonal = sum(self.idx[0][k] != j for j in range(self.n)
                                      for k in range(self.ptr[0][j], self.ptr[0][j + 1]))

    def mul(self, matrix, x, transpose=False):
        rows = (self.n, self.p, self.m)[matrix]
        out = [Decimal(0)] * (self.n if transpose else rows)
        for j in range(self.n):
            for k in range(self.ptr[matrix][j], self.ptr[matrix][j + 1]):
                i, value = self.idx[matrix][k], self.val[matrix][k]
                if transpose:
                    out[j] += value * x[i]
                else:
                    out[i] += value * x[j]
                if matrix == 0 and i != j:
                    out[j] += value * x[i]
        return out

    def audit(self, result):
        x, y, z, s = [[dec(a) for a in result[k]] for k in ('x', 'y', 'z', 's')]
        assert tuple(map(len, (x, y, z, s))) == (self.n, self.p, self.m, self.m)
        px, ax, gx = self.mul(0, x), self.mul(1, x), self.mul(2, x)
        aty, gtz = self.mul(1, y, True), self.mul(2, z, True)
        primal_abs = max(inf([a - b for a, b in zip(ax, self.b)]),
                         inf([g + a - h for g, a, h in zip(gx, s, self.h)]))
        dual_abs = inf([q + c + a + g for q, c, a, g in zip(px, self.c, aty, gtz)])
        primal = primal_abs / (1 + max(map(inf, (ax, self.b, gx, self.h, s))))
        dual = dual_abs / (1 + max(map(inf, (px, self.c, aty, gtz))))
        f = dot(x, px) / 2 + dot(self.c, x)
        d = -dot(x, px) / 2 - dot(self.b, y) - dot(self.h, z)
        scale = max(Decimal(1), abs(f), abs(d))
        primal_cone = max([Decimal(0)] + [-a for a in s[:self.l]])
        dual_cone = max([Decimal(0)] + [-a for a in z[:self.l]])
        block = inf([a * b for a, b in zip(s[:self.l], z[:self.l])])
        start = self.l
        for size in self.soc:
            stop = start + size
            primal_cone = max(primal_cone, dot(s[start + 1:stop], s[start + 1:stop]).sqrt() - s[start])
            dual_cone = max(dual_cone, dot(z[start + 1:stop], z[start + 1:stop]).sqrt() - z[start])
            block = max(block, abs(dot(s[start:stop], z[start:stop])))
            start = stop
        values = dict(primal=primal, dual=dual, gap=abs(f - d) / scale,
                      block_complementarity_normalized=block / scale,
                      primal_cone_violation=primal_cone, dual_cone_violation=dual_cone,
                      objective=f, dual_objective=d)
        passes = (max(values[k] for k in ('primal', 'dual', 'gap', 'block_complementarity_normalized')) <= dec(1e-9)
                  and max(primal_cone, dual_cone) <= dec(1e-8))
        return {**{k: float(v) for k, v in values.items()}, 'passes': passes,
                'qualified': passes and result['termination_code'] == 1}

    def mapping(self, result):
        expected_x = [float(a + b) for a, b in zip(result['x_solver'], self.origin)] if self.shift else result['x_solver']
        normal = result['normal_dual_solver']
        expected_y = normal[:self.p]
        expected_z = normal[self.p:self.p + self.l]
        first = self.p + self.l
        for size in self.soc:
            chunk = normal[first:first + size]
            expected_z += [-chunk[-1]] + [-v for v in chunk[:-1]]
            first += size
        assert expected_x == result['x'] and expected_y == result['y'] and expected_z == result['z']
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, default=Path('results/local/2026-09-09/upstream-identical-capture-v608'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = args.artifact
    live_cmake_path = Path('cpp/cuda/CMakeLists.txt')
    live_cmake = live_cmake_path.read_text()
    configure_inputs_fixed = ('tests/upstream_snapshot_replay.cu\n'
                              '        "${CMAKE_CURRENT_SOURCE_DIR}/../include/spacepdhcg/accelerator_c_api.h"\n'
                              '        "${SPACEPDHCG_PDHCG_CLEANUP_PATCH}")') in live_cmake
    assert configure_inputs_fixed
    readme = (base / 'README.md').read_text()
    assert 'selects its sparse-Q BB inner-solve path' in readme
    assert 'none was run here' in readme
    index = json.loads((base / 'sha256.json').read_text())
    assert all(sha((base / k).read_bytes()) == v for k, v in index.items())
    manifest = json.loads((base / 'manifest.json').read_text())
    report = json.loads((base / 'run/report.json').read_text())
    assert report['manifest_sha256'] == sha((base / 'manifest.json').read_bytes())
    assert report['runner_sha256'] == sha((base / 'run/run.py').read_bytes())
    assert report['complete'] and report['gpu_calls'] == len(report['cases']) == 8
    with zipfile.ZipFile(base / 'diagnostic-sources.zip') as archive:
        assert all(sha(archive.read(k)) == v for k, v in manifest['source_files_sha256'].items())
    assert sha(json.dumps(manifest['source_files_sha256'], sort_keys=True, separators=(',', ':')).encode()) == manifest['source_sha256']
    rows = []
    with localcontext() as context:
        context.prec = 65
        for case in report['cases']:
            name = case['name']
            fixture = {'shifted': 'mixed-shifted'}.get(name.rsplit('-', 1)[0], name.rsplit('-', 1)[0])
            snapshot_path = base / 'fixtures' / (fixture + '.txt')
            assert sha(snapshot_path.read_bytes()) == manifest['fixtures_sha256'][snapshot_path.name]
            raw_path = base / 'run' / (name + '.log')
            records = {}
            for line in raw_path.read_text().splitlines():
                if line.startswith('UPSTREAM_REPLAY_') and ' {' in line:
                    prefix, text = line.split(' ', 1)
                    assert prefix not in records
                    records[prefix] = json.loads(text)
            meta, result = records['UPSTREAM_REPLAY_META'], records['UPSTREAM_REPLAY_RESULT']
            assert meta['input_sha256'] == result['input_sha256'] == sha(snapshot_path.read_bytes())
            assert meta['source_sha256'] == manifest['source_sha256']
            assert meta['executable_sha256'] == manifest['executable_sha256'] == report['executable_sha256']
            assert {k: v for k, v in result.items() if k not in ('x', 'y', 'z', 's', 'x_solver', 'normal_dual_solver')} == case['result']
            snapshot = Snapshot(snapshot_path)
            audit = snapshot.audit(result)
            assert audit['passes'] == result['passes_common_kkt_gate'] == case['independent_audit']['passes_common_kkt_gate']
            assert audit['qualified'] == result['qualified'] == case['independent_audit']['qualified']
            row = dict(name=name, input_sha256=sha(snapshot_path.read_bytes()), log_sha256=sha(raw_path.read_bytes()),
                       native_termination=result['termination'], iterations=result['iterations'], inner_iterations=result['inner_iterations'],
                       original_coordinate_mapping_matches=snapshot.mapping(result), decimal_audit=audit,
                       q_numerically_zero=snapshot.q_all_zero, q_stored_upper=snapshot.q_upper_entries,
                       q_stored_full=snapshot.q_upper_entries + snapshot.q_upper_offdiagonal,
                       q_stored_offdiagonal_full=2 * snapshot.q_upper_offdiagonal,
                       inferred_upstream_dispatch='PDHCG_SPARSE_Q -> primal_BB_step_size_update' if snapshot.q_upper_offdiagonal else 'other')
            if name in ('conditioning-seeded', 'difficult-seeded'):
                point_path = base / 'fixtures' / (fixture + '-initial.txt')
                source_x = [float(x) for x in point_path.read_text().splitlines()[3].split()[2:]]
                bits = lambda x: struct.pack('<d', x)
                row['seed_primal_changed_bits'] = sum(bits(a) != bits(b) for a, b in zip(source_x, result['x']))
                row['seed_primal_maximum_absolute_change'] = max(abs(a - b) for a, b in zip(source_x, result['x']))
            rows.append(row)
    findings = {
        'scope': 'CPU-only saved evidence and source review; no new solver calls or modified acceptance gates',
        'artifact_index_sha256': sha((base / 'sha256.json').read_bytes()), 'indexed_files_verified': len(index),
        'manifest_sha256': sha((base / 'manifest.json').read_bytes()),
        'upstream_commit': manifest['upstream_commit'], 'upstream_tree': manifest['upstream_tree'],
        'reference_archive_sha256': manifest['reference_archive_sha256'],
        'executable_sha256': manifest['executable_sha256'], 'source_files_sha256': manifest['source_files_sha256'],
        'current_live_cmake_sha256': sha(live_cmake_path.read_bytes()),
        'current_live_cmake_configure_inputs_fixed': configure_inputs_fixed,
        'archived_readme_structural_zero_interpretation_fixed': True,
        'decimal_precision': 65, 'all_eight_independent_verdicts_match': True, 'rows': rows,
        'review_findings': [
            'No sign, symmetry, cone-order, translation, status-adapter or common-KKT mapping defect found for the eight recorded cases.',
            'Both real P matrices are numerically zero but retain off-diagonal structure. Upstream classifies Q by stored indices, so it executes its sparse-Q BB path; only the mathematical linear proximal map is explicit. The final archived README now states that distinction and makes no zero-removal-run claim.',
            'The archived v608 CMake hash read cpp/include/spacepdhcg/accelerator_c_api.h without adding it to CMAKE_CONFIGURE_DEPENDS. Root corrected the current live CMake dependencies to include that header and the patch; the archived evidence remains unchanged. This was an incremental-build provenance issue, not a demonstrated error in the saved executable identity.',
            'solve_wall_seconds includes solve_qp_problem internal setup, work, transfers and final synchronization; separate create_qp_problem host construction is measured in setup_seconds.',
            'The archived run.py/build.py/package.py preserve original execution context, not a portable turnkey runner from their copied archive locations.',
            'Four seeded certificates and zero qualified cold solutions do not establish a speedup, mission score improvement, or whole-pipeline throughput.'
        ],
        'source_locations': {
            'upstream_structural_Q_classification': '_upstream/pdhcg/src/utils.cu:834',
            'upstream_sparse_Q_dispatch': '_upstream/pdhcg/src/pdhg_core_op.cu:749',
            'upstream_BB_inner_count': '_upstream/pdhcg/src/pdhg_core_op.cu:668',
            'upstream_outer_inner_count_increment': '_upstream/pdhcg/src/pdhg_core_op.cu:762',
            'upstream_SOC_radius_last': '_upstream/pdhcg/src/kernels/pdhcg_soc_cone_kernels.cu:21',
            'upstream_opposite_dual_sign': '_upstream/pdhcg/src/kernels/pdhcg_kernels.cu:680',
            'upstream_native_gap_normalization': '_upstream/pdhcg/src/pdhg_core_op.cu:1292',
        },
    }
    text = json.dumps(findings, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
