"""Run only the four frozen GPU adapter cases; observe readbacks without changing arithmetic."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

KIT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    if not args.collect_only:
        supervisor = os.environ.get('SPACEPDHCG_COMPLETION_ADAPTER_SUPERVISOR')
        if not supervisor or int(supervisor) != os.getppid():
            raise RuntimeError('GPU tests require the reviewed foreground supervisor')
        os.fstat(int(os.environ['SPACEPDHCG_COMPLETION_ADAPTER_LOCK_FD']))
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    pins = json.loads((KIT / 'profile.json').read_text())
    source = Path(pins['host_source_root']).resolve()
    host = json.loads(Path(pins['host_report_path']).read_text())
    assert sha(pins['host_report_path']) == pins['host_report_sha256']
    for name, digest in host['source_sha256'].items():
        assert sha(source / name) == digest, name
    assert sha(pins['library']['path']) == pins['library']['sha256']
    sys.meta_path = [f for f in sys.meta_path
                     if f.__class__.__module__ != '_editable_skbc_spacepdhcg']
    sys.path[:0] = [str(source / 'src'), str(source / 'tests')]
    original_cdll = ctypes.CDLL

    def guarded_cdll(name, *a, **kw):
        if str(name) != 'libc.so.6':
            if args.collect_only or Path(name).resolve() != Path(pins['library']['path']).resolve():
                raise AssertionError('Unexpected native library load: ' + str(name))
        return original_cdll(name, *a, **kw)

    report = {'complete': False, 'passed': False, 'collect_only': args.collect_only,
              'started': time.time(), 'calls': [], 'gpu_scopes': [], 'pytest_reports': [],
              'forbidden_gpu_requests': [], 'source_bindings': {},
              'profile_sha256': sha(KIT / 'profile.json'), 'failures': []}
    try:
        with patch.object(ctypes, 'CDLL', side_effect=guarded_cdll):
            import numpy as np
            import pytest
            from spacepdhcg.gtoc12 import gpu_completion, gpu_lambert, search
            for module in (gpu_completion, gpu_lambert, search):
                path = Path(module.__file__).resolve()
                assert path.is_relative_to(source / 'src')
                report['source_bindings'][module.__name__] = {'path': str(path), 'sha256': sha(path)}
            report['python'] = sys.version
            report['numpy'] = np.__version__
            report['pytest'] = pytest.__version__

            def encode(value):
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and not math.isfinite(value):
                    return {'float64': 'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf'}
                if isinstance(value, np.ndarray):
                    return encode(value.tolist())
                if isinstance(value, (tuple, list)):
                    return [encode(x) for x in value]
                if isinstance(value, dict):
                    return {k: encode(v) for k, v in value.items()}
                return value

            def array(value):
                rows = ([{name: encode(row[name]) for name in value.dtype.names} for row in value]
                        if value.dtype.names else encode(value))
                return {'shape': list(value.shape), 'dtype': encode(value.dtype.descr), 'rows': rows}

            class Observer:
                current = None

                def pytest_collection_modifyitems(self, items):
                    expected = pins['test_suffixes']
                    assert len(items) == 4
                    assert all(item.nodeid.endswith(suffix) for item, suffix in zip(items, expected, strict=True))
                    report['collected_tests'] = [item.nodeid for item in items]

                def pytest_runtest_setup(self, item):
                    self.current = item.nodeid

                def pytest_runtest_logreport(self, value):
                    report['pytest_reports'].append({'nodeid': value.nodeid, 'when': value.when,
                        'outcome': value.outcome, 'duration': value.duration})

            observer = Observer()
            original_init = gpu_completion.GpuCompletion.__init__

            def observed_init(workspace, *a, **kw):
                original_init(workspace, *a, **kw)
                native_evaluate = workspace.evaluate

                def observed_evaluate(*arguments):
                    index = len(report['calls'])
                    assert index < 8, 'No extra completion call permitted'
                    n, nd, nl = map(int, arguments[1:4])
                    assert n == (259 if index % 2 == 0 else 3)
                    assert observer.current.endswith(pins['test_suffixes'][index // 2])
                    item = {'index': index, 'test': observer.current, 'candidates': n,
                            'deploy_slots': nd, 'leg_slots': nl, 'entered_native': False}
                    report['calls'].append(item)
                    inputs = [workspace.inputs[0], workspace.inputs[1][:n],
                              workspace.inputs[2][:nd], workspace.inputs[3][:nl]]
                    path = out / f'call-{index:02d}-input.json'
                    write(path, {'call': item, 'arrays': [array(x) for x in inputs]})
                    item['input_sha256'] = sha(path)
                    item['entered_native'] = True
                    status = native_evaluate(*arguments)
                    item['native_status'] = int(status)
                    path = out / f'call-{index:02d}-output.json'
                    write(path, {'call': item, 'results': array(workspace.results[:n]),
                        'leg_results': array(workspace.leg_results[:nl]),
                        'collected': array(workspace.collected[:nd]), 'stats': array(workspace.stats)})
                    item['output_sha256'] = sha(path)
                    return status

                workspace.evaluate = observed_evaluate

            def forbidden(*a, **kw):
                report['forbidden_gpu_requests'].append({'test': observer.current})
                raise AssertionError('This adapter test must not request Lambert or unrelated GPU work')

            original_gpu_init = gpu_lambert.GpuLambert.__init__

            def observed_gpu_init(gpu, *a, **kw):
                original_gpu_init(gpu, *a, **kw)
                # Block both bound raw Lambert entry points as well as public
                # request methods. The four tests only need the library/owner scope.
                gpu.evaluate = forbidden
                gpu.evaluate_hops = forbidden

            original_close = gpu_lambert.GpuLambert.close

            def observed_close(gpu):
                if gpu.closed:
                    return original_close(gpu)
                item = {'test': observer.current, 'telemetry': encode(dict(gpu.telemetry)),
                        'lambert_batches': gpu.batches, 'lambert_evaluations': gpu.evaluations}
                report['gpu_scopes'].append(item)
                result = original_close(gpu)
                item['closed'] = gpu.closed
                return result

            forbidden_methods = ('solve', 'screen_hops', 'paired_hops', 'paired_options', 'leg_table',
                                 'neighbours', 'select_collection', 'retime_dp', 'collect_table')
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.enter_context(patch.object(gpu_completion.GpuCompletion, '__init__', observed_init))
                stack.enter_context(patch.object(gpu_lambert.GpuLambert, '__init__', observed_gpu_init))
                stack.enter_context(patch.object(gpu_lambert.GpuLambert, 'close', observed_close))
                for name in forbidden_methods:
                    stack.enter_context(patch.object(gpu_lambert.GpuLambert, name, forbidden))
                command = [str(source / 'tests/test_gtoc12_gpu_completion.py') + '::' + name
                           for name in pins['test_suffixes']]
                command += ['-q', '-s', '-x', '-o', 'addopts=', '-p', 'no:cacheprovider',
                            '--rootdir', str(source), '--basetemp', str(out / 'pytest-tmp')]
                if args.collect_only:
                    command.append('--collect-only')
                report['pytest_arguments'] = command
                code = pytest.main(command, plugins=[observer])
            report['pytest_exit_code'] = int(code)
            assert code == 0
            if args.collect_only:
                assert not report['calls'] and not report['gpu_scopes']
            else:
                assert len(report['calls']) == 8 and sum(x['candidates'] for x in report['calls']) == 1048
                assert all(x['native_status'] == 0 for x in report['calls'])
                assert len(report['gpu_scopes']) == 4
                for scope in report['gpu_scopes']:
                    assert scope['closed'] and scope['lambert_batches'] == scope['lambert_evaluations'] == 0
                    assert scope['telemetry']['completion_batches'] == 2
                    assert scope['telemetry']['completion_candidates'] == 262
                phases = report['pytest_reports']
                assert len(phases) == 12 and all(x['outcome'] == 'passed' for x in phases)
            assert not report['forbidden_gpu_requests']
            for name, digest in host['source_sha256'].items():
                assert sha(source / name) == digest, name
            report['passed'] = True
    except BaseException as error:
        report['failures'].append({'type': type(error).__name__, 'message': str(error),
                                   'traceback': traceback.format_exc()})
        raise
    finally:
        report.update(complete=True, ended=time.time(),
                      actual_evaluation_calls=sum(x['entered_native'] for x in report['calls']),
                      actual_completed_evaluation_calls=sum('native_status' in x for x in report['calls']),
                      actual_candidates=sum(x['candidates'] for x in report['calls'] if x['entered_native']),
                      lambert_requests=0,
                      lambert_request_guards='Both bound native Lambert entry points and public request methods reject before GPU execution.',
                      scope='Costing parity with readback observation; timings include observation overhead and are not a benchmark.')
        write(out / 'report.json', report)


if __name__ == '__main__':
    main()
