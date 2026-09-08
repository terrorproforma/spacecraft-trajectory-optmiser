"""Verify frozen source/build evidence only; no native library or project imports."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
records = {}
for version in ('c', 'd', 'e', 'f', 'g'):
    folder = ROOT / ('build/performance/completion-model-v622' + version)
    report = json.loads((folder/'report.json').read_text())
    assert report['complete'] and report['gpu_calls'] == 0
    assert report['base'] == 'fdf52ae31d259240ffebdcf79ed0965dce8b9298'
    assert sha((folder/'source.tar.gz').read_bytes()) == report['source_archive_sha256']
    with tarfile.open(folder/'source.tar.gz') as archive:
        members = [x for x in archive if x.isfile()]
        assert set(x.name for x in members) == set(report['owned_sources'])
        for member in members:
            assert sha(archive.extractfile(member).read()) == report['owned_sources'][member.name]
    for stage in report['stages']:
        assert stage['exit_code'] == 0
        assert sha((folder/(stage['name']+'.log')).read_bytes()) == stage['log_sha256']
    records[version] = {
        'report_sha256': sha((folder/'report.json').read_bytes()),
        'owned_sources': report['owned_sources'],
        'library': report['library'],
        'archive_sha256': report['source_archive_sha256'],
        'all_stage_logs_verified': True,
    }
changed = [name for name, value in records['c']['owned_sources'].items()
           if records['d']['owned_sources'][name] != value]
assert changed == ['tests/test_gtoc12_gpu_completion_model.py']
changed_e = [name for name, value in records['d']['owned_sources'].items()
             if records['e']['owned_sources'][name] != value]
assert changed_e == ['src/spacepdhcg/gtoc12/completion_capture.py']
findings = {
    'scope': 'Independent manual ABI/model/capture/lifecycle review; frozen CPU/build evidence verification only.',
    'frozen_versions': records,
    'c_to_d_source_delta': 'Test saves six comparison NPZ readbacks before assertions; production source is identical.',
    'd_to_e_source_delta': 'Capture checks exact per-deploy collected readback and finite nonnegative COSTED inflation; CUDA/model code is identical.',
    'e_to_f_runtime_delta': 'Compact leg metadata only fills parameters for the selected model, matching the original packer; cost arithmetic/gates unchanged. Additional queue/refiner CPU files frozen.',
    'f_to_g_runtime_delta': 'No native source change. Overflow test uses valid half-day lattice and adds a CPU fixture-path regression; direct refiner continuity validation also included in frozen CPU sources.',
    'reviewed_fixes': [
        'Production compact input packing validates request epochs before model/sweep lookup.',
        'Native cached-grid return row quotient must be finite before any output write or kernel.',
    ],
    'planned_models': ['fit', 'flat', 'ratio'],
    'planned_valid_evaluate_calls': 18,
    'planned_candidate_evaluations': 2358,
    'planned_malformed_evaluate_calls_without_kernels': 6,
    'planned_valid_kernels': 42,
    'actual_review_GPU_calls': 0,
    'decision': 'Source/test GO for finite parent-owned launch; not a runtime parity or performance result.',
}
(OUT/'findings.json').write_text(json.dumps(findings, indent=2)+'\n')
print(json.dumps({'passed': True, 'report_sha256': sha((OUT/'findings.json').read_bytes())}))
