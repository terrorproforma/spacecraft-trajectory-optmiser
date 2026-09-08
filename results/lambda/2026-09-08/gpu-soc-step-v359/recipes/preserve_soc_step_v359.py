from pathlib import Path
import json,hashlib,shutil,gzip,statistics
root=Path('results/lambda/2026-09-08/gpu-soc-step-v359');root.mkdir(exist_ok=False)
def copy(src,dest):
 src=Path(src);dest=root/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
def tree(src,dest):
 src=Path(src)
 for p in src.rglob('*'):
  if p.is_file() and '__pycache__' not in str(p):copy(p,Path(dest)/p.relative_to(src))
# Final source, full final QP output vectors, all mission outputs and test logs.
tree('build/performance/step-lambda-v359/remote','lambda-final-v359')
tree('build/performance/soc-step-replay-v358','local-final-replays')
tree('build/performance/soc-step-validation-v358','local-final-validation')
tree('build/performance/soc-step-header-v358','local-header-validation')
copy('build/performance/step-lambda-v359/retrieval.json','lambda-final-retrieval.json')
# Matched timing comparison, including its full raw QP outputs and four mission filesets.
tree('build/performance/step-lambda-v353/remote','lambda-comparison-v353')
copy('build/performance/step-lambda-v353/retrieval.json','lambda-comparison-retrieval.json')
# Keep the concrete live input and the full decimal oracle for all 234 cones.
tree('build/performance/step-snapshot-v351','live-step-v351')
copy('/home/angus/build-qoco-step-snapshot-v351/snapshots/step-0000.bin','live-step-v351/step-0000.bin')
copy('/home/angus/build-qoco-step-snapshot-v351/replay.log','live-step-v351/replay.log')
# Normalization experiments were rejected. Retain summaries, kernels, inputs,
# independent references and a trace sample; full replay logs stay in build/performance.
for folder in ['cone-determinant-v343','cone-step-v345','nt-normalization-v347','nt-snapshot-v348']:
 for name in ['summary.json','report.json','probe-summary.json','snapshot-sha256.json','probe.cu','arithmetic.cuh','old_kernel.cuh','new_kernel.cuh','normalization.cuh','old_step.cuh','new_step.cuh']:
  p=Path('build/performance')/folder/name
  if p.exists():copy(p,Path('diagnostics')/folder/name)
for folder in ['cone-replay-v344','cone-step-replay-v345','nt-normalization-replay-v347','nt-step-replay-v349','step-only-replay-v352']:
 copy(Path('build/performance')/folder/'report.json',Path('diagnostics')/folder/'report.json')
copy('build/performance/nt-lambda-v350/remote/report.json','diagnostics/lambda-nt-v350/report.json')
copy('build/performance/nt-lambda-v350/retrieval.json','diagnostics/lambda-nt-v350/retrieval.json')
for name in ['nt-0000.bin','nt-0028.bin']:
 copy(Path('/home/angus/build-qoco-nt-snapshot-v348/snapshots')/name,Path('diagnostics/nt-snapshot-v348/samples')/name)
# Full compact live pairs allow re-running the independent normalization check.
for name in ['inputs.bin','reference.json','output.bin']:
 copy(Path('build/performance/nt-snapshot-v348')/name,Path('diagnostics/nt-snapshot-v348')/name)
recipes=['prepare_nt_v347.py','audit_nt_v347.py','build_nt_v347.py','replay_nt_v347.py','nt_snapshot_v348.cuh','prepare_nt_snapshot_v348.py','build_nt_snapshot_v348.py','replay_nt_snapshot_v348.py','audit_nt_snapshot_v348.py','audit_nt_probe_v348.py','prepare_nt_step_v349.py','build_nt_step_v349.py','replay_nt_step_v349.py','run_nt_lambda_v350.py','step_snapshot_v351.cuh','prepare_step_snapshot_v351.py','build_step_snapshot_v351.py','replay_step_snapshot_v351.py','audit_step_snapshot_v351.py','prepare_live_step_probe_v351.py','prepare_step_only_v352.py','build_step_only_v352.py','replay_step_only_v352.py','prepare_step_lambda_v353.py','run_step_lambda_v353.py','validate_soc_step_header_v358.py','prepare_final_step_v358.py','build_final_step_v358.py','test_final_step_v358.py','replay_final_step_v358.py','test_full_preparation_v358.py','prepare_final_lambda_v359.py','run_final_lambda_v359.py','retrieve_step_evidence.py','preserve_soc_step_v359.py']
for name in recipes:copy(Path('build/performance')/name,Path('recipes')/name)
copy('build/performance/qp-ir-v309/audit.py','recipes/audit_original_qp.py')
files=['cpp/cuda/patches/qoco_soc_step.cuh','cpp/cuda/tests/qoco_soc_step_probe.cu','scripts/gpu/prepare_qoco_soc_step.py','scripts/gpu/prepare_qoco_gpu.py']
for name in files:copy(name,Path('final-source')/name)
(root/'source-sha256.json').write_text(json.dumps({name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},indent=2))
report=json.loads((root/'lambda-comparison-v353/report.json').read_text())
med={key:statistics.median(x['seconds'] for x in report['campaigns'] if x['candidate']==value) for key,value in [('baseline',False),('candidate',True)]}
summary=dict(median_seconds=med,time_reduction_percent=100*(1-med['candidate']/med['baseline']),campaigns=report['campaigns'],final_local_qualified=31,final_local_baseline_qualified=28,final_lambda_qualified=28,final_lambda_baseline_qualified=28,repeats_per_mode=32)
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
# Compress large text/binary evidence losslessly; hashes refer to stored gzip files.
for p in list(root.rglob('*')):
 if p.is_file() and p.stat().st_size>500_000 and p.suffix in ['.log','.json','.bin'] and '/output/' not in p.as_posix():
  data=p.read_bytes();p.with_name(p.name+'.gz').write_bytes(gzip.compress(data,mtime=0));p.unlink()
print('evidence bytes',sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
