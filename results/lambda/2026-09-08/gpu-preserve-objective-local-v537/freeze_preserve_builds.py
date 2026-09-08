from pathlib import Path
import hashlib,json,shutil
p=Path('build/performance');target=p/'preserve-build-v534';target.mkdir(exist_ok=False)
source=Path('/home/angus/build-qoco-preserve-objective-v534')
for name in ['source','final']:shutil.copytree(source/name,target/name)
for name in ['build-0.log','build-1.log','report.json']:shutil.copy2(source/name,target/name)
for name in ['build_preserve_objective_v533.py','build_preserve_objective_v534.py','verify_preserve_preparation.py','preserve-preparation-verification.json']:
 shutil.copy2(p/name,target/name)
(target/'preparation-attempt-v533.json').write_text(json.dumps(dict(complete=False,stage='preparation',gpu_test_started=False,
    error='Guard rejected three CUDA cost-finish launch sites; initial guard permitted only one or two. No source files were written before rejection.',
    resolution='Guard corrected to accept synchronous, queued and graph-owned numeric update paths; clean v534 build succeeds.'),indent=2))
manifest={q.relative_to(target).as_posix():hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(target.rglob('*')) if q.is_file()}
(target/'files-sha256.json').write_text(json.dumps(manifest,indent=2))
print(len(manifest),'build source and binary files frozen')
