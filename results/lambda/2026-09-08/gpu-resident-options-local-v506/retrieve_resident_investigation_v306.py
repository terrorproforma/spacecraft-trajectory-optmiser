from pathlib import Path
import os,subprocess,tarfile,hashlib,shutil,json
key=Path('/tmp/traj-key.pem')
if not key.exists():
 fd=os.open(str(key),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as out:out.write(Path('traj-key.pem').read_bytes())
archive=Path('build/performance/collect-resident-investigation-v306.tar.gz')
subprocess.run(['scp','-i',str(key),'-o','BatchMode=yes','-o','ConnectTimeout=10','ubuntu@192.222.55.229:/tmp/collect-resident-investigation-v306.tar.gz',str(archive)],check=True,timeout=45)
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='c78fac213be7da03eebfbc7e51b1cc550d47b55bd0ad3310269465d132093ecc'
root=Path('results/lambda/2026-09-08/gpu-collect-resident-v298');root.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:tar.extractall(root,filter='data')
shutil.copyfile(archive,root/archive.name)
shutil.copytree('build/performance/gpu-collect-resident-v296',root/'rtx5090')
shutil.copyfile('build/performance/collect-resident-v296-final-tests.log',root/'rtx5090/tests.log')
report=json.loads((root/'h100/report.json').read_text())
for name,digest in report['source_sha256'].items():assert hashlib.sha256(Path(name).read_text().encode()).hexdigest()==digest,name
summary=dict(candidate_commit='cc505a20',published_main='f07e82c5',publication_ready=False,reason='Resident v299 misses the 548.255 kg retimed candidate: first Earth departure remains at virtual control 0.07504; resident v301 and host v300/v302 recover it. Cause under investigation.',campaigns={})
for v in [299,300,301,302]:
 r=json.loads((root/f'v{v}/output/run_report.json').read_text());assert r['best']['official']['ok'] and r['best']['independent']['ok']
 summary['campaigns'][str(v)]=dict(seconds=r['wall_seconds_total'],score=r['best']['score_kg'],retiming=r['ships'][0]['retiming']['certified'],screening=r['screening'])
(root/'investigation.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'README.md').write_text('''# Resident collection tables — publication held

Candidate `cc505a20` passes 75 tests on both GPUs, eight independent real-tour
replays and all four H100 sanitizers over 42 cases. However, full campaign v299
misses the 548.255 kg retimed mission after its first Earth departure refinement
fails. It instead certifies 475.975 kg. The same candidate succeeds in v301;
host-table controls v300/v302 succeed. These are not four equal-quality timing
observations, and no overall residency speedup is claimed.

`investigation.json` records the discrepancy. Full outputs and source/binary
fingerprints are retained. `arc-v304` contains 24 successful isolated replays of
the exact 465-day departure with detailed iteration reports; this does not prove
the complete pipeline reliable. `arc-v303` is an invalid diagnostic setup that
could not load cuDSS, not 24 numerical solver failures. The rejected v297 CMake
snapshot lacked required Git metadata and never ran a campaign.

The live visualiser and published main remain on their earlier verified results.
The resident candidate is on the development branch only, pending diagnosis.
''')
(root/'sha256.json').write_text(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='sha256.json'},indent=2)+'\n')
print('Downloaded investigation; all source hashes match; mixed campaign quality retained honestly.')
