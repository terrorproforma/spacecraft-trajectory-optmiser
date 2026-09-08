from pathlib import Path
p=Path('build/performance')
for tag in ['resident-options-v500','lagrange-repeat-v501','lagrange-precision-v503','resident-options-v507']:
 for kind in ['archive','retrieve']:
  source=(p/(kind+'_early_graph_v471.py')).read_text().replace('early-graph-v471',tag)
  if kind=='archive':
   source=source.replace("and p.name!='files-sha256.json'", "and p.name not in ['files-sha256.json','resident-options-probe']")
   if tag.endswith('500'):source=source.replace("assert r['complete'] and not r.get('error')", "assert not r['complete'] and r.get('error')")
   if tag.startswith('lagrange-'):
    source=source.replace("for name,sha in r['source_sha256'].items():", "for name,sha in r.get('source_sha256',{}).items():")
    extra="""basis=Path('/home/ubuntu/spacepdhcg-resident-options-v500/repo')
inputs=['tests/test_gtoc12_gpu_discretisation.py','src/spacepdhcg/gtoc12/verifier.py','src/spacepdhcg/gtoc12/low_thrust.py','src/spacepdhcg/gtoc12/gpu_scvx.py','src/spacepdhcg/gtoc12/constants.py']
for name in inputs:
 target=root/'source-overlay'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(basis/name,target)
(root/'diagnostic-source-sha256.json').write_text(json.dumps({name:hashlib.sha256((basis/name).read_bytes()).hexdigest() for name in inputs},indent=2))
libraries=[Path('/home/ubuntu/spacepdhcg-workspace-pool-v491/core-build/cuda/libspacepdhcg_cuda.so'),Path('/home/ubuntu/spacepdhcg-resident-options-v497/core-build/cuda/libspacepdhcg_cuda.so')]
(root/'diagnostic-runtime-sha256.json').write_text(json.dumps({str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in libraries},indent=2))
"""
    source=source.replace('paths=[p for p',extra+'paths=[p for p')
  (p/(kind+'_'+tag.replace('-','_')+'.py')).write_text(source)
