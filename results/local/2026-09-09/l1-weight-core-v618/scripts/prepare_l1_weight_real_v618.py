"""Prepare, never execute, six matched unit/cancel-global capture diagnostics."""
from pathlib import Path
root=Path('/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser')
source=(root/'build/performance/run_l1_real_v615.py').read_text()
def change(a,b):
    global source
    assert a in source,a
    source=source.replace(a,b)
change("tiny['actual_solve_api_calls']==9 and tiny['actual_optimization_iterations']==10",
       "tiny['actual_solve_api_calls']==22 and tiny['actual_optimization_iterations']==22")
change("assert manifest['complete']","""assert manifest['complete']
assert manifest['frozen_commit'] is None and manifest['source_identity_kind']=='sha256_tree_not_git_commit'
assert manifest['compiled_source_commit']=='uncommitted'
tree=''.join(k+':'+v+'\\n' for k,v in manifest['source_sha256'].items())
assert hashlib.sha256(tree.encode()).hexdigest()==manifest['source_tree_sha256']""")
change("assert tiny['core_sha256']==manifest['library_sha256'] and tiny['test_sha256']==manifest['persistent_l1_test_sha256']",
"""assert tiny['core_sha256']==manifest['library_sha256'] and tiny['test_sha256']==manifest['persistent_l1_test_sha256']
assert tiny['source_tree_sha256']==manifest['source_tree_sha256']
assert tiny['runner_sha256']=='874a46be33fc063caecec1f2bb3c652f02659e9981e9a7d84e281643aa76098c'""")
change("'audit_l1_v615'","'audit_l1_weight_v618'")
change("cases = [(capture, 'l1', True)","cases = [(capture, 'cancel-global', True)")
change("for mode in ('off', 'l1')","for mode in ('unit', 'cancel-global')")
change("'tiny_report_sha256': sha(tiny_path), 'manifest_sha256': args.manifest_sha256, 'runner_sha256': sha(root / 'run.py'),",
"""'tiny_report_sha256': sha(tiny_path), 'manifest_sha256': args.manifest_sha256, 'runner_sha256': sha(root / 'run.py'),
          'source_tree_sha256':manifest['source_tree_sha256'],'source_commit_scope':'uncommitted_frozen_source_tree',
          'base_commit':manifest['base_commit'],'workspace_parent_commit':manifest['workspace_parent_commit'],""")
change("fixed original common gates; same-build dual-first generic LP versus exact L1 prox with reduced preconditioning; full-layout memory retained; no SOTA or mission-backend claim",
       "fixed original common gates; same-build exact L1 dual-first unit versus coefficient-only omega=O/B fixed once after scaling; unchanged full-layout memory and scaling; no reference-driven weight or sweep; no qualified speedup or mission-backend claim without measured qualification")
change("'--common-kkt-stop']\n            if mode=='l1':command+=['--l1-prox']",
       "'--common-kkt-stop','--l1-prox']\n            if mode=='cancel-global':command+=['--l1-weight','cancel-global']")
change("assert meta['halpern_mode']=='off' and meta['l1_prox']==(mode=='l1')\n            assert meta['source_commit']==manifest['frozen_commit']",
"""assert meta['halpern_mode']=='off' and meta['l1_prox']
            assert meta['l1_weight_policy']==('unit_default' if mode=='unit' else 'cancel_global_normalization')
            assert meta['source_commit']=='uncommitted' and meta['source_commit_scope']=='uncommitted_frozen_source_tree'
            assert meta['source_dirty'] is True and meta['source_tree_sha256']==manifest['source_tree_sha256']
            assert meta['base_commit']==manifest['base_commit']""")
change("            if mode=='l1':\n                l=final['l1'];assert l['enabled'] and l['valid'] and l['updates']==final['iterations']",
"""            l=final['l1']
            assert l['enabled'] and l['valid'] and l['updates']==final['iterations']
            assert l['weight_mode']==(0 if mode=='unit' else 2)
            if l['weight_valid']:
                expected_weight=1.0 if mode=='unit' else l['objective_scale']/l['bound_scale']
                assert l['omega']==expected_weight
                assert l['primal_base_step']==l['eta']/expected_weight and l['dual_base_step']==l['eta']*expected_weight
            else:
                assert final['termination'] in (3,4)
            if True:""")
change("root.mkdir(exist_ok=False)","root.mkdir(parents=True,exist_ok=False)")
change("    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\\n')",
       "    (out / 'report.json').write_text(json.dumps(report, indent=2,allow_nan=False) + '\\n')")
target=root/'build/performance/run_l1_weight_real_v618.py'
with target.open('x') as f:f.write(source)
print(target)
