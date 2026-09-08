from pathlib import Path
p=Path('build/performance')
s='from pathlib import Path\nimport json,sys,contextlib,io\n'
for script,root in [('analyze_scaled_pool_legs.py','/home/ubuntu/spacepdhcg-scaled-followup-v549'),('analyze_scaled_pool_campaign.py','/home/ubuntu/spacepdhcg-scaled-pool-campaign-v548')]:
 s+='sys.argv=['+repr(script)+','+repr(root)+']\n'
 s+='with contextlib.redirect_stdout(io.StringIO()):exec(compile('+repr((p/script).read_text())+','+repr(script)+',"exec"))\n'
 s+='print(Path('+repr(root+'/analysis.json')+').read_text())\n'
(p/'analyze_scaled_remote.py').write_text(s)
