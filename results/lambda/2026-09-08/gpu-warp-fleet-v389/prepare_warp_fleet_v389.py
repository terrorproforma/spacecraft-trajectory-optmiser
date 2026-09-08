from pathlib import Path
s=Path('build/performance/run_warp_pipeline_v386.py').read_text().replace('warp-pipeline-v386','warp-fleet-v389')
start=s.index("  tests=[");end=s.index('  cli=',start)
s=s[:start]+s[end:]
s=s.replace("[('baseline0',False),('candidate0',True),('candidate1',True),('baseline1',False)]","[('candidate0',True),('baseline0',False)]")
s=s.replace("'warp386_'","'warp_fleet389_'").replace("'--ships','1'","'--ships','4'").replace("'--beam-width','16'","'--beam-width','32'").replace("'--refine-top','3'","'--refine-top','5','--refine-recovery','16'").replace("'--budget-seconds','600'","'--budget-seconds','900'").replace("'--retime-attempts','4'","'--retime-attempts','8'").replace('run(name,cmd,900)','run(name,cmd,1200)')
compile(s,'run389','exec');Path('build/performance/run_warp_fleet_v389.py').write_text(s)
