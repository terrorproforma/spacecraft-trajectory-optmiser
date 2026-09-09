from pathlib import Path
root=Path(__file__).resolve().parent
script=(root/'run_family_retiming_v853.py').read_text()
old="summary['collected_mass_kg'][body]-=delay*p.C.MINING_RATE_KG_PER_YEAR/p.C.YEAR_DAYS"
new="summary['collected_mass_kg'][body]=min(summary['collected_mass_kg'][body],p.C.maximum_collected_mass(summary['collect_epochs'][body]-summary['deploy_epochs'][body]))"
assert old in script;script=script.replace(old,new)
compile(script,'run.py','exec');(root/'run_family_retiming_v854.py').write_text(script)
launch=(root/'launch_family_retiming_v853.py').read_text().replace('run_family_retiming_v853.py','run_family_retiming_v854.py').replace('spacepdhcg-family-retiming-v853','spacepdhcg-family-retiming-v854').replace('family-retiming-launch-v853.json','family-retiming-launch-v854.json')
launch=launch.replace("root.mkdir();", "failed=home/'spacepdhcg-family-retiming-v853';failed_report=json.loads((failed/'report.json').read_text());assert failed_report['complete'] and not failed_report['success'] and failed_report['native_solves']==0 and not Path('/proc/'+str(failed_report['pid'])).exists()\nroot.mkdir();")
(root/'launch_family_retiming_v854.py').write_text(launch)
