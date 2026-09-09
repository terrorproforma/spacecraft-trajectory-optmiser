from pathlib import Path
script=Path(__file__).with_name('launch_family_departures_v849.py').read_text()
script=script.replace("spacepdhcg-family-departures-v849'", "spacepdhcg-family-departures-v850'")
script=script.replace("root.mkdir()", """prior=home/'spacepdhcg-family-departures-v849'
failed=json.loads((prior/'report.json').read_text())
assert failed['complete'] and not failed['success'] and failed['ship']==2 and failed['arm']=='unit'
assert 'Out of range float values' in failed['error']
assert not Path('/proc/'+str(failed['pid'])).exists()
root.mkdir()
import shutil
shutil.copytree(prior/'ship-02-unit',root/'ship-02-unit')
(root/'ship-02-screen.json').write_bytes((prior/'ship-02-screen.json').read_bytes())
(root/'prior-report.json').write_bytes((prior/'report.json').read_bytes())""")
script=script.replace("for p in root.iterdir()}", "for p in root.rglob('*') if p.is_file()}")
script=script.replace('family-departures-launch-v849.json','family-departures-launch-v850.json')
exec(compile(script,str(Path(__file__).with_name('launch_family_departures_v849.py')),'exec'))
