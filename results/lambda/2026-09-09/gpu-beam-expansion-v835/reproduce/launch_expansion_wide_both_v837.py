from pathlib import Path
import json
home=Path.home();root=home/'spacepdhcg-expansion-wide-v837';root.mkdir()
for name in ('spacepdhcg-expansion-default-v834','spacepdhcg-expansion-bench-v831'):
    report=json.loads((home/name/'report.json').read_text());assert report['complete'] and report['success']
template=home/'spacepdhcg-expansion-bench-v831/measured-2-reuse'
script=(template/'run.py').read_text().replace('spacepdhcg-expansion-final-v833','spacepdhcg-expansion-default-v834').replace('beam_width=64','beam_width=128').replace('max_per_first=64','max_per_first=128')
(root/'run.py').write_text(script)
for name in ('fleet.txt','fit.json'):(root/name).write_bytes((template/name).read_bytes())
launch=(home/'spacepdhcg-expansion-bench-v831/launch.py').read_text().replace('spacepdhcg-expansion-bench-v831','spacepdhcg-expansion-wide-v837').replace('spacepdhcg-expansion-final-v833','spacepdhcg-expansion-default-v834')
(root/'launch.py').write_text(launch);exec(launch)
