from pathlib import Path
base=Path('build/performance/run_compact_profile_v427.py').read_text().replace('compact-profile-v427','earth-beam-micro-v436').replace('build-spacepdhcg-compact-options-v423','build-spacepdhcg-earth-beam-v431').replace('profile_compact_v427.py','earth_beam_micro_v436.py').replace('compact_profile427','earth_micro436')
start=base.index("  result=json.loads");end=base.index("  r['complete']=True",start)
base=base[:start]+"  assert (root/'measurement.json').exists()\n"+base[end:]
Path('build/performance/run_earth_beam_micro_v436.py').write_text(base)
