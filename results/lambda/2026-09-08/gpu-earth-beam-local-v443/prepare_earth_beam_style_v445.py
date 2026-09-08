from pathlib import Path
base=Path('build/performance/check_earth_beam_v443.py').read_text().replace("root=Path('build/performance/earth-beam-v443')", "root=Path('build/performance/earth-beam-style-v445')")
Path('build/performance/check_earth_beam_style_v445.py').write_text(base)
