from pathlib import Path
p=Path('build/performance')
launch=(p/'launch_insertions_bench_v769.py').read_text().replace('spacepdhcg-insertions-v769','spacepdhcg-wide-insertions-v781').replace("script=Path('build/performance/bench_insertions_v769.py').read_text().replace('spacepdhcg-insertions-v768','spacepdhcg-insertions-v770')", "script=Path('build/performance/probe_wide_insertions_v781.py').read_text()")
launch=launch.replace('spacepdhcg-insertions-v770/final','spacepdhcg-layouts-v778/final')
exec(compile(launch,__file__,'exec'))
