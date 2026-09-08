from pathlib import Path
p=Path('build/performance')
(p/'probe_wide_insertions_v783.py').write_text((p/'probe_wide_insertions_v781.py').read_text().replace('spacepdhcg-layouts-v778/repo/src','spacepdhcg-turnaround-v782/repo/src'))
launch=(p/'launch_wide_insertions_v781.py').read_text().replace('spacepdhcg-wide-insertions-v781','spacepdhcg-wide-insertions-v783').replace('probe_wide_insertions_v781.py','probe_wide_insertions_v783.py')
(p/'status_wide_insertions_v783.py').write_text((p/'status_wide_insertions_v781.py').read_text().replace('v781','v783'))
exec(compile(launch,__file__,'exec'))
