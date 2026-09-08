from pathlib import Path
p=Path('build/performance')
run=(p/'search_fleet_routes_v779.py').read_text().replace('spacepdhcg-layouts-v778/repo','spacepdhcg-turnaround-v782/repo').replace('gpu_insertion_search_v779','gpu_turnaround_search_v784').replace("commit='49b1babe'", "commit='9b366c8f'")
(p/'search_fleet_routes_v784.py').write_text(run)
launch=(p/'launch_routes_v766.py').read_text().replace('v766','v784').replace('spacepdhcg-fleet-v763/final','spacepdhcg-turnaround-v782/final')
launch=launch.replace("'JOINT_DEVICE_SEARCH')", "'JOINT_DEVICE_SEARCH','JOINT_DEVICE_INSERTIONS','JOINT_DEVICE_LAYOUTS')")
(p/'status_routes_v784.py').write_text((p/'status_routes_v779.py').read_text().replace('v779','v784'))
exec(compile(launch,__file__,'exec'))
