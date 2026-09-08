from pathlib import Path
p=Path('build/performance')
run=(p/'search_fleet_routes_v784.py').read_text().replace('spacepdhcg-turnaround-v782/repo','spacepdhcg-grid-v788/repo').replace('gpu_turnaround_search_v784','gpu_insertion_grid_v789').replace("commit='9b366c8f'", "commit='59a62be3'")
run=run.replace('insert_neighbours=60,insert_trials=2','insert_neighbours=60,insert_split_points=3,insert_trials=2')
(p/'search_fleet_routes_v789.py').write_text(run)
launch=(p/'launch_routes_v766.py').read_text().replace('v766','v789').replace('spacepdhcg-fleet-v763/final','spacepdhcg-grid-v788/final')
launch=launch.replace("'JOINT_DEVICE_SEARCH')", "'JOINT_DEVICE_SEARCH','JOINT_DEVICE_INSERTIONS','JOINT_DEVICE_LAYOUTS')")
(p/'status_routes_v789.py').write_text((p/'status_routes_v784.py').read_text().replace('v784','v789'))
exec(compile(launch,__file__,'exec'))
