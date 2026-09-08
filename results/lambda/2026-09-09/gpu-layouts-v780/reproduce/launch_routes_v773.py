from pathlib import Path
root=Path('build/performance')
run=(root/'search_fleet_routes_v766.py').read_text().replace('spacepdhcg-fleet-v763/repo','spacepdhcg-insertions-v772/repo').replace("run_id='gpu_route_search_v766',commit='258e5c3a'","run_id='gpu_insertion_search_v773',commit='e80b9cc4'")
(root/'search_fleet_routes_v773.py').write_text(run)
launch=(root/'launch_routes_v766.py').read_text().replace('v766','v773').replace('spacepdhcg-fleet-v763/final','spacepdhcg-insertions-v772/final')
# Native insertion batches are selected by the new wrapper by default. The
# explicit switch records this campaign's requested execution scope as well.
launch=launch.replace("'JOINT_DEVICE_SEARCH')", "'JOINT_DEVICE_SEARCH','JOINT_DEVICE_INSERTIONS')")
exec(compile(launch,__file__,'exec'))
