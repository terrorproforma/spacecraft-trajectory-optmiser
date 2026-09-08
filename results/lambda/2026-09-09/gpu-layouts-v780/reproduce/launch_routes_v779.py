from pathlib import Path
code=Path('build/performance/launch_routes_v773.py').read_text().replace('v773','v779').replace('spacepdhcg-insertions-v772','spacepdhcg-layouts-v778').replace("commit='e80b9cc4'", "commit='49b1babe'")
code=code.replace("'JOINT_DEVICE_SEARCH','JOINT_DEVICE_INSERTIONS')", "'JOINT_DEVICE_SEARCH','JOINT_DEVICE_INSERTIONS','JOINT_DEVICE_LAYOUTS')")
exec(compile(code,__file__,'exec'))
