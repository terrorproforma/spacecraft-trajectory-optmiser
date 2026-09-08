from pathlib import Path
base=Path('build/performance/run_earth_beam_v434.py').read_text().replace("root=Path('/home/ubuntu/spacepdhcg-earth-beam-v434')", "root=Path('/home/ubuntu/spacepdhcg-earth-beam-v439')").replace('earth434_', 'earth439_')
start=base.index(" run('configure'");end=base.index(" report['runtime_sha256']",start)
base=base[:start]+base[end:]
base=base.replace("core=root/'core-build/cuda/libspacepdhcg_cuda.so'", "core=Path('/home/ubuntu/spacepdhcg-earth-beam-v434/core-build/cuda/libspacepdhcg_cuda.so')")
start=base.index(' tests=');end=base.index(' cli=',start);base=base[:start]+base[end:]
Path('build/performance/run_earth_beam_v439.py').write_text(base)
program="from pathlib import Path\nimport subprocess,json\nroot=Path('/home/ubuntu/spacepdhcg-earth-beam-v439');root.mkdir(exist_ok=False)\n(root/'run.py').write_text("+repr(base)+")\nwith (root/'runner.log').open('x') as log:p=subprocess.Popen(['python3',str(root/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\nprint(json.dumps(dict(pid=p.pid,root=str(root))))\n"
Path('build/performance/launch_earth_beam_v439.py').write_text(program)
Path('build/performance/status_earth_beam_v439.py').write_text(Path('build/performance/status_earth_beam_v434.py').read_text().replace('earth-beam-v434','earth-beam-v439'))
base=Path('build/performance/run_earth_beam_v433.py').read_text().replace('earth-beam-v433','earth-beam-v440').replace('earth433_', 'earth440_')
Path('build/performance/run_earth_beam_v440.py').write_text(base)
