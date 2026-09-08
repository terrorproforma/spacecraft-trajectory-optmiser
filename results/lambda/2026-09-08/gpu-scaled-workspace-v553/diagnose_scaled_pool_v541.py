from pathlib import Path
root=Path('/home/ubuntu/spacepdhcg-scaled-pool-v541')
print((root/'core-configure.log').read_text()[-10000:])
