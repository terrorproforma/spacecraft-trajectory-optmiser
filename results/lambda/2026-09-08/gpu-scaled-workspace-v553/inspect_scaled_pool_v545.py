from pathlib import Path
root=Path('/home/ubuntu/spacepdhcg-scaled-pool-v545/repo/build/performance/scaled-pool-v542')
for name in ['pytest.log','memcheck.log','synccheck.log']:
 text=(root/name).read_text();print(name, text[-18000:] if name=='synccheck.log' else text[-1500:])
