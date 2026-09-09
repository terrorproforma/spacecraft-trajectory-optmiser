from pathlib import Path
import hashlib,json,os,subprocess

repo=Path.cwd();evidence=repo/'results/lambda/2026-09-09/gpu-collect-composition-v807';evidence.mkdir()
base=repo/'results/lambda/2026-09-09/gpu-collect-reuse-v806/h100-best/Result.txt'
old=repo/'results/lambda/2026-09-09/gpu-regeneration-v799/h100-best/Result.txt'
correction=repo/'results/local/2026-09-09/current-fleet-composition-v628/Result.txt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sections(p):
    out={}
    for line in p.read_bytes().splitlines(keepends=True):
        row=line.split()
        assert row
        out.setdefault(int(row[0]),bytearray()).extend(line)
    return {k:bytes(v) for k,v in out.items()}
a,b,c=map(sections,(base,old,correction))
assert sha(base)=='f2527a213ce191b94557a5a48e33938b82f95dff2f97594d31a63ff6622d2aca'
assert sha(old)=='97d1f351bf6ad4907ddce887aa47d1fa974f587ab270374f4bb46a5898491d48'
assert sha(correction)=='63446ebf3ff1298911bccade7b789a8fb68a0f7db4171906c8e6783a7f174bab'
assert set(a)==set(b)==set(c)==set(range(1,24))
assert [k for k in a if a[k]!=b[k]]==[4]
assert [k for k in c if c[k]!=b[k]]==[8]
assert a[8]==b[8]
a[8]=c[8];result=evidence/'Result.txt';result.write_bytes(b''.join(a[k] for k in sorted(a)))
footprints={k:{int(line.split()[1]) for line in v.splitlines() if int(line.split()[1])>0} for k,v in a.items()}
assert len(set.union(*footprints.values()))==199
assert all(not footprints[i]&footprints[j] for i in footprints for j in footprints if i<j)
plan=dict(result_sha256=sha(result),sources={str(p.relative_to(repo)):sha(p) for p in (base,old,correction)},changed_ships_relative_v799=[4,8],all_other_21_sections_byte_exact=True,unchanged_ship8_source=True,expected_weighted_kg=12999.452764189185+0.37207919333013706,expected_raw_kg=14278.8501026699+0.43805612590040255,provenance='H100 v804 ship 4 plus locally GPU-refined endpoint-merit-corrected ship 8 from v627 via v628; composition performs no GPU solve.')
(evidence/'plan.json').write_text(json.dumps(plan,indent=2))
script=(repo/'build/performance/check_collect_v807.py').read_text()
launch='''from pathlib import Path
import os,subprocess
home=Path.home();root=home/'spacepdhcg-collect-composition-v807';remote=home.name=='ubuntu'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SPACEPDHCG_GTOC12_DATA=str(home/'spacepdhcg/gtoc12/benchmarks/gtoc12/data') if remote else '/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data')
py=str(home/'spacepdhcg/v1/.venv/bin/python') if remote else '/home/angus/worktrees/spacepdhcg-literature-venv/bin/python'
with (root/'worker.log').open('x') as log:print(subprocess.Popen([py,str(root/'run.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True).pid)
'''
root=Path.home()/'spacepdhcg-collect-composition-v807';root.mkdir();(root/'run.py').write_text(script);(root/'launch.py').write_text(launch)
for p in (result,evidence/'plan.json'):(root/p.name).write_bytes(p.read_bytes())
key='/tmp/traj-key.pem';target='ubuntu@192.222.55.229'
subprocess.run(['ssh','-i',key,'-o','BatchMode=yes',target,'mkdir /home/ubuntu/spacepdhcg-collect-composition-v807'],check=True,timeout=20)
subprocess.run(['scp','-i',key,str(root/'run.py'),str(root/'launch.py'),str(result),str(evidence/'plan.json'),target+':/home/ubuntu/spacepdhcg-collect-composition-v807/'],check=True,timeout=40)
subprocess.run(['ssh','-i',key,'-o','BatchMode=yes',target,'python3 /home/ubuntu/spacepdhcg-collect-composition-v807/launch.py'],check=True,timeout=20)
exec(launch)
print(json.dumps(plan,indent=2))
