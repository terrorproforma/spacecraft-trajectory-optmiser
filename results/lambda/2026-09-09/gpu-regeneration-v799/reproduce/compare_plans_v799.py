from pathlib import Path
import json,tarfile,math,collections
root=Path('results/lambda/2026-09-09/gpu-regeneration-v799');different=[];structural=[];count=0
def walk(a,b,path):
    global count
    if a==b:return
    if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
        for k in a:walk(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        count+=1;different.append((abs(a-b),path,a,b))
    else:structural.append((path,str(a)[:100],str(b)[:100]))
with tarfile.open(root/'local.tar.gz') as a,tarfile.open(root/'h100.tar.gz') as b:
    for name in a.getnames():
        if name.endswith('/plans.json'):walk(json.load(a.extractfile(name)),json.load(b.extractfile(name)),name)
print(json.dumps(dict(numeric_differences=count,fields=dict(collections.Counter(x[1].split('/')[-1] for x in different)),largest=sorted(different,reverse=True)[:12],structural=structural[:10],structural_count=len(structural)),indent=2))
