from pathlib import Path
p=Path('build/performance/run_joint_campaign_v636.py')
s=p.read_text().replace('import hashlib,json,os,subprocess,time,traceback','import fcntl,hashlib,json,os,subprocess,time,traceback').replace('campaign-v636','campaign-v639')
s=s.replace("(('baseline0',0),('candidate0',1),('candidate1',1),('baseline1',0))","(('baseline-retry',0),)")
s=s.replace("'--max-certifications','4']","'--max-certifications','4','--lock',str(root/'child.lock')]")
start=s.index('        start=time.perf_counter()')
end=s.index("    report['success']=True")
body=s[start:end]
s=s[:start]+"        with open('/home/angus/.spacepdhcg-gpu.lock','a') as shared_lock:\n            fcntl.flock(shared_lock,fcntl.LOCK_EX)\n"+''.join('    '+line if line.strip() else line for line in body.splitlines(keepends=True))+s[end:]
Path('build/performance/retry_joint_campaign_v639.py').write_text(s)
