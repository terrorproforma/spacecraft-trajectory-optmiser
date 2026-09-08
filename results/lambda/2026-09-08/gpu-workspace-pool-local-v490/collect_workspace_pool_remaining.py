from pathlib import Path
import json, subprocess, sys
p=Path('build/performance')
for tag in sys.argv[1:]:
    output=subprocess.run(['python3',str(p/'remote_exec.py'),str(p/('archive_'+tag.replace('-','_')+'.py'))],capture_output=True,text=True,check=True,timeout=60)
    metadata=json.loads(output.stdout.strip().splitlines()[-1])
    (p/('archive-'+tag+'.json')).write_text(json.dumps(metadata,indent=2))
    print(tag,metadata,flush=True)
    subprocess.run(['python3',str(p/('retrieve_'+tag.replace('-','_')+'.py'))],check=True,timeout=180)
