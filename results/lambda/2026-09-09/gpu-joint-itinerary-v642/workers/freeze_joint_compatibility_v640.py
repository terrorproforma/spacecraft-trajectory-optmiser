from pathlib import Path
import difflib,hashlib,json,shutil
root=Path('/home/angus/spacepdhcg-joint-selection-v630');before=root/'repo';after=root/'final-python-v640'
shutil.copytree(before,after,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','build'))
names=['src/spacepdhcg/gtoc12/gpu_joint.py','tests/test_gtoc12_gpu_joint_compatibility.py']
old=(after/names[0]).read_text();new=Path(names[0]).read_text()
diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='measured/gpu_joint.py',tofile='compatible/gpu_joint.py'))
(root/'compatibility.patch').write_text(diff)
for name in names:shutil.copy2(name,after/name)
manifest={str(p.relative_to(after)):hashlib.sha256(p.read_bytes()).hexdigest() for p in after.rglob('*') if p.is_file()}
(root/'compatibility-source.json').write_text(json.dumps(manifest,indent=2))
print(diff)
