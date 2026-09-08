#!/usr/bin/env python3
"""Reproduce the archived Decimal65 review from this package alone, CPU only."""
from decimal import localcontext
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import types

here=Path(__file__).resolve().parent
root=here.parent.parent
real=root/'real'
def digest(data):return hashlib.sha256(data).hexdigest()
source=here/'review_real.py';helper=here/'decimal_audit_source.py'
assert digest(source.read_bytes())=='727369e4376fd31d9f8501e5ded4b451dda4614aca9f6e2cf0eef5a6c16f04d4'
assert digest(helper.read_bytes())=='b8697eedc3d91ed48849725b4fd8e17dc7515c020f5f3d30f1aa151794a5dbc4'
review=types.ModuleType('archived_review');review.__file__=str(source)
exec(compile(source.read_bytes(),str(source),'exec'),review.__dict__)
math=types.ModuleType('archived_decimal_math');math.__file__=str(helper)
exec(compile(helper.read_bytes(),str(helper),'exec'),math.__dict__)
members=json.loads((real/'raw-members.json').read_text())
expected=json.loads((here/'findings.json').read_text())['findings']
with tempfile.TemporaryDirectory(prefix='halpern-v612-cpu-review-') as temporary:
    base=Path(temporary);(base/'run').mkdir()
    shutil.copy2(real/'report.json',base/'run/report.json')
    for name in ('run.py','manifest.json'):shutil.copy2(real/name,base/name)
    shutil.copytree(real/'inputs',base/'inputs')
    with tarfile.open(real/'raw-logs.tar.gz') as archive:
        assert {member.name for member in archive.getmembers()}==set(members)
        for name,entry in members.items():
            assert Path(name).name==name and name.endswith('.log')
            data=archive.extractfile(name).read()
            assert len(data)==entry['bytes'] and digest(data)==entry['sha256']
            (base/'run'/name).write_bytes(data)
    review.BASE=base;review.FROZEN=root/'cpu/v612d'
    with localcontext() as context:
        context.prec=65
        actual=review.audit(math)
    assert actual==expected,'reproduced review differs from saved Decimal findings'
print(json.dumps({'complete':True,'all_saved_findings_reproduced':True,'GPU_calls':0,
                  'reviewed_real_executions':10,'qualified_seed_cases':4,'qualified_cold_cases':0}))
