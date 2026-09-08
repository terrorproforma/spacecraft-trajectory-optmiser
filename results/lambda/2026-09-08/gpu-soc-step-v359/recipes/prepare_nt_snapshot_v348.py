from pathlib import Path
s=Path('build/performance/build_cone_determinant_v344.py').read_text().replace('build-qoco-cone-determinant-v344','build-qoco-nt-snapshot-v348')
a=s.index('header=Path(');b=s.index('p.write_text(text)',a)+len('p.write_text(text)')
s=s[:a]+'''p=source/'src/cone.cu';text=p.read_text()
anchor='void compute_nt_scaling(QOCOWorkspace* work)'
assert text.count(anchor)==1
text=text.replace(anchor,Path('build/performance/nt_snapshot_v348.cuh').read_text()+'\\n'+anchor)
anchor='work->data->m, work->data->nsoc, q);\\n}'
assert text.count(anchor)==1
text=text.replace(anchor,'work->data->m, work->data->nsoc, q);\\n  nt_dump(work);\\n}')
p.write_text(text)'''+s[b:]
Path('build/performance/build_nt_snapshot_v348.py').write_text(s)
r=Path('build/performance/replay_linear_snapshot_v333.py').read_text().replace('build-qoco-linear-snapshot-v333','build-qoco-nt-snapshot-v348').replace('SPACEPDHCG_DIAGNOSTIC_LINEAR_DIR','SPACEPDHCG_DIAGNOSTIC_NT_DIR')
r=r.replace("if not k.startswith('SPACEPDHCG_TEST_')","if not k.startswith(('SPACEPDHCG_TEST_','QOCO_REPLAY_'))")
Path('build/performance/replay_nt_snapshot_v348.py').write_text(r)
