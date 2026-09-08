from pathlib import Path
s=Path('build/performance/build_nt_snapshot_v348.py').read_text().replace('build-qoco-nt-snapshot-v348','build-qoco-step-snapshot-v351')
a=s.index("p=source/'src/cone.cu'");b=s.index('p.write_text(text)',a)+len('p.write_text(text)')
s=s[:a]+'''matches=list(source.rglob('qoco_device_step_cones.cuh'));assert len(matches)==1
p=matches[0];text=p.read_text();a=text.rindex('}');text=text[:a]+'  dump_small_step(u,direction,factor,solver,output);\\n'+text[a:]
p.write_text(Path('build/performance/step_snapshot_v351.cuh').read_text()+'\\n'+text)'''+s[b:]
Path('build/performance/build_step_snapshot_v351.py').write_text(s)
r=Path('build/performance/replay_nt_snapshot_v348.py').read_text().replace('build-qoco-nt-snapshot-v348','build-qoco-step-snapshot-v351').replace('SPACEPDHCG_DIAGNOSTIC_NT_DIR','SPACEPDHCG_DIAGNOSTIC_STEP_DIR')
Path('build/performance/replay_step_snapshot_v351.py').write_text(r)
