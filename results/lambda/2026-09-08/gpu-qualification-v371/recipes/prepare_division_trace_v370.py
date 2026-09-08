from pathlib import Path
s=Path('build/performance/build_interior_nt_v369.py').read_text()
s=s.replace('build-qoco-interior-nt-v369','build-qoco-division-trace-v370').replace('build-qoco-interior-step-v367','build-qoco-soc-step-v358')
a=s.index("header=Path(");b=s.index('cmake=',a)
patch='''
p=source/'src/cone.cu';text=p.read_text()
anchor='void cone_division(const QOCOFloat* lambda'
assert text.count(anchor)==1
text=text.replace(anchor,Path('build/performance/division_trace_v370.cuh').read_text()+'\\n'+anchor)
anchor='cone_division_kernel<<<grid, block>>>(lambda, v, d, l, nsoc, q, soc_idx);'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+'\\n    dump_division(lambda,v,d,l,nsoc,q);')
p.write_text(text)
'''
Path('build/performance/build_division_trace_v370.py').write_text(s[:a]+patch+s[b:])
r=Path('build/performance/replay_step_trace_v365.py').read_text().replace('build-qoco-step-trace-v365','build-qoco-division-trace-v370').replace('SPACEPDHCG_DIAGNOSTIC_STEP_DIR','SPACEPDHCG_DIAGNOSTIC_DIVISION_DIR')
Path('build/performance/replay_division_trace_v370.py').write_text(r)
