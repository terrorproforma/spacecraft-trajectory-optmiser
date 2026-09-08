from pathlib import Path
s=Path('build/performance/build_nt_v347.py').read_text().replace('build-qoco-nt-normalization-v347','build-qoco-nt-step-v349')
# Include the arithmetic before both the step and normalization functions.
s=s.replace("text=text[:a]+'#include \"qoco_cone_arithmetic.cuh\"\\n'+", "text='#include \"qoco_cone_arithmetic.cuh\"\\n'+text[:a]+")
anchor='p.write_text(text)'
s=s.replace(anchor,"a=text.index('__device__ QOCOFloat soc_step_length_dev(');b=text.index('/**',a)\ntext=text[:a]+Path('build/performance/cone-step-v345/new_step.cuh').read_text()+text[b:]\n"+anchor)
Path('build/performance/build_nt_step_v349.py').write_text(s)
r=Path('build/performance/replay_nt_v347.py').read_text().replace('nt-normalization','nt-step').replace('347','349').replace('nt_normalization','nt_step')
Path('build/performance/replay_nt_step_v349.py').write_text(r)
