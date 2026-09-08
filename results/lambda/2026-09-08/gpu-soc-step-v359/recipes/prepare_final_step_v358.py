from pathlib import Path
s=Path('build/performance/build_step_only_v352.py').read_text().replace('build-qoco-step-only-v352','build-qoco-soc-step-v358')
a=s.index('header=Path(');b=s.index('p.write_text(text)',a)+len('p.write_text(text)')
s=s[:a]+'''from prepare_qoco_soc_step import prepare
print(prepare(source))'''+s[b:]
Path('build/performance/build_final_step_v358.py').write_text(s)
