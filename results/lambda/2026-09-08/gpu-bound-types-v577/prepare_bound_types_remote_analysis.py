from pathlib import Path
p=Path('build/performance')
code='from pathlib import Path\nimport json,sys\n'
for root,script in [('/home/ubuntu/spacepdhcg-bound-types-v569/repo/build/performance/bound-types-v569','analyze_bound_types_legs.py'),('/home/ubuntu/spacepdhcg-bound-types-campaign-v572','analyze_bound_types_campaign.py')]:
 code+='root=Path('+repr(root)+')\nif json.loads((root/"report.json").read_text())["complete"]:\n sys.argv=["analysis",str(root)]\n exec('+repr((p/script).read_text())+')\n'
(p/'analyze_bound_types_remote.py').write_text(code)
