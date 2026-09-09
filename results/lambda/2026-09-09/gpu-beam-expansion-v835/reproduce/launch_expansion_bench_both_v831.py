from pathlib import Path
import json
home=Path.home();build=home/'spacepdhcg-expansion-final-v833';r=json.loads((build/'report.json').read_text());assert r['complete'] and r['success']
root=home/'spacepdhcg-expansion-bench-v831';root.mkdir()
script=(home/'spacepdhcg-resident-catalogue-bench-v824/run.py').read_text().replace('spacepdhcg-resident-final-v826','spacepdhcg-expansion-final-v833').replace('SPACEPDHCG_TEST_GTOC12_RETAIN_COMPLETION_CATALOGUE','SPACEPDHCG_TEST_GTOC12_GPU_EXPANSION').replace('Shared CUDA catalogue; catalogue digest and collection DP reuse enabled in both modes','CUDA beam expansion and ranking; catalogue and DP reuse enabled in both modes')
script=script.replace('    expected={}', '    expected={};reference={}\n    def compare(a,b):\n        if isinstance(a,float) and isinstance(b,(float,int)):\n            assert abs(a-b)<=2e-9,(a,b)\n            return abs(a-b)\n        assert type(a)==type(b),(type(a),type(b))\n        if isinstance(a,dict):\n            assert a.keys()==b.keys()\n            return max((compare(a[k],b[k]) for k in a),default=0.)\n        if isinstance(a,list):\n            assert len(a)==len(b)\n            return max((compare(x,y) for x,y in zip(a,b)),default=0.)\n        assert a==b,(a,b)\n        return 0.')
script=script.replace('expected.setdefault(ship,digest);assert expected[ship]==digest,(name,ship)', 'expected.setdefault((ship,enabled),digest);assert expected[(ship,enabled)]==digest,(name,ship)\n            plans=json.loads((p/"plans.json").read_text());reference.setdefault(ship,plans);difference=compare(reference[ship],plans)')
script=script.replace("assert digest==hashlib.sha256(prior.read_bytes()).hexdigest(),(name,ship,'previous published candidates')", "assert enabled or digest==hashlib.sha256(prior.read_bytes()).hexdigest(),(name,ship,'previous published candidates')")
script=script.replace('plans_sha256=digest,telemetry=', 'plans_sha256=digest,max_numeric_difference=difference,telemetry=')
(root/'run.py').write_text(script)
launch=(home/'spacepdhcg-resident-catalogue-bench-v824/launch.py').read_text().replace('spacepdhcg-resident-catalogue-bench-v824','spacepdhcg-expansion-bench-v831').replace('spacepdhcg-resident-final-v826','spacepdhcg-expansion-final-v833')
(root/'launch.py').write_text(launch);exec(launch)
