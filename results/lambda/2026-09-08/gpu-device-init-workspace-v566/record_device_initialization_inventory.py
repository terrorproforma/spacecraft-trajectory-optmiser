from pathlib import Path
import ast,json,sys
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg']
sys.path.insert(0,str(Path('src').resolve()))
import pytest
script=ast.parse(Path('build/performance/validate_device_initialization_v558.py').read_text())
assignment=next(n for n in ast.walk(script) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='tests' for t in n.targets))
tests=['tests/'+n.value for n in assignment.value.generators[0].iter.elts]
class Inventory:
 def pytest_collection_finish(self,session):
  ids=[item.nodeid for item in session.items]
  assert ids[111]=='tests/test_gtoc12_gpu_scvx.py::test_gpu_outer_graph_retains_physics_and_objective[False-0-False]'
  remaining=[name for name in ids if name.split('::')[0] in ['tests/test_gtoc12_gpu_scvx.py','tests/test_gtoc12_gpu_cli.py','tests/test_gtoc12_run_final_verification.py']]
  assert len(remaining)==45 and set(ids[:111])|set(remaining)==set(ids)
  Path('build/performance/device-initialization-test-inventory.json').write_text(json.dumps(dict(unique_tests=len(ids),early_passed=111,remaining_passed=45,overlap=len(set(ids[:111])&set(remaining)),nodeids=ids),indent=2))
sys.exit(pytest.main(['--collect-only','-q',*tests],plugins=[Inventory()]))
