from pathlib import Path
import json
p=Path('build/performance')
s=(p/'run_conditioning_campaign_v523.py').read_text().replace('conditioning-campaign-v523','preserve-campaign-v537').replace('conditioning523','preserve537')
s=s.replace('build-qoco-soc-step-v358','build-qoco-preserve-objective-v534')
s=s.replace("   env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='0'", "   env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='0'\n   env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1' if enabled else '0'")
r=json.loads((p/'retained-replay-v515/report.json').read_text())
command=next(row['command'] for row in r['stages'] if row['name']=='pytest')
before="  r['campaigns']=[]"
test=f'''  env['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1'
  env['SPACEPDHCG_TEST_QOCO_RETAINED_REPLAY']='1'
  env['SPACEPDHCG_GTOC12_GPU_TESTS']='1'
  test_command={command!r}
  with (root/'pytest.log').open('x') as log:
   child=subprocess.Popen(test_command,env=env,stdout=log,stderr=subprocess.STDOUT);r.update(stage='pytest',child_pid=child.pid);save();code=child.wait(timeout=1200)
  r['pytest']=dict(command=test_command,returncode=code);save();assert code==0
'''+before
assert s.count(before)==1;s=s.replace(before,test)
(p/'run_preserve_campaign_v537.py').write_text(s)
