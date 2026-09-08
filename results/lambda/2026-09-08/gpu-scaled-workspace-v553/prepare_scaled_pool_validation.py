from pathlib import Path
p=Path('build/performance')
s=(p/'replay_conditioning.py').read_text()
s=s.replace("os.environ['SPACEPDHCG_TEST_GTOC12_QOCO_POOL']='1'", "os.environ['SPACEPDHCG_TEST_GTOC12_QOCO_POOL']='1'\nos.environ['SPACEPDHCG_TEST_GTOC12_SCALED_QOCO_POOL']=mode\nos.environ['SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE']='1'")
s=s.replace('qoco_ruiz_iterations=int(mode)','qoco_ruiz_iterations=2')
(p/'replay_scaled_pool.py').write_text(s)
