from pathlib import Path
import hashlib,json,os,sys,time,traceback
home=Path.home();root=Path(__file__).resolve().parent
sys.meta_path=[f for f in sys.meta_path if f.__class__.__module__!='_editable_skbc_spacepdhcg'];sys.path.insert(0,str(home/'spacepdhcg-collect-final-v805/repo/src'))
from spacepdhcg.gtoc12.data import load_catalogue,load_bonus_table
from spacepdhcg.gtoc12.solution import Solution
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier
from spacepdhcg.gtoc12.official import run_official_verifier
from spacepdhcg.gtoc12.viewer_export import write_viewer_dataset
started=time.perf_counter();report=dict(complete=False,pid=os.getpid(),gpu_solves=0,success=False)
def save():
    p=root/'report.tmp';p.write_text(json.dumps(report,indent=2));p.replace(root/'report.json')
save()
try:
    path=root/'Result.txt';plan=json.loads((root/'plan.json').read_text());digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==plan['result_sha256']
    catalogue,bonus=load_catalogue(),load_bonus_table();history={}
    check=Gtoc12Verifier(catalogue,bonus=bonus,history=history).verify_file(path)
    report.update(independent=check.summary(),independent_seconds=time.perf_counter()-started);save()
    official=run_official_verifier(path,timeout=60,keep_directory=root/'official')
    (root/'official.stdout').write_text(official.stdout);(root/'official.stderr').write_text(official.stderr)
    report.update(official=official.summary(),qualified=check.ok and official.ok,score_kg=check.weighted_score_fixed_bonus_kg,total_mass_kg=check.total_mass_kg,solution_sha256=digest)
    assert report['qualified'] and check.summary()['ships']==23 and check.summary()['mined_asteroids']==199
    assert abs(report['score_kg']-plan['expected_weighted_kg'])<1e-8 and abs(report['total_mass_kg']-plan['expected_raw_kg'])<1e-8
    assert digest==hashlib.sha256((root/'official/Result.txt').read_bytes()).hexdigest()==hashlib.sha256(path.read_bytes()).hexdigest()
    report['viewer_manifest']=write_viewer_dataset(root/'viewer',Solution.read(path),history,catalogue,run_id='gpu_collect_composition_v807',commit='7f07c6b6',verification=check.summary(),solution_path=path)
    report['success']=True
except BaseException:report['error']=traceback.format_exc()
report.update(complete=True,seconds=time.perf_counter()-started);save()
