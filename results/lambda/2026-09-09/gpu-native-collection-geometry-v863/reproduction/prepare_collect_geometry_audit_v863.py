from pathlib import Path
import hashlib, json

repo = Path(__file__).resolve().parents[2]
previous = repo / 'results/lambda/2026-09-09/gpu-native-collection-plan-v859'
out = repo / 'results/lambda/2026-09-09/gpu-native-collection-geometry-v863'
out.mkdir(exist_ok=True)
prior = json.loads((previous / 'saved-audit.json').read_text())
assert prior['passed']
reference = dict(previous_commit='bd664c279a90e30c914b38aef0b8f9df630be56e', previous_audit_sha256=hashlib.sha256((previous / 'saved-audit.json').read_bytes()).hexdigest(), candidate_sha256={side: {ship: prior[side]['comparison'][ship]['plans_sha256'] for ship in ('10', '21')} for side in ('local', 'h100')})
(out / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n', newline='\n')
script = (previous / 'audit_saved.py').read_text()
script = script.replace('236 passed', '248 passed').replace('98 passed', '110 passed').replace('31 passed', '43 passed')
script = script.replace('tests=236,sanitizer_tests_per_tool=98,leak_tests=31', 'tests=248,sanitizer_tests_per_tool=110,leak_tests=43')
script = script.replace("stable=['completed_collection_dp_passes','collect_dp_allocations','collect_dp_rebinds']", "stable=['completed_collection_dp_passes','collect_dp_allocations','collect_dp_rebinds','native_collection_plans','collection_plan_metadata_upload_bytes','collection_plan_result_download_bytes']")
script = script.replace("if enabled:stable+=['native_collection_plans','collection_plan_metadata_upload_bytes','collection_plan_result_download_bytes']", "if enabled:stable+=['collection_geometry_plans','collection_geometry_upload_bytes']")
script = script.replace("modes['host']['result_download_bytes']=modes['host']['counters']['completed_collection_dp_passes']*632", "assert modes['host']['counters']['collection_plan_result_download_bytes']==modes['native']['counters']['collection_plan_result_download_bytes']")
(out / 'audit_saved.py').write_text(script, newline='\n')
print(out)
