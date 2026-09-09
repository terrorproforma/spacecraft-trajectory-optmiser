from pathlib import Path
import subprocess
inventory=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not inventory.stdout.strip(),inventory.stdout
code=Path('build/performance/launch_collect_plan_bench_host_v857.py').read_text()
before="(root/'run.py').write_text(script)"
after="""script=script.replace('    for name,enabled,measured in schedule:', '''    report['gpu_context']=[]
    for name,enabled,measured in schedule:
        context=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,check=True)
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
        report['gpu_context'].append(dict(run=name,gpu=context.stdout,compute_processes=apps.stdout));save()''')
(root/'run.py').write_text(script)"""
assert before in code;code=code.replace(before,after)
exec(code)
