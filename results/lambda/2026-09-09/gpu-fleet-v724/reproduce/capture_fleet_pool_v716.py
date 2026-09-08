from pathlib import Path
source=Path('build/performance/capture_fleet_pool_v714.py').read_text()
source=source.replace("out=Path('/home/angus/spacepdhcg-fleet-pool-v714')", "out=Path('/home/angus/spacepdhcg-fleet-pool-v716')")
source=source.replace('from spacepdhcg.gtoc12.solution import Solution', 'from spacepdhcg.gtoc12.solution import Solution\nfrom spacepdhcg.gtoc12.pipeline import plan_from_route_summary')
source=source.replace("    elif not solution.is_file():reason='missing solution'\n",'')
source=source.replace("plan=data['plan'];identifier=len(rows)", "plan=plan_from_route_summary(data).summary();identifier=len(rows)")
source=source.replace("solution=solution.as_posix(),ship_id=1)","solution=solution.as_posix(),ship_id=1,source_available=solution.is_file())")
exec(compile(source,'capture-v716','exec'))
