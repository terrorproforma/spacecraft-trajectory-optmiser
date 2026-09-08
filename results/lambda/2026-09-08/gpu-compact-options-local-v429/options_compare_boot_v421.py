from pathlib import Path
import sys,os,importlib.util,runpy
from spacepdhcg.gtoc12 import search
# Load both definitions in both modes so startup work is comparable.
path=Path(os.environ['SPACEPDHCG_BENCHMARK_SEARCH_SOURCE'])
spec=importlib.util.spec_from_file_location('spacepdhcg.gtoc12._search_baseline421',path)
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
if os.environ['SPACEPDHCG_BENCHMARK_SEARCH_BASELINE']=='1':
 search.RouteSearch._return_options=old.RouteSearch._return_options
 search.RouteSearch._collect_hop_options=old.RouteSearch._collect_hop_options
runpy.run_module('spacepdhcg',run_name='__main__')
