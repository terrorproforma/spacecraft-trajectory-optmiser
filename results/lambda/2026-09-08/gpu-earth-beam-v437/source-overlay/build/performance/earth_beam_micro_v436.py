from pathlib import Path
import time, os, json, runpy, dataclasses
import numpy as np
from spacepdhcg.gtoc12.search import RouteSearch

root=Path(os.environ.get('SPACEPDHCG_BEAM_MICRO_OUTPUT','build/performance/earth-beam-micro-v436'))
original=RouteSearch._first_level
class Completed(BaseException):pass

def measured(self):
    rows=[];reference=None
    self._earth_beam_partials=lambda options:options
    for name,candidate in [('warm_host',False),('warm_gpu',True),('host0',False),('gpu0',True),('gpu1',True),('host1',False),('host2',False),('gpu2',True),('gpu3',True),('host3',False)]:
        os.environ['SPACEPDHCG_TEST_GTOC12_EARTH_BEAM']='1' if candidate else '0'
        start=time.perf_counter();options=original(self);seconds=time.perf_counter()-start
        values=np.asarray(options)
        if reference is None:reference=values.copy()
        np.testing.assert_array_equal(values[:,1:4],reference[:,1:4])
        np.testing.assert_allclose(values[:,[0,4,5]],reference[:,[0,4,5]],rtol=2e-9,atol=2e-8)
        rows.append(dict(name=name,candidate=candidate,seconds=seconds,selected=len(options)))
    (root/'measurement.json').write_text(json.dumps(dict(rows=rows,scope='Isolated actual first-level method from full-catalogue campaign; includes metadata and route-row construction, excludes later search/refinement. First host/GPU calls are separate warmups.',settings=dataclasses.asdict(self.settings)),indent=2))
    raise Completed

RouteSearch._first_level=measured
try:runpy.run_module('spacepdhcg',run_name='__main__')
except Completed:pass
