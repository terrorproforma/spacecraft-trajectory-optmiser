"""Collection tables retain window gates and costs with GPU-built ephemerides."""

import os

import numpy as np
import pytest

from spacepdhcg.gtoc12 import collectdp
from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.lambert import using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("source,target", [(45738, 25792), (57530, 53410), (10664, 0)])
@pytest.mark.parametrize("remaining", [900.0, 35.0])
def test_collection_tables_match_host_ephemerides(monkeypatch, source, target, remaining):
    settings = collectdp.CollectDPSettings(
        step_days=15.0,
        tofs=(30.0, 180.0, 60.0, 180.0),
        return_tofs=(240.0, 720.0, 420.0),
        end_margin_days=2.25,
        cache_pairs=1,
    )
    start = C.MISSION_END_MJD - remaining
    catalogue = load_catalogue()
    reference = collectdp.CollectPairTable(catalogue, settings, start_epoch=start)
    actual = collectdp.CollectPairTable(catalogue, settings, start_epoch=start)
    def evaluate(table):
        return table.hop(source, target) if target else table.earth_return(source)

    with using_lambert_backend("cuda", maximum_batch_size=31) as gpu:
        with monkeypatch.context() as patch:
            patch.setattr(gpu, "collect_tables_resident", False)
            patch.setattr(collectdp, "cuda_leg_table", lambda *args: None)
            expected = evaluate(reference)
        with monkeypatch.context() as patch:

            def forbidden(*args, **kwargs):
                pytest.fail("collection table used host ephemerides")

            patch.setattr(collectdp, "asteroid_state", forbidden)
            patch.setattr(collectdp, "earth_state", forbidden)
            value = evaluate(actual)
            np.testing.assert_array_equal(np.isfinite(value), np.isfinite(expected))
            # Existing table storage is float32. Device ephemerides may round
            # the final cost to a neighbouring float32 value.
            np.testing.assert_allclose(value, expected, rtol=2e-7, atol=2e-7)
            assert value.dtype == np.float32
            count = gpu.evaluations
            assert evaluate(actual) is value
            assert gpu.evaluations == count
            assert actual.lambert_evaluations == 2 * value.size
            if target:
                actual.hop(target, source)
                assert len(actual._hops) == 1
                assert (source, target) not in actual._hops
            else:
                assert len(actual._returns) == 1
