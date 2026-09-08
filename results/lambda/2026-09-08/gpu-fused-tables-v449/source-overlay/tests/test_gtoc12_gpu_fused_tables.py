"""Direct float32 collection output preserves the existing two-stage GPU table."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.gpu_collect_tables import GpuCollectTable
from spacepdhcg.gtoc12.lambert import using_lambert_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPACEPDHCG_GTOC12_GPU_TESTS") != "1", reason="requires serialized CUDA tests"
)


@pytest.mark.parametrize("source,target", [(45738, 25792), (57530, 53410), (10664, 0)])
def test_fused_tables_bitwise_and_retained_outputs(monkeypatch, source, target):
    catalogue = load_catalogue()
    tofs = np.array([300.0, 300.0 + 5e-10, 300.0 + 2e-9, 100.0, np.nan, 0.0, -1.0])
    retained = []
    with using_lambert_backend("cuda", maximum_batch_size=31) as gpu:
        for n in [1, 3, 257, 2, 501, 7]:
            epochs = 64328.0 + np.arange(n) * 0.25
            if n > 2:
                epochs[-1] = np.nan
            table = SimpleNamespace(catalogue=catalogue, epochs=epochs, lambert_evaluations=0)
            for end in [64628.0, 70000.0, 64000.0]:
                monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_FUSED_TABLES", "0")
                old = GpuCollectTable(gpu, table, source, target, tofs, end)
                expected = old.read()
                old.close()
                monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_FUSED_TABLES", "1")
                actual = GpuCollectTable(gpu, table, source, target, tofs, end)
                value = actual.read()
                np.testing.assert_array_equal(value.view(np.uint32), expected.view(np.uint32))
                assert np.isinf(value[:, 4:]).all()
                if end == 64628.0:
                    assert np.isfinite(value[0, :2]).all()
                    assert np.isinf(value[0, 2])
                retained.append((actual, value.copy()))
        # Subsequent axis replacement cannot mutate tables kept by their owners.
        for table, value in retained:
            np.testing.assert_array_equal(table.read().view(np.uint32), value.view(np.uint32))
            table.close()


def test_fused_table_invalid_arguments_leave_workspace_reusable(monkeypatch):
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_FUSED_TABLES", "1")
    table = SimpleNamespace(
        catalogue=load_catalogue(), epochs=np.array([64328.0]), lambert_evaluations=0
    )
    with using_lambert_backend("cuda", maximum_batch_size=31) as gpu:
        for tofs, end in [(np.array([]), 70000.0), (np.array([300.0]), np.inf)]:
            with pytest.raises(RuntimeError, match="native status"):
                GpuCollectTable(gpu, table, 57530, 53410, tofs, end)
        actual = GpuCollectTable(gpu, table, 57530, 53410, np.array([300.0]), 70000.0)
        assert np.isfinite(actual.read()).all()
        actual.close()
