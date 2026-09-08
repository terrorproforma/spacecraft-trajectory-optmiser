"""Every preparation test forbids loading native libraries, including CUDA."""

import ctypes

import common
import pytest


@pytest.fixture(scope="session", autouse=True)
def cpu_only():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            ctypes, "CDLL", lambda *a, **k: pytest.fail("Native load forbidden in CPU tests")
        )
        patch.setenv("SPACEPDHCG_GTOC12_DATA", common.read(common.ROOT / "profile.json")["data"])
        common.activate()
        yield


@pytest.fixture(scope="session")
def actual_inputs(cpu_only):
    from domain import load_inputs

    return load_inputs()


@pytest.fixture(scope="session")
def old_pool(cpu_only):
    from spacepdhcg.gtoc12.search import RoutePlan

    return [
        RoutePlan.from_summary(row)
        for row in common.read(common.ROOT / "inputs/old-generated-pool.json")
    ]
