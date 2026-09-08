"""Search coverage for a turnaround whose miner is collected on a later revisit."""

from dataclasses import replace

import numpy as np

from spacepdhcg.gtoc12.gpu_joint_layouts import PreparedInsertions
from spacepdhcg.gtoc12.jointopt import JointItinerary, insertion_pivot
from spacepdhcg.gtoc12.retiming import Visit


def test_later_revisit_separates_new_deploy_and_collect_without_moving_old_epochs():
    visits = [
        Visit(0, False, False, "earth_out"),
        Visit(11, True, False, "deploy_hop"),
        Visit(22, True, False, "collect_hop"),
        Visit(11, False, True, "collect_hop"),
        Visit(22, False, True, "earth_return"),
        Visit(0, False, False, ""),
    ]
    assert insertion_pivot(visits) == 2
    epochs = np.array([0.0, 100.0, 200.0, 800.0, 900.0, 1000.0])
    # A deploy-only turnaround with no dwell must still generate midpoint seeds.
    for k in (2, 3, 4):
        seeds = list(JointItinerary._insertion_seeds(visits, epochs, epochs, 2, 1, k, 33))
        assert len(seeds) == 1
        expanded, arr, dep = seeds[0]
        assert expanded[2].body == 33 and expanded[2].deploy
        assert expanded[k + 2].body == 33 and expanded[k + 2].collect
        keep = [j for j, v in enumerate(expanded) if v.body != 33]
        np.testing.assert_array_equal(arr[keep], epochs)
        np.testing.assert_array_equal(dep[keep], epochs)
        assert arr[2] < arr[k + 2]
    # Preserve the established first combined-camp choice on older routes.
    assert insertion_pivot([*visits[:1], replace(visits[1], collect=True), *visits[2:]]) == 1


def test_routes_without_a_deployment_have_no_prepared_search():
    visits = [
        Visit(0, False, False, "earth_out"),
        Visit(11, False, True, "earth_return"),
        Visit(0, False, False, ""),
    ]
    assert insertion_pivot(visits) is None
    prepared = PreparedInsertions(None, visits, np.zeros(3), np.zeros(3), [33])
    assert prepared.total == 0
