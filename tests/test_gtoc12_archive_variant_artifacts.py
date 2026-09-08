"""Persistence must retain the same candidate schedules and cooperative primaries.

Only the numerical emission/refinement boundaries are replaced; archive discovery,
plan reconstruction, consistency repair and master column construction are real.
"""

import json
from types import SimpleNamespace

import pytest

from spacepdhcg.gtoc12 import archive, cli, pipeline
from spacepdhcg.gtoc12.bundles import BundleShip, ClusterBundle, bundle_columns
from spacepdhcg.gtoc12.pipeline import RefinedRoute
from spacepdhcg.gtoc12.search import PlannedLeg, RoutePlan


def route(deploys, masses, *, foreign=None, certified=True):
    first = next(iter(deploys or foreign))
    plan = RoutePlan(
        (
            PlannedLeg(0, first, 64000.0, 65000.0, 1.0, 1.0, "earth_out"),
            PlannedLeg(first, 0, 66000.0, 67000.0, 1.0, 1.0, "earth_return"),
        ),
        deploys,
        {asteroid: 66000.0 for asteroid in masses},
        masses,
        0.0,
        1500.0,
        foreign or {},
    )
    return RefinedRoute(plan, [], dict(masses), 1500.0, certified, certified, 1, 0.0, {})


def fingerprint(column):
    return json.dumps(
        {
            "ships": column.ships,
            "deploys": column.deploys,
            "collects": column.collects,
            "foreign": column.foreign,
            "collected_mass": column.collected_mass,
        },
        sort_keys=True,
    )


@pytest.mark.parametrize("directory", ["clusters/family_0007", "columns/f10007"])
def test_archived_variants_round_trip_without_replacing_cooperative_primaries(
    tmp_path, monkeypatch, directory
):
    deployer = route({1: 65000.0, 10: 65000.0}, {1: 100.0})
    collector = route({2: 65000.0}, {2: 100.0, 10: 500.0}, foreign={10: 65000.0})
    rich_deployer_variant = route({3: 65000.0}, {3: 900.0})
    rich_collector_variant = route({4: 65000.0}, {4: 800.0})
    uncertified = route({5: 65000.0}, {5: 1000.0}, certified=False)
    orphaned = route({6: 65000.0, 11: 65000.0}, {6: 1000.0})
    foreign = route({7: 65000.0}, {7: 600.0, 12: 400.0}, foreign={12: 65000.0})
    bundle = ClusterBundle(
        7,
        (1, 2, 10),
        [
            BundleShip(
                1, deployer, [deployer, rich_deployer_variant, uncertified, orphaned, foreign]
            ),
            BundleShip(2, collector, [collector, rich_collector_variant]),
        ],
    )
    assert bundle.consistent() == ""
    original = bundle_columns(bundle, 0)
    assert len(original) == 5  # Two primaries, two standalone variants, one cooperative bundle.
    monkeypatch.setattr(
        pipeline,
        "emit_solution",
        lambda *a, **kw: SimpleNamespace(
            write=lambda path: path.write_text("synthetic trajectory")
        ),
    )
    output = tmp_path / directory
    cli._write_bundle_route_artifacts(bundle, None, output)
    assert len(list(output.rglob("route_summary.json"))) == 4
    assert len(list(output.rglob("Result.txt"))) == 4
    for slot in (1, 2):
        manifest = json.loads((output / f"ship_{slot:02d}" / "archive_manifest.json").read_text())
        assert manifest["qualification"] == "route_certified"
        assert manifest["requires_final_fleet_verification"]
        assert [item["role"] for item in manifest["routes"]] == [
            "bundle_primary",
            "standalone_variant",
        ]
        assert manifest["routes"][0]["summary"] == "route_summary.json"
        for item in manifest["routes"]:
            assert (output / f"ship_{slot:02d}" / item["summary"]).is_file()
            assert (output / f"ship_{slot:02d}" / item["solution"]).is_file()
    groups = archive.discover_archives([tmp_path])
    assert len(groups) == 1 and [len(ship.summaries) for ship in groups[0].ships] == [2, 2]
    # A heavier standalone alternate cannot replace the emitted deployer/collector.
    assert [ship.primary["total_collected_kg"] for ship in groups[0].ships] == [100.0, 600.0]
    rebuilt_plans = archive.group_plans(groups[0])
    assert rebuilt_plans[1][0].deploy_epochs == deployer.plan.deploy_epochs
    assert rebuilt_plans[2][0].foreign_deploy_epochs == collector.plan.foreign_deploy_epochs
    recertified = []

    def refine(plan, *a, **kw):
        recertified.append(plan)
        return RefinedRoute(plan, [], dict(plan.collected_mass), 1500.0, True, True, 1, 0.0, {})

    monkeypatch.setattr(archive, "refine_route", refine)
    monkeypatch.setattr(archive, "_WORKER", {})
    reloaded = archive.recertify_archives(None, groups, workers=1)
    assert len(recertified) == 4
    assert len(reloaded) == 1 and reloaded[0].consistent() == ""
    recovered = bundle_columns(reloaded[0], 0)
    assert len(recovered) == len(original)
    assert sorted(map(fingerprint, recovered)) == sorted(map(fingerprint, original))


def test_primary_and_repeated_variant_objects_are_emitted_once(tmp_path, monkeypatch):
    primary = route({1: 65000.0}, {1: 100.0})
    variant = route({2: 65000.0}, {2: 200.0})
    bundle = ClusterBundle(7, (1,), [BundleShip(1, primary, [primary, variant, variant, primary])])
    monkeypatch.setattr(
        pipeline,
        "emit_solution",
        lambda *a, **kw: SimpleNamespace(
            write=lambda path: path.write_text("synthetic trajectory")
        ),
    )
    cli._write_bundle_route_artifacts(bundle, None, tmp_path)
    assert len(list(tmp_path.rglob("route_summary.json"))) == 2
    manifest = json.loads((tmp_path / "ship_01" / "archive_manifest.json").read_text())
    assert len(manifest["routes"]) == 2
    assert manifest["routes"][1]["variant_index"] == 1
