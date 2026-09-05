"""cluster-fleet: union of family partitions (several radii x band sets) priced cheapest-first.

Edits src/spacepdhcg/gtoc12/bundles.py (family_partitions), src/spacepdhcg/gtoc12/cli.py
(--cluster-radius list, --all-family-bands, 480-min budget mark, partition report) and appends a
test to tests/test_gtoc12_bundles.py.  Every anchor must match exactly once.
"""
from __future__ import annotations

import pathlib


def replace_once(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    assert count == 1, f"{path}: anchor found {count} times:\n{old[:200]}"
    path.write_text(text.replace(old, new, 1))


bundles = pathlib.Path("src/spacepdhcg/gtoc12/bundles.py")
cli = pathlib.Path("src/spacepdhcg/gtoc12/cli.py")
tests = pathlib.Path("tests/test_gtoc12_bundles.py")

# -- bundles.py: family_partitions ------------------------------------------------------------
replace_once(
    bundles,
    "\n\n_WORKER: dict[str, Any] = {}\n",
    '''

FAMILY_LABEL_STRIDE = 100_000  # label offset between partitions (family directories never collide)


def family_partitions(
    catalogue: AsteroidCatalogue,
    ids: IntArray,
    *,
    bands: Sequence[tuple[str, ClusterBands]],
    min_members: int = 12,
    settings: SearchSettings | None = None,
    excluded: set[int] | frozenset[int] | None = None,
) -> list[tuple[int, IntArray, dict[str, Any]]]:
    """Cheapest-first union of the co-moving families of several partitions.

    Every ``(name, bands)`` partition (a radius x band-set choice) is clustered and ranked on its
    own (``family_clusters`` + ``rank_families`` at the partition's visit epochs); the k-th
    partition's labels are offset by ``k * FAMILY_LABEL_STRIDE`` so ``clusters/family_*``
    directories and column identifiers never collide, and a family whose member set already
    appeared in an earlier partition is dropped (the same pool would be priced twice).  Nested
    families (a radius-1.6 family inside its radius-1.75 parent) are kept: their beams see
    different pools and yield different chains, which is the point of pricing several radii.
    Each stats dict records ``partition``, ``partition_index``, ``radius`` and
    ``label_in_partition``.  Deterministic: partitions in the given order, then the global
    cheapest-first order of ``rank_families`` (ties on the offset label).
    """

    seen: set[frozenset[int]] = set()
    ranked: list[tuple[int, IntArray, dict[str, Any]]] = []
    for index, (name, band) in enumerate(bands):
        families = family_clusters(
            catalogue, ids, bands=band, min_members=min_members, excluded=excluded
        )
        for label, members, stats in rank_families(
            catalogue, families, settings, visit_epochs=band.phase_epochs
        ):
            key = frozenset(int(a) for a in members)
            if key in seen:
                continue
            seen.add(key)
            if int(label) >= FAMILY_LABEL_STRIDE:
                raise ValueError(f"family label {label} exceeds the partition stride")
            ranked.append(
                (
                    int(label) + index * FAMILY_LABEL_STRIDE,
                    members,
                    {
                        **stats,
                        "partition": name,
                        "partition_index": index,
                        "radius": float(band.radius),
                        "label_in_partition": int(label),
                    },
                )
            )
    ranked.sort(key=lambda item: (item[2]["score"], item[0]))
    return ranked


_WORKER: dict[str, Any] = {}
''',
)

# -- cli.py -----------------------------------------------------------------------------------
replace_once(cli, "BUDGET_MARKS_MINUTES = (30, 60, 120, 240)\n", "BUDGET_MARKS_MINUTES = (30, 60, 120, 240, 480)\n")

replace_once(
    cli,
    '''def cmd_cluster_fleet(args: argparse.Namespace) -> int:
    """Cooperative cluster pricing -> bundle master -> verified fleet, with checkpoints.
''',
    '''def cluster_band_partitions(args: argparse.Namespace) -> list[tuple[str, Any]]:
    """The family partitions of a cluster-fleet run: one per radius x band set.

    ``--cluster-radius`` accepts a comma-separated list; ``--all-family-bands`` prices both the
    collect-window families (``--collect-epoch-families``) and the phasing-aware deploy/collect
    families for every radius.  Deterministic order: radii as given, collect-window first.
    """

    from .clusters import ClusterBands

    radii = [float(item) for item in str(args.cluster_radius).split(",") if item.strip()]
    if not radii:
        raise ValueError("--cluster-radius needs at least one radius")
    all_bands = bool(getattr(args, "all_family_bands", False))
    partitions: list[tuple[str, Any]] = []
    for radius in radii:
        tag = f"{radius:g}"
        if args.collect_epoch_families or all_bands:
            partitions.append(
                (
                    f"collect_r{tag}",
                    ClusterBands.collect_window(radius=radius, phase_deg=args.cluster_phase_deg),
                )
            )
        if not args.collect_epoch_families or all_bands:
            kind = "static" if args.static_families else "phasing"
            partitions.append(
                (
                    f"{kind}_r{tag}",
                    ClusterBands(
                        radius=radius,
                        phase_deg=args.cluster_phase_deg,
                        visit_epochs=None if args.static_families else ClusterBands().visit_epochs,
                    ),
                )
            )
    return partitions


def cmd_cluster_fleet(args: argparse.Namespace) -> int:
    """Cooperative cluster pricing -> bundle master -> verified fleet, with checkpoints.
''',
)

replace_once(
    cli,
    '''        cluster_search_settings,
        family_clusters,
        price_clusters,
        rank_families,
    )
    from .clusters import ClusterBands
    from .cooperative import FleetColumn, solve_fleet_master
''',
    '''        cluster_search_settings,
        family_partitions,
        price_clusters,
    )
    from .cooperative import FleetColumn, solve_fleet_master
''',
)

replace_once(
    cli,
    '''    ids = catalogue_pool(catalogue, args)
    if args.collect_epoch_families:
        bands = ClusterBands.collect_window(
            radius=args.cluster_radius, phase_deg=args.cluster_phase_deg
        )
    else:
        bands = ClusterBands(
            radius=args.cluster_radius,
            phase_deg=args.cluster_phase_deg,
            visit_epochs=None if args.static_families else ClusterBands().visit_epochs,
        )
    clusters = family_clusters(catalogue, ids, bands=bands, min_members=args.min_members)
    # cheapest families first (Earth access + internal hops over the visit epochs), not largest
    ranked = rank_families(
        catalogue,
        clusters,
        cluster_search_settings(ClusterPricingSettings(), 2),
        visit_epochs=bands.phase_epochs,
    )
''',
    '''    ids = catalogue_pool(catalogue, args)
    partitions = cluster_band_partitions(args)
    bands = partitions[0][1]
    # cheapest families first (Earth access + internal hops over the visit epochs), not largest;
    # several partitions (radii x band sets) are unioned without duplicate member sets
    ranked = family_partitions(
        catalogue,
        ids,
        bands=partitions,
        min_members=args.min_members,
        settings=cluster_search_settings(ClusterPricingSettings(), 2),
    )
''',
)

replace_once(
    cli,
    '''                "phasing_aware": bands.visit_epochs is not None,
                "collect_epoch_families": bool(args.collect_epoch_families),
            },
            "families_priced": [
''',
    '''                "phasing_aware": bands.visit_epochs is not None,
                "collect_epoch_families": bool(args.collect_epoch_families),
            },
            "partitions": [
                {
                    "name": name,
                    "radius": band.radius,
                    "phase_deg": band.phase_deg,
                    "visit_epochs": list(band.phase_epochs),
                    "phase_weights": list(band.epoch_weights),
                    "phasing_aware": band.visit_epochs is not None,
                    "families": sum(1 for _l, _m, s in ranked if s["partition"] == name),
                }
                for name, band in partitions
            ],
            "families_priced": [
''',
)

replace_once(
    cli,
    '''    cluster.add_argument("--cluster-radius", type=float, default=1.5)
''',
    '''    cluster.add_argument(
        "--cluster-radius",
        default="1.5",
        help="neighbourhood radius in band units; a comma-separated list unions the partitions",
    )
    cluster.add_argument(
        "--all-family-bands",
        action="store_true",
        help="price both the collect-window and the phasing-aware families of every radius",
    )
''',
)

# -- test -------------------------------------------------------------------------------------
with tests.open("a") as handle:
    handle.write(
        '''

@requires_data
def test_family_partitions_unions_radii_and_bands_without_duplicates(catalogue, monkeypatch):
    """Several partitions are priced as one cheapest-first list: labels offset per partition,
    duplicate member sets dropped, every partition ranked at its own visit epochs."""

    from spacepdhcg.gtoc12 import bundles
    from spacepdhcg.gtoc12.bundles import (
        FAMILY_LABEL_STRIDE,
        family_clusters,
        family_partitions,
    )
    from spacepdhcg.gtoc12.clusters import ClusterBands
    from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance

    ids = build_reduced_instance(catalogue).asteroid_ids
    seen_epochs: list[tuple[float, ...]] = []

    def cheap_rank(_catalogue, families, _settings=None, *, visit_epochs=None, **_kw):
        seen_epochs.append(tuple(visit_epochs))
        # cheaper the larger the family; ties on the label like the real ranker
        ranked = [
            (int(label), members, {"members": float(members.shape[0]), "score": 1000.0 - members.shape[0]})
            for label, members in families
        ]
        ranked.sort(key=lambda item: (item[2]["score"], item[0]))
        return ranked

    monkeypatch.setattr(bundles, "rank_families", cheap_rank)
    collect = ClusterBands.collect_window(radius=2.0, phase_deg=12.0)
    phasing = ClusterBands(radius=2.0, phase_deg=12.0)
    partitions = [("collect_r2", collect), ("phasing_r2", phasing), ("collect_r2_again", collect)]
    ranked = family_partitions(catalogue, ids, bands=partitions, min_members=8)
    assert ranked, "the reduced instance has co-moving families at radius 2.0"
    # every partition was ranked at its own visit epochs, in order
    assert seen_epochs == [collect.phase_epochs, phasing.phase_epochs, collect.phase_epochs]
    # the first partition is present verbatim (labels unchanged) and the repeat adds nothing
    first = sorted((label, tuple(int(a) for a in m)) for label, m, s in ranked if s["partition"] == "collect_r2")
    single = sorted(
        (int(label), tuple(int(a) for a in m))
        for label, m in family_clusters(catalogue, ids, bands=collect, min_members=8)
    )
    assert first == single
    assert not any(s["partition"] == "collect_r2_again" for _l, _m, s in ranked)
    # member sets and labels are unique; later partitions carry the stride offset
    keys = [frozenset(int(a) for a in m) for _l, m, _s in ranked]
    assert len(keys) == len(set(keys))
    labels = [label for label, _m, _s in ranked]
    assert len(labels) == len(set(labels))
    for label, _m, stats in ranked:
        assert label == stats["label_in_partition"] + stats["partition_index"] * FAMILY_LABEL_STRIDE
        assert stats["radius"] == 2.0
    assert any(s["partition_index"] == 1 for _l, _m, s in ranked)
    # cheapest first across partitions
    scores = [s["score"] for _l, _m, s in ranked]
    assert scores == sorted(scores)


def test_cluster_band_partitions_parses_radius_lists_and_band_sets():
    import argparse

    from spacepdhcg.gtoc12.cli import cluster_band_partitions

    args = argparse.Namespace(
        cluster_radius="1.75, 1.6",
        cluster_phase_deg=8.0,
        collect_epoch_families=True,
        static_families=False,
        all_family_bands=False,
    )
    only_collect = cluster_band_partitions(args)
    assert [name for name, _b in only_collect] == ["collect_r1.75", "collect_r1.6"]
    assert [b.radius for _n, b in only_collect] == [1.75, 1.6]
    assert all(len(b.phase_epochs) == 4 for _n, b in only_collect)
    args.all_family_bands = True
    both = cluster_band_partitions(args)
    assert [name for name, _b in both] == [
        "collect_r1.75",
        "phasing_r1.75",
        "collect_r1.6",
        "phasing_r1.6",
    ]
    assert [len(b.phase_epochs) for _n, b in both] == [4, 2, 4, 2]
    # the legacy single float still works
    args = argparse.Namespace(
        cluster_radius=1.5,
        cluster_phase_deg=8.0,
        collect_epoch_families=False,
        static_families=True,
        all_family_bands=False,
    )
    assert [name for name, _b in cluster_band_partitions(args)] == ["static_r1.5"]
'''
    )
print("patched bundles.py, cli.py, tests/test_gtoc12_bundles.py")