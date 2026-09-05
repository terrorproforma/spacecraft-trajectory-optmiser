import argparse, time, json
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.cli import catalogue_pool, cluster_band_partitions
from spacepdhcg.gtoc12.bundles import family_partitions, family_clusters, ClusterPricingSettings, cluster_search_settings

cat = load_catalogue()
args = argparse.Namespace(pool_a_min=2.2, pool_a_max=3.0, pool_e_max=0.15, pool_i_max=8.0,
                          cluster_radius="1.75,1.6", cluster_phase_deg=8.0, collect_epoch_families=True,
                          static_families=False, all_family_bands=True)
ids = catalogue_pool(cat, args)
print("pool", ids.shape[0])
parts = cluster_band_partitions(args)
for mm in (20, 18, 16):
    for name, band in parts:
        fams = family_clusters(cat, ids, bands=band, min_members=mm)
        sizes = sorted((int(m.shape[0]) for _l, m in fams), reverse=True)
        print(f"min_members={mm} {name}: {len(fams)} families; sizes max {sizes[:3]} ... min {sizes[-3:] if sizes else None}")
t = time.perf_counter()
ranked = family_partitions(cat, ids, bands=parts, min_members=20, settings=cluster_search_settings(ClusterPricingSettings(), 2))
print(f"ranked union (min 20, 4 partitions): {len(ranked)} unique families in {time.perf_counter()-t:.1f} s")
from collections import Counter
print("per partition:", Counter(s["partition"] for _l, _m, s in ranked))
print("first 12:", [(l, s["partition"], int(s["members"]), round(s["score"],1)) for l, _m, s in ranked[:12]])
print("score quantiles:", [round(ranked[int(q*(len(ranked)-1))][2]["score"],1) for q in (0, .25, .5, .75, 1.0)])
# compare with the v1 families (collect_r1.75 only) to see where they rank
v1 = [(l, round(s["score"],1)) for l, _m, s in ranked if s["partition"] == "collect_r1.75"]
print("collect_r1.75 count", len(v1))