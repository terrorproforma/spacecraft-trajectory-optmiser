#pragma once

#include "persistent_snapshot.hpp"

namespace spacepdhcg::snapshot {

// Diagnostic transfer between two captures with the same variable/row indexing.
// Qualification belongs to the predecessor. The successor is independently
// audited after rebuilding its slack; it may require actual solver work.
struct WarmSnapshotPoint {
    InitialPoint predecessor;
    InitialPoint successor;
    std::string predecessor_sha256;
};

inline WarmSnapshotPoint warm_snapshot_point(const std::string& point_bytes,
        const Snapshot& source, const Snapshot& target, const Canonical& target_c) {
    require(!source.shifted && !target.shifted && !target_c.fold_singleton_bounds,
        "warm snapshot transfer requires unshifted captures without folded bounds");
    auto same_topology=[](const Csc& a,const Csc& b) {
        return a.rows==b.rows && a.columns==b.columns
            && a.offsets==b.offsets && a.indices==b.indices;
    };
    require(source.n==target.n && source.p==target.p && source.m==target.m
        && source.nonnegative==target.nonnegative && source.soc==target.soc
        && same_topology(source.P,target.P) && same_topology(source.A,target.A)
        && same_topology(source.G,target.G), "warm snapshot topology or cone layout differs");
    require(source.P.values!=target.P.values || source.A.values!=target.A.values
        || source.G.values!=target.G.values || source.c!=target.c || source.b!=target.b
        || source.h!=target.h, "warm snapshot requires changed problem coefficients");

    WarmSnapshotPoint out;
    out.predecessor_sha256=source.input_sha256;
    // Keep the strict existing source identity, finiteness and KKT checks.
    out.predecessor=initial_point(point_bytes,source,canonical(source));
    auto& seed=out.successor;
    seed.file_sha256=out.predecessor.file_sha256;
    seed.coordinates="original";
    seed.primal=out.predecessor.supplied.x;
    seed.dual.assign(target_c.A.rows+target_c.F.rows,0);
    for(int i=0;i<target.p;++i)seed.dual[i]=out.predecessor.supplied.y[i];
    for(int i=0;i<target.m;++i) {
        if(i>=target.nonnegative)
            seed.dual[target_c.A.rows+target_c.soc_to_affine[i]]=-out.predecessor.supplied.z[i];
        else seed.dual[target_c.nonnegative_to_scalar[i]]=out.predecessor.supplied.z[i];
    }
    seed.reconstructed=original_vectors(target,target_c,seed.primal,seed.dual);
    require(same_fp64_bits(seed.reconstructed.x,out.predecessor.supplied.x)
        && same_fp64_bits(seed.reconstructed.y,out.predecessor.supplied.y)
        && same_fp64_bits(seed.reconstructed.z,out.predecessor.supplied.z),
        "warm snapshot transfer changed original primal or dual bits");
    seed.reconstructed_audit=audit(target,seed.reconstructed,
        initial_point_tolerance,initial_point_cone_tolerance);
    require(seed.reconstructed_audit.finite,"nonfinite successor warm-start audit");
    // Source slack certifies only the source. Never import it into the target.
    seed.supplied=seed.reference=seed.reconstructed;
    seed.supplied_audit=seed.roundtrip_audit=seed.reconstructed_audit;
    return out;
}

} // namespace spacepdhcg::snapshot
