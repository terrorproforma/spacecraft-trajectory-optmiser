// CPU-only detection for the restricted original-coordinate causal mass chain.
#pragma once
#include "persistent_l1_snapshot.hpp"
#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"
namespace spacepdhcg::snapshot::mass {
inline std::vector<spacepdhcg_cuda_mass_node> detect(const Snapshot& q,const Canonical& c,
                                                    const l1::Reduction& l) {
    require(!q.shifted && !c.fold_singleton_bounds,"mass elimination requires unshifted generic coordinates");
    require(std::all_of(c.Q.values.begin(),c.Q.values.end(),[](double x){return x==0.0;}),"mass elimination requires exact Q=0");
    require(!l.pairs.empty(),"mass elimination requires the exact L1 representation");
    std::vector<std::vector<std::pair<int,double>>> rows(c.A.rows);
    for(int j=0;j<q.n;++j)for(int k=c.A.offsets[j];k<c.A.offsets[j+1];++k)
        if(c.A.values[k]!=0.0)rows[c.A.indices[k]].emplace_back(j,c.A.values[k]);
    std::vector<int> candidate(q.n,0),initial_row(q.n,-1),next(q.n,-1),entry(q.n,-1),gamma(q.n,-1),virtuals(q.n,-1);
    int count=0;
    for(int j=0;j<q.n;++j) {
        if(c.c[j]!=0.0 || l.lambda[j]!=0.0 || l.smooth_c[j]!=c.c[j])continue;
        bool ok=true;int bounds=0;
        for(int k=c.F.offsets[j];k<c.F.offsets[j+1];++k)if(c.F.values[k]!=0.0)ok=false;
        for(int k=c.A.offsets[j];k<c.A.offsets[j+1];++k) {
            if(c.A.values[k]==0.0)continue;const int r=c.A.indices[k];
            if(r<q.p)continue;
            if(rows[r].size()!=1 || (c.A.values[k]!=1.0 && c.A.values[k]!=-1.0)
                || l.removed_scalar[r]>=0 || !std::isfinite(c.upper[r])
                || c.lower[r]!=-std::numeric_limits<double>::infinity())ok=false;
            ++bounds;
        }
        if(ok && bounds==3){candidate[j]=1;++count;}
    }
    require(count>=2 && count<=4096,"mass chain not found or exceeds diagnostic node bound");
    int initial=-1;
    for(int r=0;r<q.p;++r) {
        bool touches=false;for(auto [j,v]:rows[r])if(candidate[j])touches=true;
        if(!touches)continue;
        require(std::isfinite(c.upper[r]) && c.lower[r]==c.upper[r],"mass equality domain mismatch");
        if(rows[r].size()==1) {
            const auto [j,v]=rows[r][0];require(candidate[j] && v==1.0 && c.upper[r]==1.0 && initial<0,"ambiguous initial mass equality");
            initial=j;initial_row[j]=r;continue;
        }
        require(rows[r].size()==4,"mass equality has unsupported coupling");
        int before=-1,after=-1,g=-1,nu=-1;
        for(auto [j,v]:rows[r]) {
            if(candidate[j]) {
                if(v==-1.0 && before<0)before=j;
                else if(v==1.0 && after<0)after=j;
                else require(false,"ambiguous mass coefficients");
            } else if(v==-1.0 && nu<0)nu=j;
            else if(v>0.0 && std::isfinite(v) && g<0)g=j;
            else require(false,"unsupported Gamma/virtual coefficients");
        }
        require(before>=0 && after>=0 && g>=0 && nu>=0 && next[before]<0 && entry[after]<0,"ambiguous causal mass chain");
        next[before]=after;entry[after]=r;gamma[after]=g;virtuals[after]=nu;
    }
    require(initial>=0 && entry[initial]<0,"missing or cyclic initial mass node");
    std::vector<spacepdhcg_cuda_mass_node> result;std::vector<int> used(q.n,0);
    for(int j=initial;j>=0;j=next[j]) {
        require(!used[j],"mass chain overlaps or cycles");used[j]=1;
        if(j==initial)result.push_back({j,initial_row[j],-1,-1});
        else {
            const int g=gamma[j],nu=virtuals[j];
            require(entry[j]>=0 && g>=0 && nu>=0 && g!=nu && !used[g] && !used[nu]
                && !candidate[g] && !candidate[nu],"mass controls overlap another role");
            for(const auto pair:l.pairs)require(g!=pair.epigraph && nu!=pair.epigraph,"mass control is an eliminated epigraph");
            used[g]=used[nu]=1;result.push_back({j,entry[j],g,nu});
        }
    }
    require(static_cast<int>(result.size())==count,"disconnected or ambiguous mass chain");
    return result;
}
}
