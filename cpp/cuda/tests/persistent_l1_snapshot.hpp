// CPU proof of the restricted exact epigraph structure, before CUDA is loaded.
#pragma once
#include "persistent_snapshot.hpp"
namespace spacepdhcg::snapshot::l1 {
struct Pair {int epigraph,variable,positive_row,negative_row;double lambda;};
struct Reduction {
    std::vector<Pair> pairs;
    std::vector<int> variable_pair,removed_scalar;
    std::vector<double> smooth_c,lambda;
    Csc masked_scalar;
};
inline Reduction detect(const Snapshot& q,const Canonical& c) {
    require(!q.shifted && !c.fold_singleton_bounds,"L1 reduction requires unshifted generic coordinates");
    require(std::all_of(c.Q.values.begin(),c.Q.values.end(),[](double x){return x==0.0;}),"L1 reduction supports exactly zero Q");
    const int n=q.n;
    std::vector<std::vector<std::pair<int,double>>> rows(c.A.rows);
    for(int j=0;j<n;++j)for(int k=c.A.offsets[j];k<c.A.offsets[j+1];++k)
        if(c.A.values[k]!=0.0)rows[c.A.indices[k]].emplace_back(j,c.A.values[k]);
    Reduction out;out.variable_pair.assign(n,-1);out.removed_scalar.assign(c.A.rows,-1);
    out.smooth_c=c.c;out.lambda.assign(n,0.0);out.masked_scalar=c.A;
    for(int t=0;t<n;++t) {
        if(!(c.c[t]>0.0) || !std::isfinite(c.c[t]))continue;
        std::vector<int> column_rows;
        bool eligible=true;
        for(int k=c.A.offsets[t];k<c.A.offsets[t+1];++k)if(c.A.values[k]!=0.0) {
            column_rows.push_back(c.A.indices[k]);if(c.A.values[k]!=-1.0)eligible=false;
        }
        for(int k=c.F.offsets[t];k<c.F.offsets[t+1];++k)if(c.F.values[k]!=0.0)eligible=false;
        if(!eligible || column_rows.size()!=2)continue;
        int variable=-1,positive=-1,negative=-1;
        for(int row:column_rows) {
            if(row<q.p || c.lower[row]!=-std::numeric_limits<double>::infinity()
                || c.upper[row]!=0.0 || rows[row].size()!=2){eligible=false;break;}
            for(auto [j,value]:rows[row])if(j!=t) {
                if(variable>=0 && variable!=j)eligible=false;
                variable=j;
                if(value==1.0)positive=row;
                else if(value==-1.0)negative=row;
                else eligible=false;
            }
        }
        if(!eligible || variable<0 || positive<0 || negative<0 || positive==negative)continue;
        out.pairs.push_back({t,variable,positive,negative,c.c[t]});
    }
    // First diagnostic supports disjoint targets; combining repeated |v|
    // coefficients would require a separate summation/dual-distribution policy.
    for(std::size_t i=0;i<out.pairs.size();++i) {
        const auto pair=out.pairs[i];
        require(out.variable_pair[pair.epigraph]<0 && out.variable_pair[pair.variable]<0,
                "overlapping epigraph variables/targets are unsupported");
        require(out.removed_scalar[pair.positive_row]<0 && out.removed_scalar[pair.negative_row]<0,
                "overlapping epigraph rows are unsupported");
        out.variable_pair[pair.epigraph]=static_cast<int>(i);
        out.variable_pair[pair.variable]=static_cast<int>(i);
        out.removed_scalar[pair.positive_row]=out.removed_scalar[pair.negative_row]=static_cast<int>(i);
        out.smooth_c[pair.epigraph]=0.0;out.lambda[pair.variable]=pair.lambda;
    }
    for(int j=0;j<n;++j)for(int k=c.A.offsets[j];k<c.A.offsets[j+1];++k)
        if(out.removed_scalar[c.A.indices[k]]>=0)out.masked_scalar.values[k]=0.0;
    return out;
}
}
