#pragma once

// CPU-only reader, conversion and independent audit for diagnostic QOCO snapshots.
// No CUDA initialization is permitted until read() and canonical() have succeeded.
#include "spacepdhcg/cuda/persistent_pdhcg_c_api.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spacepdhcg::snapshot {

inline void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

// SHA-256 identifies exact file bytes, independently of parsed numeric identity.
inline std::string sha256(const std::string& bytes) {
    constexpr std::array<std::uint32_t,64> k{
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t,8> h{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::vector<unsigned char> data(bytes.begin(),bytes.end());
    const auto bits=std::uint64_t(data.size())*8;
    data.push_back(0x80);
    while(data.size()%64!=56) data.push_back(0);
    for(int shift=56;shift>=0;shift-=8) data.push_back(static_cast<unsigned char>(bits>>shift));
    for(std::size_t begin=0;begin<data.size();begin+=64) {
        std::array<std::uint32_t,64> w{};
        for(int i=0;i<16;++i) for(int j=0;j<4;++j) w[i]=(w[i]<<8)|data[begin+4*i+j];
        for(int i=16;i<64;++i) {
            const auto a=w[i-15],b=w[i-2];
            w[i]=w[i-16]+(std::rotr(a,7)^std::rotr(a,18)^(a>>3))+w[i-7]
                +(std::rotr(b,17)^std::rotr(b,19)^(b>>10));
        }
        auto [a,b,c,d,e,f,g,z]=h;
        for(int i=0;i<64;++i) {
            const auto t1=z+(std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25))
                +((e&f)^((~e)&g))+k[i]+w[i];
            const auto t2=(std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22))
                +((a&b)^(a&c)^(b&c));
            z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        const std::array<std::uint32_t,8> delta{a,b,c,d,e,f,g,z};
        for(int i=0;i<8;++i) h[i]+=delta[i];
    }
    std::ostringstream out;out<<std::hex<<std::setfill('0');
    for(auto v:h) out<<std::setw(8)<<v;
    return out.str();
}

inline std::string file_bytes(const std::string& path, std::size_t limit=512ULL*1024*1024) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    require(bool(in),"cannot open file: "+path);
    const auto size=in.tellg();
    require(size>=0 && static_cast<std::uint64_t>(size)<=limit,"file exceeds diagnostic size limit");
    std::string bytes(static_cast<std::size_t>(size),'\0');in.seekg(0);
    if(!bytes.empty()) in.read(bytes.data(),static_cast<std::streamsize>(bytes.size()));
    require(bool(in),"cannot read file: "+path);return bytes;
}

struct Csc {
    int rows{},columns{};
    std::vector<int> offsets,indices;
    std::vector<double> values;
};
struct Snapshot {
    int n{},p{},m{},nonnegative{};
    std::array<int,4> settings_integer{};
    std::array<double,9> settings_float{};
    Csc P,A,G;
    std::vector<int> soc;
    std::vector<double> c,b,h,translated,origin;
    double offset{};
    bool shifted{};
    std::string input_sha256;
};

inline std::vector<long double> multiply(const Csc& matrix,const std::vector<long double>& x,
                                       bool transpose=false,bool symmetric_upper=false) {
    require(x.size()==static_cast<std::size_t>(transpose?matrix.rows:matrix.columns),"multiply dimension mismatch");
    std::vector<long double> y(transpose?matrix.columns:matrix.rows);
    for(int j=0;j<matrix.columns;++j) for(int k=matrix.offsets[j];k<matrix.offsets[j+1];++k) {
        const auto i=matrix.indices[k];const long double v=matrix.values[k];
        if(transpose) y[j]+=v*x[i];else y[i]+=v*x[j];
        if(symmetric_upper && i!=j) y[j]+=v*x[i];
    }
    return y;
}
inline std::vector<long double> extended(const std::vector<double>& x) { return {x.begin(),x.end()}; }
inline long double dot(const std::vector<long double>& a,const std::vector<long double>& b) {
    require(a.size()==b.size(),"dot dimension mismatch");long double out=0;
    for(std::size_t i=0;i<a.size();++i) out+=a[i]*b[i];
    return out;
}
inline long double norm_inf(const std::vector<long double>& x) {
    long double out=0;for(auto v:x) { if(!std::isfinite(v)) return INFINITY;out=std::max(out,std::abs(v)); }return out;
}

inline void topology(const Csc& a,bool upper=false) {
    require(a.offsets.size()==static_cast<std::size_t>(a.columns)+1 && a.indices.size()==a.values.size(),"CSC length mismatch");
    require(a.offsets.front()==0 && a.offsets.back()==static_cast<int>(a.indices.size()),"CSC endpoints mismatch");
    for(int j=0;j<a.columns;++j) {
        require(a.offsets[j]>=0 && a.offsets[j]<=a.offsets[j+1] && a.offsets[j+1]<=static_cast<int>(a.indices.size()),"invalid CSC offsets");
        int previous=-1;
        for(int k=a.offsets[j];k<a.offsets[j+1];++k) {
            const int row=a.indices[k];
            require(row>=0 && row<a.rows && row>previous,"CSC rows must be in range, strictly sorted and unique");
            require(!upper || row<=j,"P must contain upper-triangular entries only");
            require(std::isfinite(a.values[k]),"nonfinite matrix value");previous=row;
        }
    }
}

inline Snapshot read(const std::string& bytes) {
    constexpr std::int64_t limit=8'000'000;
    require(bytes.size()<=128ULL*1024*1024,"snapshot exceeds 128 MiB diagnostic limit");
    std::istringstream in(bytes);std::string magic;in>>magic;
    require(magic=="SPACEPDHCG_QOCO_QP_V1","unsupported snapshot version");
    auto integer=[&](std::int64_t maximum=8'000'000) {
        std::string word;in>>word;std::size_t consumed=0;std::int64_t v;
        try {v=std::stoll(word,&consumed);}catch(...) {throw std::runtime_error("invalid integer");}
        require(consumed==word.size() && v>=0 && v<=maximum,"integer outside diagnostic limits");return static_cast<int>(v);
    };
    auto number=[&]() {
        std::string word;in>>word;double v{};
        const auto parsed=std::from_chars(word.data(),word.data()+word.size(),v);
        require(parsed.ec==std::errc{} && parsed.ptr==word.data()+word.size() && std::isfinite(v),
            "nonfinite, unrepresentable or malformed numeric input");return v;
    };
    Snapshot q;q.n=integer(1'000'000);q.p=integer();q.m=integer();
    const int np=integer(),na=integer(),ng=integer();q.nonnegative=integer();const int ns=integer();
    q.shifted=integer(1)!=0;
    require(q.n>0 && q.nonnegative<=q.m && ns<=(q.m-q.nonnegative)/3,"invalid dimensions or unsupported SOC size");
    require(std::int64_t(np)*2+na+ng+q.n+q.p+q.m<=limit,"expanded problem exceeds diagnostic entry limit");
    for(auto& v:q.settings_integer) v=integer();
    require(q.settings_integer[0]>0 && q.settings_integer[3]<=1,"invalid captured QOCO settings");
    for(auto& v:q.settings_float) {v=number();require(v>=0,"negative captured QOCO settings");}
    require(q.settings_float[5]>0 && q.settings_float[6]>0,"invalid captured stopping tolerances");
    auto integers=[&](int expected) {require(integer()==expected,"integer vector length mismatch");std::vector<int> v(expected);for(auto& a:v)a=integer();return v;};
    q.P={q.n,q.n,integers(q.n+1),integers(np),{}};
    q.A={q.p,q.n,integers(q.n+1),integers(na),{}};
    q.G={q.m,q.n,integers(q.n+1),integers(ng),{}};
    q.soc=integers(ns);std::int64_t rows=q.nonnegative;
    for(int n:q.soc) {require(n>=3,"SOC dimension two is unsupported by the persistent C ABI");rows+=n;}
    require(rows==q.m,"cone inventory does not cover G rows");
    auto numbers=[&](int expected) {require(integer()==expected,"numeric vector length mismatch");std::vector<double> v(expected);for(auto& a:v)a=number();return v;};
    const int count=np+na+ng+q.n+q.p+q.m;
    const auto original=numbers(count);q.translated=numbers(q.shifted?count:0);q.origin=numbers(q.shifted?q.n:0);q.offset=number();
    std::string extra;require(!(in>>extra),"trailing snapshot data");
    int begin=0;auto part=[&](int n) {std::vector<double> v(original.begin()+begin,original.begin()+begin+n);begin+=n;return v;};
    q.P.values=part(np);q.A.values=part(na);q.G.values=part(ng);q.c=part(q.n);q.b=part(q.p);q.h=part(q.m);
    topology(q.P,true);topology(q.A);topology(q.G);
    for(int j=0;j<q.n;++j) for(int k=q.P.offsets[j];k<q.P.offsets[j+1];++k)
        require(q.P.indices[k]!=j || q.P.values[k]>=0,"P has a negative diagonal and cannot be convex");
    if(!q.shifted) require(q.offset==0,"unshifted snapshot has nonzero objective offset");
    else {
        const int matrices=np+na+ng;
        require(std::equal(original.begin(),original.begin()+matrices,q.translated.begin()),"shift changed P/A/G coefficients");
        const auto o=extended(q.origin),po=multiply(q.P,o,false,true),ao=multiply(q.A,o),go=multiply(q.G,o);
        const auto magnitude=[&](Csc matrix,bool symmetric=false) {
            for(auto& v:matrix.values)v=std::abs(v);
            auto absolute=o;for(auto& v:absolute)v=std::abs(v);
            return multiply(matrix,absolute,false,symmetric);
        };
        const auto pm=magnitude(q.P,true),am=magnitude(q.A),gm=magnitude(q.G);
        auto matches=[](double actual,long double expected,long double scale) {
            // Translation can be formed with a different FP64 reduction tree.
            return std::abs(static_cast<long double>(actual)-expected)
                <=256*std::numeric_limits<double>::epsilon()*std::max(1.0L,scale);
        };
        auto check=[&](const std::vector<double>& base,const std::vector<long double>& delta,
                       const std::vector<long double>& magnitude,int start,int sign) {
            for(std::size_t i=0;i<base.size();++i) require(matches(q.translated[start+i],base[i]+sign*delta[i],
                std::abs(static_cast<long double>(base[i]))+magnitude[i]),"inconsistent shifted c/b/h");
        };
        check(q.c,po,pm,matrices,1);check(q.b,ao,am,matrices+q.n,-1);check(q.h,go,gm,matrices+q.n+q.p,-1);
        const long double offset=0.5L*dot(o,po)+dot(o,extended(q.c));
        long double scale=0;for(int j=0;j<q.n;++j) scale+=std::abs(o[j])*(0.5L*pm[j]+std::abs(q.c[j]));
        require(matches(q.offset,offset,scale),"inconsistent shifted objective offset");
    }
    q.input_sha256=sha256(bytes);return q;
}

struct Canonical {
    Csc Q,A,F;
    std::vector<double> c,lower,upper,offset,variable_lower,variable_upper;
    std::vector<int> soc_to_affine,nonnegative_to_scalar,lower_owner,upper_owner;
    struct FoldedBound {int row,variable;double coefficient,bound;};
    std::vector<FoldedBound> folded_bounds;
    std::size_t singleton_rows{},retained_nonexact_ratios{},retained_out_of_range_ratios{};
    bool fold_singleton_bounds{};
    std::vector<spacepdhcg_cuda_cone_descriptor> cones;
    std::string convexity_evidence;
};
enum class ExactRatio { representable, non_binary, out_of_range };
inline ExactRatio exact_ratio(double h,double a,double& result) {
    require(std::isfinite(h) && std::isfinite(a) && a!=0,"invalid singleton ratio");
    if(h==0) {result=h/a;return ExactRatio::representable;}
    // Every nonzero binary64 is an odd integer mantissa times a power of two.
    // Divisibility of those integers proves exactness without rounded products.
    auto parts=[](double value) {
        const auto bits=std::bit_cast<std::uint64_t>(value)&0x7fffffffffffffffULL;
        const auto exponent=static_cast<int>(bits>>52);
        std::uint64_t mantissa=(bits&0xfffffffffffffULL)|(exponent?1ULL<<52:0);
        int power=exponent?exponent-1023-52:-1074;
        const int zeros=std::countr_zero(mantissa);mantissa>>=zeros;power+=zeros;
        return std::pair{mantissa,power};
    };
    const auto [H,eh]=parts(h);const auto [A,ea]=parts(a);
    if(H%A!=0)return ExactRatio::non_binary;
    const auto quotient=H/A;const int power=eh-ea,highest=63-std::countl_zero(quotient);
    if(power < -1074 || power+highest>1023)return ExactRatio::out_of_range;
    result=h/a;
    require(std::isfinite(result) && result!=0,"proved singleton quotient was not represented");
    return ExactRatio::representable;
}
inline Csc from_columns(int rows,const std::vector<std::vector<std::pair<int,double>>>& columns) {
    Csc out;out.rows=rows;out.columns=static_cast<int>(columns.size());out.offsets.push_back(0);
    for(auto entries:columns) {
        std::sort(entries.begin(),entries.end());
        for(auto [i,v]:entries) {out.indices.push_back(i);out.values.push_back(v);}
        out.offsets.push_back(static_cast<int>(out.indices.size()));
    }
    topology(out);return out;
}
inline Canonical canonical(const Snapshot& q,bool fold_singleton_bounds=false) {
    Canonical out;out.fold_singleton_bounds=fold_singleton_bounds;
    const int start=static_cast<int>(q.P.values.size()+q.A.values.size()+q.G.values.size());
    out.c=q.shifted?std::vector<double>(q.translated.begin()+start,q.translated.begin()+start+q.n):q.c;
    const auto b=q.shifted?std::vector<double>(q.translated.begin()+start+q.n,q.translated.begin()+start+q.n+q.p):q.b;
    const auto h=q.shifted?std::vector<double>(q.translated.begin()+start+q.n+q.p,q.translated.end()):q.h;
    out.variable_lower.assign(q.n,-INFINITY);out.variable_upper.assign(q.n,INFINITY);
    out.lower_owner.assign(q.n,-1);out.upper_owner.assign(q.n,-1);
    out.nonnegative_to_scalar.assign(q.nonnegative,-1);
    std::vector<int> counts(q.nonnegative),variables(q.nonnegative);
    std::vector<double> coefficients(q.nonnegative);
    if(fold_singleton_bounds) for(int j=0;j<q.n;++j) for(int k=q.G.offsets[j];k<q.G.offsets[j+1];++k) {
        const int row=q.G.indices[k];const double value=q.G.values[k];
        if(row<q.nonnegative && value!=0) {++counts[row];variables[row]=j;coefficients[row]=value;}
    }
    out.lower=b;out.upper=b;
    for(int row=0;row<q.nonnegative;++row) {
        bool folded=false;
        if(fold_singleton_bounds && counts[row]==1) {
            ++out.singleton_rows;double bound{};
            const double a=coefficients[row];const int j=variables[row];
            const auto ratio=exact_ratio(h[row],a,bound);
            if(ratio==ExactRatio::non_binary)++out.retained_nonexact_ratios;
            else if(ratio==ExactRatio::out_of_range)++out.retained_out_of_range_ratios;
            else {
                const int index=static_cast<int>(out.folded_bounds.size());
                out.folded_bounds.push_back({row,j,a,bound});folded=true;
                auto& limit=a>0?out.variable_upper[j]:out.variable_lower[j];
                auto& owner=a>0?out.upper_owner[j]:out.lower_owner[j];
                const bool tighter=a>0?bound<limit:bound>limit;
                // Stable first-row tie break; larger coefficients limit dual magnitude.
                if(tighter || (bound==limit && (owner<0 || std::abs(a)>std::abs(out.folded_bounds[owner].coefficient)))) {
                    limit=bound;owner=index;
                }
                require(out.variable_lower[j]<=out.variable_upper[j],"inconsistent exact singleton bounds");
            }
        }
        if(!folded) {
            out.nonnegative_to_scalar[row]=static_cast<int>(out.upper.size());
            out.lower.push_back(-INFINITY);out.upper.push_back(h[row]);
        }
    }
    std::vector<std::vector<std::pair<int,double>>> Q(q.n),A(q.n),F(q.n);
    std::vector<long double> diagonal(q.n),off_diagonal(q.n);bool zero=true;
    for(int j=0;j<q.n;++j) for(int k=q.P.offsets[j];k<q.P.offsets[j+1];++k) {
        const int i=q.P.indices[k];const double v=q.P.values[k];zero=zero && v==0;
        Q[j].emplace_back(i,v);
        if(i!=j) {Q[i].emplace_back(j,v);off_diagonal[i]+=std::abs(v);off_diagonal[j]+=std::abs(v);}
        else diagonal[j]=v;
    }
    bool dominant=true;for(int j=0;j<q.n;++j) dominant=dominant && diagonal[j]>=off_diagonal[j];
    out.convexity_evidence=zero?"zero_hessian":dominant?"symmetric_diagonal_dominance":"assumed_from_capture_not_proved";
    out.Q=from_columns(q.n,Q);
    out.soc_to_affine.assign(q.m,-1);int input=q.nonnegative,output=0;
    for(int size:q.soc) {
        out.soc_to_affine[input]=output+size-1;
        for(int i=1;i<size;++i) out.soc_to_affine[input+i]=output+i-1;
        out.cones.push_back({SPACEPDHCG_CUDA_CONE_SECOND_ORDER,output,size-2,0});input+=size;output+=size;
    }
    for(int j=0;j<q.n;++j) {
        for(int k=q.A.offsets[j];k<q.A.offsets[j+1];++k) A[j].emplace_back(q.A.indices[k],q.A.values[k]);
        for(int k=q.G.offsets[j];k<q.G.offsets[j+1];++k) {
            const int row=q.G.indices[k];
            if(row<q.nonnegative) {
                if(out.nonnegative_to_scalar[row]>=0)A[j].emplace_back(out.nonnegative_to_scalar[row],q.G.values[k]);
            }
            else F[j].emplace_back(out.soc_to_affine[row],-q.G.values[k]);
        }
    }
    out.A=from_columns(static_cast<int>(out.upper.size()),A);out.F=from_columns(q.m-q.nonnegative,F);
    out.offset.resize(out.F.rows);for(int i=q.nonnegative;i<q.m;++i) out.offset[out.soc_to_affine[i]]=h[i];
    return out;
}

struct Vectors {
    std::vector<double> x,y,z,s;
    bool folded_dual_reconstruction_supported=true;
    std::size_t reconstructed_bound_duals{},off_contact_bound_normals{},off_contact_one_ulp_normals{};
    long double max_off_contact_bound_distance{};
};
inline Vectors original_vectors(const Snapshot& q,const Canonical& c,const std::vector<double>& primal,
                                const std::vector<double>& dual) {
    require(primal.size()==static_cast<std::size_t>(q.n) && dual.size()==static_cast<std::size_t>(c.A.rows+c.F.rows),"iterate dimension mismatch");
    Vectors out;out.x=primal;out.y.assign(dual.begin(),dual.begin()+q.p);out.z.resize(q.m);out.s.resize(q.m);
    if(q.shifted) for(int j=0;j<q.n;++j) out.x[j]+=q.origin[j];
    for(int i=0;i<q.m;++i) {
        if(i>=q.nonnegative)out.z[i]=-dual[c.A.rows+c.soc_to_affine[i]];
        else if(c.nonnegative_to_scalar[i]>=0)out.z[i]=dual[c.nonnegative_to_scalar[i]];
    }
    if(!c.folded_bounds.empty()) {
        const auto px=multiply(q.P,extended(out.x),false,true),aty=multiply(q.A,extended(out.y),true),gtz=multiply(q.G,extended(out.z),true);
        for(int j=0;j<q.n;++j) {
            const long double normal=-(px[j]+q.c[j]+aty[j]+gtz[j]);
            if(!std::isfinite(normal)) {out.folded_dual_reconstruction_supported=false;continue;}
            const int owner=normal>0?c.upper_owner[j]:normal<0?c.lower_owner[j]:-1;
            if(owner<0)continue;
            const auto& bound=c.folded_bounds[owner];
            // No near-active tolerance: unsupported/interior iterates retain zero
            // folded multipliers, so their actual stationarity remains visible.
            if(primal[j]!=bound.bound) {
                ++out.off_contact_bound_normals;
                if(primal[j]==std::nextafter(bound.bound,-INFINITY) || primal[j]==std::nextafter(bound.bound,INFINITY))++out.off_contact_one_ulp_normals;
                out.max_off_contact_bound_distance=std::max(out.max_off_contact_bound_distance,
                    std::abs(static_cast<long double>(primal[j])-bound.bound));
                continue;
            }
            const long double multiplier=normal/bound.coefficient;
            const double represented=static_cast<double>(multiplier);
            if(!std::isfinite(represented) || represented<=0) {
                out.z[bound.row]=std::numeric_limits<double>::quiet_NaN();
                out.folded_dual_reconstruction_supported=false;continue;
            }
            out.z[bound.row]=represented;++out.reconstructed_bound_duals;
        }
    }
    const auto gx=multiply(q.G,extended(out.x));for(int i=0;i<q.m;++i) out.s[i]=static_cast<double>(q.h[i]-gx[i]);
    return out;
}
struct Quality {
    long double primal{},dual{},gap{},objective{},dual_objective{},signed_gap{};
    long double equality_absolute{},conic_equation_absolute{},stationarity_absolute{};
    long double primal_cone_violation{},dual_cone_violation{},complementarity{};
    long double global_complementarity{},block_complementarity_normalized{},global_complementarity_normalized{};
    bool finite{},qualified{};
};
inline Quality audit(const Snapshot& q,const Vectors& v,double tolerance,double cone_tolerance) {
    Quality out;
    const auto x=extended(v.x),y=extended(v.y),z=extended(v.z),s=extended(v.s);
    require(x.size()==static_cast<std::size_t>(q.n) && y.size()==static_cast<std::size_t>(q.p)
        && z.size()==static_cast<std::size_t>(q.m) && s.size()==z.size(),"audit dimension mismatch");
    if(!std::isfinite(norm_inf(x)+norm_inf(y)+norm_inf(z)+norm_inf(s))) return out;
    auto px=multiply(q.P,x,false,true),ax=multiply(q.A,x),gx=multiply(q.G,x),aty=multiply(q.A,y,true),gtz=multiply(q.G,z,true);
    for(int i=0;i<q.p;++i) out.equality_absolute=std::max(out.equality_absolute,std::abs(ax[i]-q.b[i]));
    for(int i=0;i<q.m;++i) out.conic_equation_absolute=std::max(out.conic_equation_absolute,std::abs(gx[i]+s[i]-q.h[i]));
    for(int i=0;i<q.n;++i) out.stationarity_absolute=std::max(out.stationarity_absolute,std::abs(px[i]+q.c[i]+aty[i]+gtz[i]));
    for(int i=0;i<q.nonnegative;++i) {
        out.primal_cone_violation=std::max(out.primal_cone_violation,-s[i]);
        out.dual_cone_violation=std::max(out.dual_cone_violation,-z[i]);
        out.complementarity=std::max(out.complementarity,std::abs(s[i]*z[i]));
    }
    int start=q.nonnegative;
    for(int n:q.soc) {
        long double ss=0,zz=0,sz=0;
        for(int i=0;i<n;++i) {sz+=s[start+i]*z[start+i];if(i) {ss+=s[start+i]*s[start+i];zz+=z[start+i]*z[start+i];}}
        out.primal_cone_violation=std::max(out.primal_cone_violation,std::sqrt(ss)-s[start]);
        out.dual_cone_violation=std::max(out.dual_cone_violation,std::sqrt(zz)-z[start]);
        out.complementarity=std::max(out.complementarity,std::abs(sz));start+=n;
    }
    out.objective=0.5L*dot(x,px)+dot(extended(q.c),x);
    out.dual_objective=-0.5L*dot(x,px)-dot(extended(q.b),y)-dot(extended(q.h),z);
    out.signed_gap=out.objective-out.dual_objective;
    const long double objective_scale=std::max({1.0L,std::abs(out.objective),std::abs(out.dual_objective)});
    out.gap=std::abs(out.signed_gap)/objective_scale;
    out.global_complementarity=std::abs(dot(s,z));
    out.global_complementarity_normalized=out.global_complementarity/objective_scale;
    out.block_complementarity_normalized=out.complementarity/objective_scale;
    out.primal=std::max(out.equality_absolute,out.conic_equation_absolute)
        /(1+std::max({norm_inf(ax),norm_inf(extended(q.b)),norm_inf(gx),norm_inf(extended(q.h)),norm_inf(s)}));
    out.dual=out.stationarity_absolute/(1+std::max({norm_inf(px),norm_inf(extended(q.c)),norm_inf(aty),norm_inf(gtz)}));
    out.finite=std::isfinite(out.primal+out.dual+out.gap+out.objective+out.dual_objective
        +out.primal_cone_violation+out.dual_cone_violation+out.complementarity+out.global_complementarity);
    out.qualified=out.finite && std::max({out.primal,out.dual,out.gap,out.block_complementarity_normalized})<=tolerance
        && std::max(out.primal_cone_violation,out.dual_cone_violation)<=cone_tolerance;
    return out;
}
} // namespace spacepdhcg::snapshot
