#include "persistent_snapshot.hpp"
#include "spacepdhcg/cuda/common_kkt_arithmetic.hpp"

#include <filesystem>
#include <iostream>

namespace s=spacepdhcg::snapshot;
namespace {
std::string fixture(bool shifted) {
    // min x'x + x0*x1 - 4*x0 - 4*x1, x1=0, x0<=1, (1,x0,x1) in SOC.
    // x=(1,0), y=3, z=(1;1,-1,0), objective=-3.
    std::ostringstream out;
    out<<"SPACEPDHCG_QOCO_QP_V1\n2 1 4 3 1 3 1 1 "<<shifted<<"\n"
       <<"200 0 20 0\n1e-12 1e-8 1e-8 1e-8 1e-13 1e-11 1e-11 1e-5 1e-5\n"
       <<"3 0 1 3\n3 0 0 1\n3 0 0 1\n1 0\n3 0 2 3\n3 0 2 3\n1 3\n"
       <<"14 2 1 2 1 1 -1 -1 -4 -4 0 1 1 0 0\n";
    auto bytes=out.str();
    bytes+=shifted?"14 2 1 2 1 1 -1 -1 -1 -4 1 -1 1 2 -1\n2 2 -1\n-1\n":"0\n0\n0\n";
    return bytes;
}
std::string bounds_fixture(double objective,const std::vector<double>& a,const std::vector<double>& h) {
    s::require(a.size()==h.size(),"invalid bounds fixture");const auto m=a.size();
    std::ostringstream out;out<<std::setprecision(17);
    out<<"SPACEPDHCG_QOCO_QP_V1\n1 0 "<<m<<" 1 0 "<<m<<' '<<m<<" 0 0\n"
       <<"200 0 20 0\n1e-12 1e-8 1e-8 1e-8 1e-13 1e-11 1e-11 1e-5 1e-5\n"
       <<"2 0 1\n1 0\n2 0 0\n0\n2 0 "<<m<<'\n'<<m;
    for(std::size_t i=0;i<m;++i)out<<' '<<i;
    out<<"\n0\n"<<2+2*m<<" 0";
    for(double v:a)out<<' '<<v;
    out<<' '<<objective;for(double v:h)out<<' '<<v;
    out<<"\n0\n0\n0\n";return out.str();
}
std::string point_fixture(const s::Snapshot& q,const s::Vectors& v,const std::string& coordinates="original") {
    std::ostringstream out;out<<std::setprecision(17);
    out<<"SPACEPDHCG_QOCO_INITIAL_POINT_V1\nsnapshot_sha256 "<<q.input_sha256<<"\ncoordinates "<<coordinates<<'\n';
    auto vector=[&](const char* name,const std::vector<double>& values) {
        out<<name<<' '<<values.size();for(double value:values)out<<' '<<value;out<<'\n';
    };
    vector("x",v.x);vector("y",v.y);vector("z",v.z);vector("s",v.s);return out.str();
}
void equal(const std::vector<double>& a,const std::vector<double>& b) {s::require(a==b,"array mismatch");}
void near(long double a,long double b) {s::require(std::isfinite(a) && std::abs(a-b)<1e-15L,"analytic scalar mismatch");}
void reject(std::string bytes,const std::string& before,const std::string& after) {
    const auto position=bytes.find(before);s::require(position!=std::string::npos,"invalid mutation fixture");
    bytes.replace(position,before.size(),after);
    bool rejected=false;try {static_cast<void>(s::read(bytes));}catch(const std::exception&) {rejected=true;}
    s::require(rejected,"malformed snapshot was accepted");
}
void reject_point(const s::Snapshot& q,const s::Canonical& c,std::string bytes,const std::string& before,const std::string& after) {
    const auto position=bytes.find(before);s::require(position!=std::string::npos,"invalid point mutation fixture");
    bytes.replace(position,before.size(),after);
    bool rejected=false;try {static_cast<void>(s::initial_point(bytes,q,c));}catch(const std::exception&) {rejected=true;}
    s::require(rejected,"malformed or unqualified initial point was accepted");
}
void checks() {
    // Independent cancellation identities exercise arithmetic used by the GPU
    // gate. Ordinary FP64 loses these terms before a final tolerance comparison.
    namespace ck=spacepdhcg::cuda::common_kkt;
    const double epsilon=std::ldexp(1.0,-27);
    const auto cancelled_product=ck::add(ck::product(1+epsilon,1-epsilon),{-1,0});
    s::require(ck::value(cancelled_product)==-std::ldexp(1.0,-54),"compensated product lost the exact remainder");
    s::require(ck::value(ck::add(ck::add({1e16,0},{1,0}),{-1e16,0}))==1,"sparse reduction lost a cancelled unit");
    const auto norm=ck::square_root(ck::add(ck::product(3,3),ck::product(4,4)));
    s::require(ck::value(norm)==5 && ck::finite(norm),"compensated SOC norm identity failed");
    s::require(!ck::finite(ck::product(std::numeric_limits<double>::max(),2)),"overflow was hidden by compensated arithmetic");
    const double magnitude=std::ldexp(1.0,40);
    const auto grouped=ck::add({magnitude+1,0},{-magnitude,0});
    const double separate_denominator=1+std::max(magnitude+1,magnitude);
    s::require(ck::value(grouped)/separate_denominator<1e-9 && ck::value(grouped)/(1+ck::absolute(grouped))>1e-9,
        "equality and original G normalization fixture failed");
    const auto conic_sum=ck::add({magnitude,0},{-magnitude,0});
    s::require(1/(1+std::max(1.0,ck::absolute(conic_sum)))>1e-9 && 1/(1+magnitude)<1e-9,
        "combined nonnegative/SOC normalization fixture failed");
    s::require(s::sha256("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA empty");
    s::require(s::sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA abc");
    s::require(s::sha256(std::string(1000000,'a'))=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","SHA multiblock");
    for(bool shifted:{false,true}) {
        const auto q=s::read(fixture(shifted));const auto c=s::canonical(q);
        s::require(c.Q.offsets==std::vector<int>({0,2,4}) && c.Q.indices==std::vector<int>({0,1,0,1}),"symmetric P pattern");
        equal(c.Q.values,{2,1,1,2});
        s::require(c.A.offsets==std::vector<int>({0,1,2}) && c.A.indices==std::vector<int>({1,0}),"mixed scalar rows");
        equal(c.A.values,{1,1});equal(c.F.values,{1,1});
        s::require(c.F.indices==std::vector<int>({0,1}) && c.cones.size()==1
            && c.cones[0].start==0 && c.cones[0].vector_dimension==1,"SOC dimension/permutation");
        equal(c.c,shifted?std::vector<double>{-1,-4}:std::vector<double>{-4,-4});
        equal(c.upper,shifted?std::vector<double>{1,-1}:std::vector<double>{0,1});
        s::require(c.lower[0]==(shifted?1:0) && c.lower[1]==-INFINITY,"scalar lower bounds");
        equal(c.offset,shifted?std::vector<double>{2,-1,1}:std::vector<double>{0,0,1});
        const auto v=s::original_vectors(q,c,shifted?std::vector<double>{-1,1}:std::vector<double>{1,0},{3,1,1,0,-1});
        equal(v.x,{1,0});equal(v.y,{3});equal(v.z,{1,1,-1,0});equal(v.s,{0,1,1,0});
        const auto audit=s::audit(q,v,1e-9,1e-8);
        s::require(audit.qualified && audit.finite,"exact KKT fixture failed");near(audit.objective,-3);near(audit.dual_objective,-3);near(audit.complementarity,0);
        auto invalid=v;invalid.z[2]=1; s::require(!s::audit(q,invalid,1e-9,1e-8).qualified,"wrong SOC dual sign passed");
        invalid=v;invalid.y[0]=-3;s::require(!s::audit(q,invalid,1e-9,1e-8).qualified,"wrong equality dual sign passed");
        invalid=v;invalid.x[0]=2;s::require(!s::audit(q,invalid,1e-9,1e-8).qualified,"infeasible primal passed");
        invalid=v;invalid.s[0]=0.1;s::require(!s::audit(q,invalid,1e-9,1e-8).qualified,"inconsistent supplied slack passed");
        invalid=v;invalid.x[0]=std::numeric_limits<double>::quiet_NaN();s::require(!s::audit(q,invalid,1e-9,1e-8).finite,"nonfinite output passed");
        invalid=v;invalid.z[0]=-1;s::require(!s::audit(q,invalid,1e-9,1e-8).qualified,"negative cone dual passed");
        const auto folded=s::canonical(q,true);
        s::require(folded.folded_bounds.size()==1 && folded.A.rows==1,"mixed fixture scalar fold");
        const auto fv=s::original_vectors(q,folded,shifted?std::vector<double>{-1,1}:std::vector<double>{1,0},{3,1,0,-1});
        equal(fv.z,{1,1,-1,0});equal(fv.x,v.x);equal(fv.s,v.s);
        s::require(fv.reconstructed_bound_duals==1 && s::audit(q,fv,1e-9,1e-8).qualified,"folded shifted KKT failed");
        for(const std::string coordinates:{"original","translated"}) {
            auto encoded=v;if(shifted && coordinates=="translated")encoded.x={-1,1};
            const auto bytes=point_fixture(q,encoded,coordinates);
            const auto point=s::initial_point(bytes,q,c),folded_point=s::initial_point(bytes,q,folded);
            equal(point.primal,shifted?std::vector<double>{-1,1}:std::vector<double>{1,0});
            equal(point.dual,{3,1,1,0,-1});equal(folded_point.dual,{3,1,0,-1});
            equal(point.reference.x,v.x);equal(folded_point.reference.z,v.z);
            s::require(point.supplied_audit.qualified && point.roundtrip_audit.qualified && point.reconstructed_audit.qualified
                && folded_point.roundtrip_audit.qualified && folded_point.reconstructed_audit.qualified,"initial-point coordinate/dual roundtrip failed");
            s::require(point.file_sha256==s::sha256(bytes),"initial point identity missing");
        }
        const auto point_bytes=point_fixture(q,{{1,0},{3},{1,1,-1,0},{0,1,1,0}});
        reject_point(q,c,point_bytes,"SPACEPDHCG_QOCO_INITIAL_POINT_V1","SPACEPDHCG_QOCO_INITIAL_POINT_V2");
        reject_point(q,c,point_bytes,q.input_sha256,std::string(64,'0'));
        reject_point(q,c,point_bytes,"coordinates original","coordinates unknown");
        reject_point(q,c,point_bytes,"x 2 1 0","x 3 1 0");
        reject_point(q,c,point_bytes,"x 2 1 0","x 2 nan 0");
        reject_point(q,c,point_bytes,"y 1 3","y 1 -3");
        reject_point(q,c,point_bytes,"z 4 1 1 -1 0","z 4 -1 1 -1 0");
        reject_point(q,c,point_bytes,"z 4 1 1 -1 0","z 4 1 1 1 0");
        reject_point(q,c,point_bytes,"s 4 0 1 1 0","s 4 0 -1 1 0");
        reject_point(q,c,point_bytes,"s 4 0 1 1 0","s 4 0.1 1 1 0");
        reject_point(q,c,point_bytes,"s 4 0 1 1 0","s 4 0 1 1");
        reject_point(q,c,point_bytes,"s 4 0 1 1 0","s 4 0 1 1 0 trailing");
    }
    const auto q=fixture(false);
    reject(q,"SPACEPDHCG_QOCO_QP_V1","SPACEPDHCG_QOCO_QP_V2");
    reject(q,"2 1 4 3 1 3 1 1 0","1000000000 1 4 3 1 3 1 1 0");
    reject(q,"3 0 1 3","3 0 4 3");
    reject(q,"3 0 0 1\n3 0 0 1","3 1 0 1\n3 0 0 1"); // lower P
    reject(q,"3 0 0 1\n3 0 0 1","3 0 0 0\n3 0 0 1"); // duplicate P
    reject(q,"1 3\n14","1 2\n14");
    reject(q,"14 2 1 2","14 nan 1 2");
    reject(q,"14 2 1 2","14 -2 1 2");
    reject(q,"14 2 1 2","15 2 1 2");
    reject(q,"0\n0\n0\n","0\n0\n1\n");
    reject(q,"0\n0\n0\n","0\n0\n");
    reject(q,"0\n0\n0\n","0\n0\n0\ntrailing\n");
    reject(fixture(true),"14 2 1 2 1 1 -1 -1 -1 -4","14 2 1 2 1 1 -1 -1 -2 -4");
    reject(fixture(true),"2 2 -1\n-1\n","2 2 -1\n0\n");
    reject(fixture(true),"14 2 1 2 1 1 -1 -1 -1 -4","14 2 2 2 1 1 -1 -1 -1 -4");
    // A finite, zero-Hessian unconstrained LP is legal; zero and empty arrays survive.
    const auto empty=s::read("SPACEPDHCG_QOCO_QP_V1\n1 0 0 1 0 0 0 0 0\n200 0 20 0\n1e-12 1e-8 1e-8 1e-8 1e-13 1e-11 1e-11 1e-5 1e-5\n2 0 1\n1 0\n2 0 0\n0\n2 0 0\n0\n0\n2 0 0\n0\n0\n0\n");
    const auto ce=s::canonical(empty);s::require(ce.Q.values==std::vector<double>({0}) && ce.A.values.empty() && ce.F.values.empty(),"structural zero/empty rows lost");
    s::require(s::audit(empty,s::original_vectors(empty,ce,{0},{}),1e-9,1e-8).qualified,"zero LP failed");
    auto psd_bytes=fixture(false);
    psd_bytes.replace(psd_bytes.find("14 2 1 2"),8,"14 1 2 4");
    s::require(s::canonical(s::read(psd_bytes)).convexity_evidence=="assumed_from_capture_not_proved",
        "valid non-diagonally-dominant PSD was rejected or incorrectly proved");
    // Adjacent SOCs of different sizes exercise nonzero descriptor starts.
    const auto two=s::read("SPACEPDHCG_QOCO_QP_V1\n2 0 7 0 0 0 0 2 0\n200 0 20 0\n1e-12 1e-8 1e-8 1e-8 1e-13 1e-11 1e-11 1e-5 1e-5\n3 0 0 0\n0\n3 0 0 0\n0\n3 0 0 0\n0\n2 3 4\n9 0 0 1 0 0 2 0 0 0\n0\n0\n0\n");
    const auto ct=s::canonical(two);
    s::require(ct.soc_to_affine==std::vector<int>({2,0,1,6,3,4,5}) && ct.cones.size()==2
        && ct.cones[1].start==3 && ct.cones[1].vector_dimension==2,"adjacent SOC mapping");
    equal(ct.offset,{0,0,1,0,0,0,2});
    equal(s::original_vectors(two,ct,{0,0},{1,2,3,4,5,6,7}).z,{-3,-1,-2,-7,-4,-5,-6});
    // Small permitted cone violations plus huge duals must not hide cancellation
    // between nonzero block complementarity products in the global duality gap.
    const auto cancellation=s::read("SPACEPDHCG_QOCO_QP_V1\n1 0 2 0 0 0 2 0 0\n200 0 20 0\n1e-12 1e-8 1e-8 1e-8 1e-13 1e-11 1e-11 1e-5 1e-5\n2 0 0\n0\n2 0 0\n0\n2 0 0\n0\n0\n3 0 -1e-8 1\n0\n0\n0\n");
    const auto cancelled=s::audit(cancellation,{{0},{},{1e12,1e4},{-1e-8,1}},1e-9,1e-8);
    s::require(cancelled.gap<1e-9 && cancelled.global_complementarity_normalized<1e-9
        && cancelled.block_complementarity_normalized>=1e4 && !cancelled.qualified,"complementarity cancellation passed");
    // Exact bound intersection and deterministic dual ownership.
    const auto upper=s::read(bounds_fixture(-3,{1,2,-1,1,2},{1,2,0,2,2}));
    const auto cu=s::canonical(upper,true);
    s::require(cu.folded_bounds.size()==5 && cu.A.rows==0 && cu.variable_lower[0]==0 && cu.variable_upper[0]==1,"bound intersection");
    const auto vu=s::original_vectors(upper,cu,{1},{});
    equal(vu.z,{0,1.5,0,0,0});s::require(s::audit(upper,vu,1e-9,1e-8).qualified,"upper/duplicate bound KKT");
    const auto inside=s::original_vectors(upper,cu,{std::nextafter(1.0,0.0)},{});
    equal(inside.z,{0,0,0,0,0});s::require(inside.off_contact_bound_normals==1 && inside.off_contact_one_ulp_normals==1
        && !s::audit(upper,inside,1e-9,1e-8).qualified,"interior iterate gained invented active dual");
    auto weak=inside;weak.z=vu.z;
    const auto weak_point=s::initial_point(point_fixture(upper,weak),upper,cu);
    s::require(weak_point.supplied_audit.qualified && weak_point.roundtrip_audit.qualified
        && !weak_point.reconstructed_audit.qualified && weak_point.dual.empty(),"weak-contact reference point rejected or snapped");
    const auto lossy=s::read("SPACEPDHCG_QOCO_QP_V1\n1 0 1 1 0 1 1 0 1\n200 0 20 0\n1e-12 1e-8 1e-8 1e-8 1e-13 1e-11 1e-11 1e-5 1e-5\n2 0 1\n1 0\n2 0 0\n0\n2 0 1\n1 0\n0\n4 0 1 -3 1\n4 0 1 -3 -1e20\n1 1e20\n-3e20\n");
    const s::Vectors exact_lossy{{1},{},{3},{0}};
    s::require(s::audit(lossy,exact_lossy,1e-9,1e-8).qualified,"lossy-coordinate source point must qualify");
    bool lost=false;try {static_cast<void>(s::initial_point(point_fixture(lossy,exact_lossy),lossy,s::canonical(lossy)));}
    catch(const std::exception&) {lost=true;}s::require(lost,"FP64 coordinate loss silently changed qualified seed");
    const auto lower=s::read(bounds_fixture(2,{1,-1},{1,0}));
    const auto vl=s::original_vectors(lower,s::canonical(lower,true),{0},{});
    equal(vl.z,{0,2});s::require(s::audit(lower,vl,1e-9,1e-8).qualified,"lower bound KKT");
    for(double objective:{-4,6}) {
        const auto fixed=s::read(bounds_fixture(objective,{2,-3},{2,-3}));
        const auto vf=s::original_vectors(fixed,s::canonical(fixed,true),{1},{});
        equal(vf.z,objective<0?std::vector<double>{2,0}:std::vector<double>{0,2});
        s::require(s::audit(fixed,vf,1e-9,1e-8).qualified,"fixed-variable normal sign");
    }
    bool conflict=false;try {static_cast<void>(s::canonical(s::read(bounds_fixture(0,{1,-1},{1,-2})),true));}
    catch(const std::exception&) {conflict=true;}s::require(conflict,"inconsistent exact bounds silently clamped");
    const auto zero_row=s::canonical(s::read(bounds_fixture(0,{0,1},{-1,2})),true);
    s::require(zero_row.A.rows==1 && zero_row.A.values==std::vector<double>{0} && zero_row.upper[0]==-1,"constant infeasible row or structural zero removed");
    double ratio{};
    s::require(s::exact_ratio(6,3,ratio)==s::ExactRatio::representable && ratio==2,"exact non-unit divisor");
    s::require(s::exact_ratio(1,3,ratio)==s::ExactRatio::non_binary,"non-binary quotient folded");
    const double tiny=std::numeric_limits<double>::denorm_min();
    s::require(s::exact_ratio(tiny,1,ratio)==s::ExactRatio::representable && ratio==tiny,"exact subnormal quotient");
    s::require(s::exact_ratio(0,tiny,ratio)==s::ExactRatio::representable && ratio==0,"tiny coefficient ignored");
    s::require(s::exact_ratio(1,tiny,ratio)==s::ExactRatio::out_of_range && s::exact_ratio(tiny,2,ratio)==s::ExactRatio::out_of_range,"quotient overflow/underflow hidden");
    const auto underflow=s::canonical(s::read(bounds_fixture(0,{2},{tiny})),true);
    const auto overflow=s::canonical(s::read(bounds_fixture(0,{tiny},{1})),true);
    s::require(underflow.A.rows==1 && underflow.retained_out_of_range_ratios==1
        && overflow.A.rows==1 && overflow.retained_out_of_range_ratios==1,"out-of-range singleton row removed");
    const auto nonexact=s::canonical(s::read(bounds_fixture(0,{3},{1})),true);
    s::require(nonexact.folded_bounds.empty() && nonexact.A.rows==1 && nonexact.retained_nonexact_ratios==1,"nonexact singleton not retained");
    // Retain a tiny second coefficient exactly; this row is not a singleton.
    auto two_coefficients=s::read(bounds_fixture(0,{1},{1}));two_coefficients.n=2;two_coefficients.c={0,0};
    two_coefficients.P={2,2,{0,1,1},{0},{0}};two_coefficients.A={0,2,{0,0,0},{},{}};
    two_coefficients.G={1,2,{0,1,2},{0,0},{1,tiny}};
    const auto two_columns=s::canonical(two_coefficients,true);
    s::require(two_columns.folded_bounds.empty() && two_columns.A.values==std::vector<double>({1,tiny}),"tiny nonzero changed topology");
    const auto huge_dual=s::read(bounds_fixture(-1,{tiny},{0}));
    const auto vh=s::original_vectors(huge_dual,s::canonical(huge_dual,true),{0},{});
    s::require(!vh.folded_dual_reconstruction_supported && !s::audit(huge_dual,vh,1e-9,1e-8).qualified,"unrepresentable bound dual silently clamped");
}
} // namespace
int main(int argc,char** argv) try {
    checks();
    if(argc==3 && std::string(argv[1])=="--write-fixtures") {
        const std::filesystem::path root(argv[2]);
        s::require(!std::filesystem::exists(root),"fixture output already exists");
        std::filesystem::create_directories(root);
        for(bool shifted:{false,true}) {
            const std::string name=shifted?"mixed-shifted":"mixed";
            std::ofstream out(root/(name+".txt"));out<<fixture(shifted);s::require(bool(out),"fixture write failed");
            const auto q=s::read(fixture(shifted));const s::Vectors exact{{1,0},{3},{1,1,-1,0},{0,1,1,0}};
            std::ofstream point(root/(name+"-initial-original.txt"));point<<point_fixture(q,exact);s::require(bool(point),"initial point write failed");
            auto translated=exact;if(shifted)translated.x={-1,1};
            std::ofstream local(root/(name+"-initial-translated.txt"));local<<point_fixture(q,translated,"translated");s::require(bool(local),"translated point write failed");
        }
        std::ofstream upper(root/"bounds-duplicate.txt");upper<<bounds_fixture(-3,{1,2,-1,1,2},{1,2,0,2,2});s::require(bool(upper),"bound fixture write failed");
        const auto uq=s::read(bounds_fixture(-3,{1,2,-1,1,2},{1,2,0,2,2}));
        const auto weak=s::original_vectors(uq,s::canonical(uq),{std::nextafter(1.0,0.0)},{0,1.5,0,0,0});
        std::ofstream weak_point(root/"bounds-weak-initial.txt");weak_point<<point_fixture(uq,weak);s::require(bool(weak_point),"weak-contact point write failed");
        std::ofstream fixed(root/"bounds-fixed.txt");fixed<<bounds_fixture(6,{2,-3},{2,-3});s::require(bool(fixed),"fixed fixture write failed");
    } else s::require(argc==1,"usage: persistent_snapshot_conversion_test [--write-fixtures NEW_DIRECTORY]");
    std::cout<<"persistent snapshot conversion and independent analytic KKT checks passed (CPU only)\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
