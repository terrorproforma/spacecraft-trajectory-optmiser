// Isolate repeated stopping calculations in one library. Graph capture cost is
// reported separately; both strategies retain the scalar result download.
#include "qoco.h"
#include "../algebra/cuda/cuda_types.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
int qoco_gpu_begin_reduction_scope();
void qoco_gpu_end_reduction_scope();
void qoco_gpu_iteration_metrics(QOCOSolver*, double*);
void qoco_gpu_metric_graph_stats(unsigned long long*);
}
namespace {
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
struct Fixture {
    QOCOSolver solver{}; QOCOWorkspace work{}; QOCOProblemData data{};
    QOCOScaling scaling{}; QOCOSettings settings{};
    std::vector<QOCOVectorf*> vectors;
    QOCOVectorf* vector(int n, double base) {
        std::vector<double> values(n);
        for (int i=0;i<n;++i) values[i]=base + .01 * std::sin(i*.13);
        auto* result=new_qoco_vectorf(values.data(),n); vectors.push_back(result); return result;
    }
    QOCOMatrix* matrix(int rows, int cols, double base, bool diagonal) {
        std::vector<int> p{0},i; std::vector<double>x;
        for (int col=0;col<cols;++col) {
            i.push_back(diagonal?col:col%rows); x.push_back(base+.01*std::sin(col*.11));
            p.push_back(static_cast<int>(i.size()));
        }
        QOCOCscMatrix c{rows,cols,static_cast<int>(x.size()),i.data(),p.data(),x.data()};
        return new_qoco_matrix(&c);
    }
    explicit Fixture(int n) {
        set_default_settings(&settings); settings.kkt_static_reg_P=1e-8;
        solver.work=&work; solver.settings=&settings; work.data=&data; work.scaling=&scaling;
        data.n=n; data.p=5; data.m=9;
        data.P=matrix(n,n,1.0,true); data.A=matrix(5,n,.2,false); data.G=matrix(9,n,.3,false);
        data.c=vector(n,.2); data.b=vector(5,.3); data.h=vector(9,.5);
        work.x=vector(n,.4); work.y=vector(5,.3); work.z=vector(9,.2); work.s=vector(9,1.2);
        work.xbuff=vector(n,0); work.ybuff=vector(5,0);
        work.ubuff1=vector(9,0); work.ubuff2=vector(9,0); work.ubuff3=vector(9,0);
        work.kktres=vector(n+14,.07);
        scaling.Dinvruiz=vector(n,.8); scaling.Einvruiz=vector(5,.7);
        scaling.Finvruiz=vector(9,.6); scaling.Fruiz=vector(9,1.6);
        scaling.k=2.0; scaling.kinv=.5;
    }
    ~Fixture() {
        for (auto* v:vectors) free_qoco_vectorf(v);
        free_qoco_matrix(data.P); free_qoco_matrix(data.A); free_qoco_matrix(data.G);
    }
};
using Clock=std::chrono::steady_clock;
void run(int n) {
    Fixture fixture(n);
    double reference[8]{};
    require(setenv("SPACEPDHCG_TEST_QOCO_METRIC_GRAPH_DISABLE","1",1)==0,"disable graph");
    qoco_gpu_iteration_metrics(&fixture.solver,reference);
    for (int repeat=0;repeat<9;++repeat) for (int lane=0;lane<2;++lane) {
        const bool graph=(repeat+lane)%2;
        require(setenv("SPACEPDHCG_TEST_QOCO_METRIC_GRAPH_DISABLE",graph?"0":"1",1)==0,"select strategy");
        require(qoco_gpu_begin_reduction_scope()==0,"scope");
        double output[8]{};
        auto start=Clock::now();
        for (int i=0;i<3;++i) qoco_gpu_iteration_metrics(&fixture.solver,output);
        const double prime=std::chrono::duration<double>(Clock::now()-start).count();
        unsigned long long before[6]{},after[6]{}; qoco_gpu_metric_graph_stats(before);
        start=Clock::now();
        for (int i=0;i<50;++i) qoco_gpu_iteration_metrics(&fixture.solver,output);
        const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
        qoco_gpu_metric_graph_stats(after);
        qoco_gpu_end_reduction_scope();
        require(graph ? after[1]-before[1]==50 : after[1]==before[1],"selected replay behavior");
        for (int i=0;i<8;++i) require(std::isfinite(output[i]) &&
            std::abs(output[i]-reference[i])<=2e-12*std::max(1.0,std::abs(reference[i])),"metric parity");
        std::printf("{\"n\":%d,\"repeat\":%d,\"warmup\":%s,\"graph\":%s,\"calls\":50,"
                    "\"priming_seconds\":%.17g,\"seconds_per_call\":%.17g,\"passed\":true}\n",
                    n,repeat,repeat<2?"true":"false",graph?"true":"false",prime,elapsed/50);
    }
}
}
int main() {
    run(17); run(4103); run(100000);
    require(cudaDeviceSynchronize()==cudaSuccess,"completion");
    return 0;
}
