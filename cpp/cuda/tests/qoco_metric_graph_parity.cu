// Reuse the isolated device fixture to compare captured/uncaptured operators
// on identical changing inputs, independently of factorization variability.
#define main metric_benchmark_unused_main
#include "qoco_metric_graph_benchmark.cu"
#undef main
#include <cstring>

int main() {
    for (int n : {17,4103,100000}) {
        Fixture fixture(n);
        require(qoco_gpu_begin_reduction_scope()==0,"parity scope");
        double maximum_absolute=0, maximum_relative=0;
        int identical=0;
        for (int repeat=0;repeat<30;++repeat) {
            std::vector<double> input(n);
            for (int i=0;i<n;++i) input[i]=std::sin(i*.13+repeat*.37)*(.1+repeat*.03);
            require(cudaMemcpy(fixture.work.x->d_data,input.data(),n*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess,"update parity input");
            double baseline[8]{},graph[8]{};
            require(setenv("SPACEPDHCG_TEST_QOCO_METRIC_GRAPH_DISABLE","1",1)==0,"uncaptured parity");
            qoco_gpu_iteration_metrics(&fixture.solver,baseline);
            require(setenv("SPACEPDHCG_TEST_QOCO_METRIC_GRAPH_DISABLE","0",1)==0,"graph parity");
            qoco_gpu_iteration_metrics(&fixture.solver,graph);
            if (repeat==0) { qoco_gpu_iteration_metrics(&fixture.solver,graph); }
            identical+=std::memcmp(baseline,graph,sizeof(graph))==0;
            for (int j=0;j<8;++j) {
                const double difference=std::abs(baseline[j]-graph[j]);
                maximum_absolute=std::max(maximum_absolute,difference);
                maximum_relative=std::max(maximum_relative,difference/std::max(1.0,std::abs(baseline[j])));
                require(std::isfinite(graph[j]) && difference<=2e-12*std::max(1.0,std::abs(baseline[j])),"captured metric accuracy");
            }
        }
        qoco_gpu_end_reduction_scope();
        std::printf("{\"n\":%d,\"inputs\":30,\"bitwise_identical\":%d,\"max_absolute_difference\":%.17g,\"max_relative_difference\":%.17g,\"passed\":true}\n",
            n,identical,maximum_absolute,maximum_relative);
    }
}
