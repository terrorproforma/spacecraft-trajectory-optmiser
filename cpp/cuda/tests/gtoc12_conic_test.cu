#include "spacepdhcg/cuda/gtoc12_conic_c_api.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
template<class T> struct Device {
    T* data{};
    explicit Device(size_t n) { check(cudaMalloc(&data, n*sizeof(T))); }
    ~Device() { cudaFree(data); }
};
}
int main() {
    for (int hold : {0, 1}) for (int k : {3, 257}) {
        const int n = k+1;
        std::vector<double> times(n), states(7*n), controls(4*n), fuel(n, 0.002);
        for (int i = 0; i < n; ++i) {
            times[i] = 0.003*i;
            states[7*i] = 2.7; states[7*i+4] = 0.61; states[7*i+6] = 0.95;
            controls[4*i] = 0.03; controls[4*i+3] = 0.04;
        }
        const double boundary[]{2.7,0,0,0,0.61,0,2.7,0,0,0,0.61,0};
        spacepdhcg_gtoc12_conic* w{};
        require(spacepdhcg_gtoc12_conic_create(k, hold, 1, 1, 0.15, 0.03,
            times.data(), boundary, fuel.data(), &w) == 0, "create");
        spacepdhcg_gtoc12_conic_dimensions d{};
        require(spacepdhcg_gtoc12_conic_get_dimensions(w, &d) == 0, "dimensions");
        require(d.variables == 25*n-8 && d.equalities == 7*k+13+(hold ? 0 : 4)
            && d.inequalities == 26*n+14*k && d.soc_count == n+2, "independent layout counts");
        spacepdhcg_gtoc12_conic_device_outputs out{};
        require(spacepdhcg_gtoc12_conic_outputs(w, &out) == 0, "device outputs");
        const size_t count = d.a_nonzeros+d.rows+d.variables+d.p_nonzeros;
        std::vector<double> actual(count), expected(count);
        std::vector<int> ao(d.variables+1), ai(d.a_nonzeros), po(d.variables+1), pi(d.p_nonzeros);
        require(spacepdhcg_gtoc12_conic_copy_topology_host(w, ao.data(), ai.data(), po.data(), pi.data()) == 0, "topology");
        require(ao.back() == d.a_nonzeros && po.back() == d.p_nonzeros, "CSC terminal offsets");
        for (int col = 0; col < d.variables; ++col)
            for (int i = ao[col]+1; i < ao[col+1]; ++i) require(ai[i] > ai[i-1], "strict sorted unique CSC");
        Device<double> ds(states.size()), du(controls.size());
        Device<spacepdhcg_gtoc12_conic_parameters> dp(1);
        cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaGraph_t graph{}; cudaGraphExec_t executable{};
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        require(spacepdhcg_gtoc12_conic_launch_device(w, ds.data, du.data, dp.data, 8, stream) == 0,
            "device dynamics plus assembly capture without host sync or allocation");
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int repeat = 0; repeat < 12; ++repeat) {
            spacepdhcg_gtoc12_conic_parameters p{0.1/(repeat+1),0.2,13.0+repeat,0.3,0.05,0.03,repeat*0.001};
            for (int i = 0; i < n; ++i) { controls[4*i] += 0.0001; states[7*i] += 0.00001; }
            check(cudaMemcpyAsync(ds.data, states.data(), states.size()*sizeof(double), cudaMemcpyHostToDevice, stream));
            check(cudaMemcpyAsync(du.data, controls.data(), controls.size()*sizeof(double), cudaMemcpyHostToDevice, stream));
            check(cudaMemcpyAsync(dp.data, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
            check(cudaGraphLaunch(executable, stream));
            int invalid = -1;
            check(cudaMemcpyAsync(actual.data(), out.a, count*sizeof(double), cudaMemcpyDeviceToHost, stream));
            check(cudaMemcpyAsync(&invalid, out.invalid, sizeof(int), cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            require(invalid == 0, "valid graph outputs");
            require(spacepdhcg_gtoc12_conic_evaluate_host(w, states.data(), controls.data(), &p, 8, expected.data()) == 0,
                "host bridge");
            for (size_t i = 0; i < count; ++i) require(actual[i] == expected[i], "graph/bridge identical values");
            // Independent objective and Hessian terms, including zero smoothness.
            const double* q = actual.data()+d.a_nonzeros+d.rows;
            const double* pv = q+d.variables;
            for (int node = 0; node < n; ++node) require(q[7*n+4*node+3] == fuel[node], "fuel weights");
            for (int i = 11*n+7*k; i < 11*n+14*k; ++i) require(q[i] == p.virtual_weight, "virtual penalty");
            for (int col = 0; col < d.variables; ++col) for (int i = po[col]; i < po[col+1]; ++i) {
                const int node = (col-7*n)/4;
                const double factor = pi[i] != col ? -2.0 : (node == 0 || node == k ? 2.0 : 4.0);
                require(pv[i] == factor*p.smoothness_weight, "smoothness Hessian");
            }
        }
        spacepdhcg_gtoc12_conic_parameters bad{0.1,0.2,13.0,0.3,0.05,0.03,-1.0};
        check(cudaMemcpyAsync(dp.data, &bad, sizeof(bad), cudaMemcpyHostToDevice, stream));
        check(cudaGraphLaunch(executable, stream));
        int invalid = 0;
        check(cudaMemcpyAsync(&invalid, out.invalid, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream)); require(invalid == 1, "invalid graph parameter surfaced");
        bad.smoothness_weight = 0.0;
        require(spacepdhcg_gtoc12_conic_evaluate_host(w, states.data(), controls.data(), &bad, 8, expected.data()) == 0, "invalid flag reset");
        require(spacepdhcg_gtoc12_conic_launch_device(w, ds.data, du.data, dp.data, 0, stream) == 1, "bad substeps");
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        check(cudaStreamDestroy(stream)); spacepdhcg_gtoc12_conic_destroy(w);
        spacepdhcg_gtoc12_conic* rejected{};
        require(spacepdhcg_gtoc12_conic_create(k, hold, 2, 0, 0.15, 0.03,
            times.data(), boundary, fuel.data(), &rejected) == 1 && !rejected, "invalid topology rejected");
    }
    std::puts("GTOC12 conic: 48 changing-input graph replays, retained topology, objective/Hessian and lifecycle PASS");
}
