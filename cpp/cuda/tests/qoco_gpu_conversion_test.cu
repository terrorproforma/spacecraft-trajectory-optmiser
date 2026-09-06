#include "../internal/native_qoco_gpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
void check(cudaError_t status) {
    if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); }
}
__global__ void consume_guard(const int* flags, const double* guarded, double* result) {
    for (int i=0;i<9;++i) result[i]=guarded[i];
    result[9]=*flags;
}
void run_case(int outputs) {
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> host[9]{{2, 3, 3, 4}, std::vector<double>(131073, 0.25), {1, 2, 3, 4},
        {5, 6, 7}, {-inf, -inf, -2, -3, 4}, {inf, 2, inf, 3, 4}, {7, 8, 9, 10},
        {-inf, -inf, -2, -3, 4}, {inf, 2, inf, 3, 4}};
    int kinds[]{0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    QocoConversionPair pairs[]{{1, 2}};
    std::vector<int> offsets{0};
    std::vector<QocoConversionTerm> terms;
    for (int i = 0; i < outputs; ++i) {
        switch (i % 7) {
        case 0: terms.push_back({1, 131072, -1, 2, 0}); terms.push_back({1, 0, -1, -1, 0}); break;
        case 1: terms.push_back({6, 2, 3, 1 / std::sqrt(2.0), 1}); break;
        case 2: terms.push_back({6, 2, 3, 1 / std::sqrt(2.0), -1}); break;
        case 3: terms.push_back({-1, 0, -1, -1, 0}); break;
        case 4: terms.push_back({3, 1, -1, 1, 0}); break;
        case 5: break; // structurally empty row remains zero
        case 6:
            terms.push_back({-1, 0, -1, 1e16, 0});
            terms.push_back({-1, 0, -1, 1, 0});
            terms.push_back({-1, 0, -1, -1e16, 0}); break;
        }
        offsets.push_back(static_cast<int>(terms.size()));
    }
    QocoConversionInputs input{};
    QocoConversionPlan plan{};
    double* device[9]{};
    for (int i = 0; i < 9; ++i) {
        plan.input_counts[i] = static_cast<int>(host[i].size());
        check(cudaMalloc(&device[i], host[i].size() * sizeof(double)));
        check(cudaMemcpyAsync(device[i], host[i].data(), host[i].size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        input.arrays[i] = device[i];
    }
    plan.outputs = outputs; plan.terms = static_cast<int>(terms.size());
    plan.offsets = offsets.data(); plan.entries = terms.data(); plan.bound_types = kinds;
    plan.symmetry_pairs = 1; plan.symmetry = pairs;
    QocoGpuConversion* conversion{}; check(qoco_gpu_conversion_create(plan, stream, &conversion));
    const auto before = qoco_gpu_conversion_memory(conversion);
    std::vector<double> result(outputs), retained(outputs);
    int invalid{};
    const auto run = [&] { check(qoco_gpu_conversion_run(conversion, input, result.data(), &invalid, stream)); };
    const auto set = [&](int array, int index, double value) {
        host[array][index] = value;
        check(cudaMemcpyAsync(device[array] + index, &host[array][index], sizeof(double), cudaMemcpyHostToDevice, stream));
    };
    for (int update = 0; update < 4; ++update) {
        set(1, 131072, 0.25 + update * 0.125);
        set(4, 3, -3 - update); // same bound type, changed numeric value
        set(7, 4, 4 + update); set(8, 4, 4 + update); // equality remains equality
        run(); require(invalid == 0, "valid numeric update rejected");
        for (int i = 0; i < outputs; ++i) {
            long double expected = 0;
            switch (i % 7) {
            case 0: expected = 2 * static_cast<long double>(host[1].back()) - host[1][0]; break;
            case 1: expected = 19 / std::sqrt(2.0L); break;
            case 2: expected = -1 / std::sqrt(2.0L); break;
            case 3: expected = -1; break;
            case 4: expected = 6; break;
            case 5: case 6: expected = 0; break; // fixed duplicate accumulation order for cancellation
            }
            require(std::isfinite(result[i]) && std::abs(result[i] - expected) < 2e-14L,
                "conversion differs from independent arithmetic");
        }
        if (outputs) {
            check(cudaMemcpyAsync(retained.data(), qoco_gpu_conversion_values(conversion), outputs * sizeof(double), cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream)); require(retained == result, "resident conversion values differ");
        }
    }
    set(4, 1, 0); run(); require(invalid & 1, "new finite bound must invalidate compiled layout"); set(4, 1, -inf);
    set(8, 4, host[7][4] + 1); run(); require(invalid & 1, "equality-to-inequality mutation missed"); set(8, 4, host[7][4]);
    set(0, 2, 3.001); run(); require(invalid & 4, "asymmetric quadratic missed"); set(0, 2, 3);
    set(2, 3, std::numeric_limits<double>::quiet_NaN()); run(); require(invalid & 2, "unused NaN input must reject"); set(2, 3, 4);
    set(4, 0, std::numeric_limits<double>::quiet_NaN()); run(); require(invalid & 2, "NaN bound must reject"); set(4, 0, -inf);
    set(1, 131072, std::numeric_limits<double>::max()); run();
    if (outputs) require(invalid & 2, "nonfinite transformed coefficient must reject");
    set(1, 131072, 0.25); run(); require(invalid == 0, "rejected update poisoned retained workspace");
    const auto transfers = qoco_gpu_conversion_transfers(conversion);
    check(qoco_gpu_conversion_run(conversion, input, nullptr, &invalid, stream));
    const auto resident_transfers = qoco_gpu_conversion_transfers(conversion);
    require(invalid == 0 && resident_transfers.d2h_count == transfers.d2h_count + 1
        && resident_transfers.d2h_bytes == transfers.d2h_bytes + sizeof(int),
        "device-only conversion downloads only validation status");

    // Capture the complete validation -> numeric guard -> device consumer chain.
    // Exercise every combination of canonical errors and an upstream scale error.
    // A mutation at the topology grid-stride tail must reach the consumer too.
    std::vector<int> topology_values(131073, 7);
    int* topology_device{}; check(cudaMalloc(&topology_device, topology_values.size()*sizeof(int)));
    check(cudaMemcpyAsync(topology_device,topology_values.data(),topology_values.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
    QocoTopologyInput topology_host{}, topology_input{};
    topology_host.counts[5]=topology_input.counts[5]=static_cast<int>(topology_values.size());
    topology_host.arrays[5]=topology_values.data(); topology_input.arrays[5]=topology_device;
    QocoGpuTopology* topology{}; check(qoco_gpu_topology_create(topology_host,stream,&topology));
    const auto topology_memory=qoco_gpu_topology_memory(topology);
    double numeric_host[]{2,0.5,3,4,5,6,7,8,0}, observed[10]{};
    double *numeric_device{}, *consumer_device{};
    check(cudaMalloc(&numeric_device,sizeof(numeric_host)));
    check(cudaMalloc(&consumer_device,sizeof(observed)));
    check(cudaStreamSynchronize(stream));
    cudaGraph_t graph{}; cudaGraphExec_t executable{};
    const int *topology_flag{}, *flags{}; const double* guarded{};
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    check(qoco_gpu_topology_validate_device(topology,topology_input,stream,&topology_flag));
    check(qoco_gpu_conversion_run_device(conversion,input,topology_flag,stream,&flags));
    check(qoco_gpu_conversion_guard_numeric(conversion,numeric_device,stream,&guarded));
    consume_guard<<<1,1,0,stream>>>(flags,guarded,consumer_device); check(cudaGetLastError());
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for (int numeric_invalid=0;numeric_invalid<2;++numeric_invalid) for (int mask=0;mask<16;++mask) {
        set(4,1,(mask&1) ? 0 : -inf);
        set(2,3,(mask&2) ? std::numeric_limits<double>::quiet_NaN() : 4);
        set(0,2,(mask&4) ? 3.001 : 3);
        topology_values.back()=(mask&8) ? -1 : 7;
        check(cudaMemcpyAsync(topology_device+topology_values.size()-1,&topology_values.back(),sizeof(int),cudaMemcpyHostToDevice,stream));
        numeric_host[8]=numeric_invalid;
        check(cudaMemcpyAsync(numeric_device,numeric_host,sizeof(numeric_host),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));
        check(cudaMemcpyAsync(observed,consumer_device,sizeof(observed),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for (int i=0;i<8;++i) require(observed[i]==numeric_host[i],"guard altered valid numeric scale report fields");
        require(observed[8]==double(mask!=0 || numeric_invalid!=0),"combined guard did not reject invalid input");
        require(observed[9]==mask,"captured device validation lost or retained an error bit");
    }
    set(4,1,-inf); set(2,3,4); set(0,2,3);
    const auto queued_transfers=qoco_gpu_conversion_transfers(conversion);
    require(queued_transfers.d2h_count==resident_transfers.d2h_count
        && queued_transfers.d2h_bytes==resident_transfers.d2h_bytes,
        "captured conversion/guard must not download or wait internally");
    require(qoco_gpu_topology_transfers(topology).d2h_count==0,
        "captured topology validation must not download a host flag");
    require(qoco_gpu_topology_memory(topology).allocations==topology_memory.allocations,
        "captured topology validation allocated storage");
    check(cudaStreamSynchronize(stream));
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    qoco_gpu_topology_destroy(topology);
    check(cudaFree(topology_device)); check(cudaFree(numeric_device)); check(cudaFree(consumer_device));
    const auto after = qoco_gpu_conversion_memory(conversion);
    require(before.allocations == after.allocations && before.bytes == after.bytes && before.peak_bytes == after.peak_bytes,
        "numeric conversion must not allocate");
    auto malformed = input; malformed.arrays[1] = nullptr;
    require(qoco_gpu_conversion_run(conversion, malformed, result.data(), &invalid, stream) == cudaErrorInvalidValue,
        "missing numeric input accepted");
    qoco_gpu_conversion_destroy(conversion);
    for (auto* p : device) check(cudaFree(p));
    check(cudaStreamDestroy(stream));
}
}

int main() {
    run_case(0); run_case(7); run_case(131075);
    std::puts("GPU conversion: independent arithmetic, bounds, symmetry, nonfinite rejection, stream ordering and retained buffers PASS");
    std::puts("Captured topology/conversion/guard/consumer: 96 combined-error cases, exact scale preservation, no internal downloads PASS");
}
