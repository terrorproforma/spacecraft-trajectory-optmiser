#include "../internal/native_qoco_adapter.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); } }
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
template<class T> struct Array {
    std::vector<T> values; T* device{};
    Array(std::vector<T> v, cudaStream_t stream) : values(std::move(v)) {
        check(cudaMalloc(&device, (values.size() + 1) * sizeof(T))); upload(stream);
    }
    ~Array() { cudaFree(device); }
    void upload(cudaStream_t stream) {
        if (!values.empty()) check(cudaMemcpyAsync(device + 1, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    spacepdhcg_accelerator_buffer_view view() const {
        spacepdhcg_accelerator_buffer_view out{};
        out.data = device; out.elements = values.size(); out.byte_offset = sizeof(T); out.element_stride = 1;
        return out;
    }
};
}
int main() {
    if (!std::getenv("SPACEPDHCG_QOCO_LIBRARY")) { std::puts("SKIP: isolated QOCO library required"); return 0; }
    setenv("SPACEPDHCG_TEST_QOCO_GPU_CONVERSION_COMPARE", "1", 1);
    setenv("SPACEPDHCG_TEST_QOCO_GPU_AUDIT_COMPARE", "1", 1);
    cudaStream_t stream{}; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<int> qo{0}, qi, ao{0}, ai, fo{0}, fi;
    std::vector<double> qv, av, fv;
    for (int col = 0; col < 8; ++col) {
        qi.push_back(col); qv.push_back(0.5); qi.push_back(col); qv.push_back(0.5);
        if (col < 2) { qi.push_back(1 - col); qv.push_back(0.01); }
        qo.push_back(qi.size());
        if (col < 5) { ai.push_back(col); av.push_back(0.25); ai.push_back(col); av.push_back(0.75); }
        ao.push_back(ai.size());
        fi.push_back(col); fv.push_back(0.1); fi.push_back(col); fv.push_back(0.2);
        fi.push_back(6); fv.push_back(0.01); fi.push_back(7); fv.push_back(0.02);
        fo.push_back(fi.size());
    }
    Array<int> q_offsets(qo, stream), q_indices(qi, stream), a_offsets(ao, stream), a_indices(ai, stream),
        f_offsets(fo, stream), f_indices(fi, stream);
    Array<double> q(qv, stream), a(av, stream), f(fv, stream), c({0, 0, 0, 0, 0, 0, -0.1, -0.1}, stream),
        lo({0, -inf, -2, 0.5, -inf}, stream), hi({0, 2, inf, 2, inf}, stream),
        offset({0.1, 0.2, 0.3, 2, 0.1, 0.2, 2, 2}, stream),
        vlo({-inf, -inf, -2, 0.25, 0, -inf, -inf, -inf}, stream),
        vhi({inf, 2, inf, 3, 0, inf, inf, inf}, stream);
    spacepdhcg_cuda_cone_descriptor cones[]{
        {SPACEPDHCG_CUDA_CONE_SECOND_ORDER, 0, 2, 0},
        {SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER, 4, 2, 0}};
    spacepdhcg_cuda_scvx_problem problem{};
    problem.topology_fingerprint = 123;
    problem.canonical_structure = {SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION, 123, 8, 5, 8,
        qv.size(), av.size(), fv.size(), cones, 2, cones, 2};
    problem.canonical_topology = {q_offsets.view(), q_indices.view(), a_offsets.view(), a_indices.view(), f_offsets.view(), f_indices.view()};
    problem.numeric.quadratic = q.view(); problem.numeric.scalar_constraint = a.view(); problem.numeric.affine_cone = f.view();
    problem.numeric.linear_objective = c.view(); problem.numeric.scalar_lower = lo.view(); problem.numeric.scalar_upper = hi.view();
    problem.numeric.affine_offset = offset.view(); problem.numeric.variable_lower = vlo.view(); problem.numeric.variable_upper = vhi.view();
    spacepdhcg_native_qoco* workspace{};
    require(spacepdhcg_native_qoco_create(&problem, stream, 0, &workspace) == SPACEPDHCG_CUDA_SUCCESS,
        "compile mixed scalar/box/SOC/RSOC conversion with duplicate sparse entries");
    double *primal{}, *dual{}; check(cudaMalloc(&primal, 8 * sizeof(double))); check(cudaMalloc(&dual, 13 * sizeof(double)));
    spacepdhcg_native_qoco_report report{};
    const auto solve = [&] { return spacepdhcg_native_qoco_update_solve(workspace, &problem, stream,
        SPACEPDHCG_CUDA_WARM_START_NONE, primal, dual, &report); };
    require(solve() == SPACEPDHCG_CUDA_SUCCESS && report.solves == 1, "initial synthetic solve");
    for (double& value : q.values) value *= 1.25; q.upload(stream);
    offset.values[6] += 0.25; offset.upload(stream);
    c.values[6] -= 0.01; c.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_SUCCESS && report.numeric_updates == 1, "compiled update agrees with CPU conversion and KKT audit");
    lo.values[1] = -1; lo.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH, "new finite scalar bound rejected");
    lo.values[1] = -inf; lo.upload(stream);
    vhi.values[4] = 1; vhi.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH, "box equality classification mutation rejected");
    vhi.values[4] = 0; vhi.upload(stream);
    cones[1].kind = SPACEPDHCG_CUDA_CONE_SECOND_ORDER;
    require(solve() == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH, "cone descriptor mutation rejected");
    cones[1].kind = SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER;
    a_indices.values[0] = 1; a_indices.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH, "in-place sparse topology mutation rejected");
    a_indices.values[0] = 0; a_indices.upload(stream);
    q.values[2] += 0.1; q.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_UNSUPPORTED, "asymmetric quadratic update rejected");
    q.values[2] -= 0.1; q.upload(stream);
    f.values.back() = std::numeric_limits<double>::quiet_NaN(); f.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_NUMERICAL_FAILURE && report.failure == SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL,
        "nonfinite coefficients rejected and correctly reported");
    f.values.back() = 0.02; f.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_SUCCESS, "restored inputs usable after rejected conversions");
    spacepdhcg_native_qoco_destroy(workspace);
    check(cudaFree(primal)); check(cudaFree(dual)); check(cudaStreamDestroy(stream));
    std::puts("Native GPU conversion: mixed bound/cone maps, duplicate entries, offset views, CPU oracle and mutation contracts PASS");
}
