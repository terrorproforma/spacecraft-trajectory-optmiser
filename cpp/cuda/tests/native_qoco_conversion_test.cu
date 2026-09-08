#include "../internal/native_qoco_adapter.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
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
int main(int argc, char** argv) {
    const int ruiz_iterations = argc > 1 ? std::atoi(argv[1]) : 0;
    const bool origin_test = argc > 3 && std::strcmp(argv[3], "origin") == 0;
    const bool device_initialization = argc > 4 && std::strcmp(argv[4], "device-init") == 0;
    require(ruiz_iterations >= 0 && ruiz_iterations <= 100, "Ruiz iteration argument");
    if (!std::getenv("SPACEPDHCG_QOCO_LIBRARY")) { std::puts("SKIP: isolated QOCO library required"); return 0; }
    const bool device_validation = argc > 2 && std::strcmp(argv[2], "device-validation") == 0;
    if (device_validation) {
        unsetenv("SPACEPDHCG_TEST_QOCO_GPU_CONVERSION_COMPARE");
        unsetenv("SPACEPDHCG_TEST_QOCO_GPU_AUDIT_COMPARE");
        setenv("SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION", "1", 1);
        setenv("SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY", "1", 1);
        setenv("SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY", "1", 1);
        setenv("SPACEPDHCG_TEST_QOCO_IPM_GRAPH", "1", 1);
    } else {
        setenv("SPACEPDHCG_TEST_QOCO_GPU_CONVERSION_COMPARE", "1", 1);
        setenv("SPACEPDHCG_TEST_QOCO_GPU_AUDIT_COMPARE", "1", 1);
    }
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
    const bool trajectory_test = argc > 2 && std::strcmp(argv[2], "trajectory") == 0;
    const auto create = [&] {
        require((device_initialization
            ? spacepdhcg_native_qoco_create_configured(&problem,stream,ruiz_iterations,1e-8,true,&workspace)
            : spacepdhcg_native_qoco_create(&problem,stream,ruiz_iterations,&workspace)) == SPACEPDHCG_CUDA_SUCCESS,
            "compile mixed scalar/box/SOC/RSOC conversion with duplicate sparse entries");
    };
    if (trajectory_test) {
        Array<int> states({0,1,2},stream),controls({3,4},stream),virtuals({5,6},stream);
        problem.intervals=2; problem.state_dimension=1; problem.control_dimension=1;
        auto indices=[](const Array<int>& a) {
            auto view=a.view(); view.device.type=SPACEPDHCG_DEVICE_CUDA;
            view.scalar_type=SPACEPDHCG_SCALAR_INT32; return view;
        };
        problem.state_variable_indices=indices(states); problem.control_variable_indices=indices(controls);
        problem.virtual_variable_indices=indices(virtuals); create();
        problem.state_variable_indices={}; problem.control_variable_indices={}; problem.virtual_variable_indices={};
    } else create();
    double *primal{}, *dual{}; check(cudaMalloc(&primal, 8 * sizeof(double))); check(cudaMalloc(&dual, 13 * sizeof(double)));
    spacepdhcg_native_qoco_report report{};
    const auto solve = [&] { return spacepdhcg_native_qoco_update_solve(workspace, &problem, stream,
        SPACEPDHCG_CUDA_WARM_START_NONE, primal, dual, &report); };
    if (origin_test) {
        Array<double> origin({std::numeric_limits<double>::quiet_NaN(),-.5,.25,1,.125,-.25,.5,-.125},stream);
        require(spacepdhcg_native_qoco_set_origin(workspace,origin.device+1,0,stream)==SPACEPDHCG_CUDA_INVALID_ARGUMENT,"zero origin count rejected");
        require(spacepdhcg_native_qoco_set_origin(workspace,origin.device+1,9,stream)==SPACEPDHCG_CUDA_INVALID_ARGUMENT,"oversized origin rejected");
        require(spacepdhcg_native_qoco_set_origin(workspace,origin.device+1,8,stream)==SPACEPDHCG_CUDA_SUCCESS,"owned origin setup");
        require(solve()==SPACEPDHCG_CUDA_NUMERICAL_FAILURE && report.status_code==3
            && report.iterations==0 && report.failure==SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL,"first nonfinite origin rejected with accurate failure report");
        origin.values[0]=.25; origin.upload(stream);
        std::vector<double> reference(8);
        for (int repeat=0;repeat<4;++repeat) {
            for (double& value:origin.values) value*=-.5;
            origin.upload(stream);
            require(spacepdhcg_native_qoco_set_origin(workspace,origin.device+1,repeat%2 ? 4 : 8,stream)==SPACEPDHCG_CUDA_SUCCESS,"changing full/prefix origin");
            require(solve()==SPACEPDHCG_CUDA_SUCCESS && report.primal_residual<=1e-8
                && report.dual_residual<=1e-8,"translated solve passes original-coordinate KKT audit");
            std::vector<double> physical(8);check(cudaMemcpy(physical.data(),primal,8*sizeof(double),cudaMemcpyDeviceToHost));
            if (!repeat) reference=physical;
            else for (int i=0;i<8;++i) require(std::abs(physical[i]-reference[i])<1e-7,"translation preserves unique physical solution");
        }
        require(report.workspace_creations==2,"first invalid origin forces a fresh workspace on recovery");
        require(spacepdhcg_native_qoco_accept(workspace,&report)==SPACEPDHCG_CUDA_INVALID_STATE,"translated accepted-primal cache rejected");
        require(spacepdhcg_native_qoco_update_solve(workspace,&problem,stream,SPACEPDHCG_CUDA_WARM_START_PRIMAL,
            primal,dual,&report)==SPACEPDHCG_CUDA_INVALID_STATE,"translated warm start rejected");
        if (device_validation) {
            require(spacepdhcg_native_qoco_can_enqueue(workspace),"translated cold replay ready");
            require(spacepdhcg_native_qoco_enqueue(workspace,&problem,stream,primal,dual,nullptr,nullptr,nullptr)==SPACEPDHCG_CUDA_SUCCESS,"translated asynchronous submission");
            require(spacepdhcg_native_qoco_set_origin(workspace,origin.device+1,8,stream)==SPACEPDHCG_CUDA_INVALID_STATE,"pending origin cannot be overwritten");
            require(spacepdhcg_native_qoco_finish(workspace,stream,&report)==SPACEPDHCG_CUDA_SUCCESS
                && report.primal_residual<=1e-8 && report.dual_residual<=1e-8,"deferred original-coordinate audit");
        }
        spacepdhcg_native_qoco_destroy(workspace);check(cudaFree(primal));check(cudaFree(dual));check(cudaStreamDestroy(stream));
        std::puts("Native translated QP: original KKT/unique solution, changing origins, first-invalid recovery, cold-only and pending ownership PASS");
        return 0;
    }
    require(solve() == SPACEPDHCG_CUDA_SUCCESS && report.solves == 1, "initial synthetic solve");
    const auto before_accept_count = report.d2d_copy_count;
    const auto before_accept_bytes = report.d2d_bytes;
    require(spacepdhcg_native_qoco_accept(workspace, &report) == SPACEPDHCG_CUDA_SUCCESS,
        "accept completed primal without a host round trip");
    if (std::getenv("SPACEPDHCG_TEST_QOCO_DEVICE_IO_REQUIRED"))
        require(report.d2d_copy_count == before_accept_count + 1
                && report.d2d_bytes == before_accept_bytes + 8 * sizeof(double),
            "accepted-primal transfer is reported before any subsequent solve");
    require(spacepdhcg_native_qoco_reset_warm_state(workspace, true) == SPACEPDHCG_CUDA_SUCCESS,
        "retain accepted start");
    require(spacepdhcg_native_qoco_reset_warm_state(workspace, false) == SPACEPDHCG_CUDA_SUCCESS,
        "discard accepted start");
    for (double& value : q.values) value *= 1.25; q.upload(stream);
    offset.values[6] += 0.25; offset.upload(stream);
    c.values[6] -= 0.01; c.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_SUCCESS && report.numeric_updates == 1, "compiled update agrees with CPU conversion and KKT audit");
    if (std::getenv("SPACEPDHCG_TEST_QOCO_DEVICE_UPDATE_REQUIRED"))
        require(report.device_numeric_updates == (ruiz_iterations > 0 ? 2U : 1U)
                + (device_initialization ? 1U : 0U),
            "device extension must perform requested initial equilibration and numeric update");
    const auto reprime = [&] {
        if (!device_validation) return;
        require(solve()==SPACEPDHCG_CUDA_SUCCESS,"recover after guarded invalid input");
        require(solve()==SPACEPDHCG_CUDA_SUCCESS,"prime numerical scale and replay after recovery");
        require(report.primal_residual<=1e-8 && report.dual_residual<=1e-8,"recovered independent KKT accuracy");
    };
    reprime();
    lo.values[1] = -1; lo.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH, "new finite scalar bound rejected");
    lo.values[1] = -inf; lo.upload(stream);
    reprime();
    vhi.values[4] = 1; vhi.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH, "box equality classification mutation rejected");
    vhi.values[4] = 0; vhi.upload(stream);
    reprime();
    cones[1].kind = SPACEPDHCG_CUDA_CONE_SECOND_ORDER;
    require(solve() == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH, "cone descriptor mutation rejected");
    cones[1].kind = SPACEPDHCG_CUDA_CONE_ROTATED_SECOND_ORDER;
    reprime();
    a_indices.values[0] = 1; a_indices.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_TOPOLOGY_MISMATCH, "in-place sparse topology mutation rejected");
    a_indices.values[0] = 0; a_indices.upload(stream);
    reprime();
    q.values[2] += 0.1; q.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_UNSUPPORTED, "asymmetric quadratic update rejected");
    q.values[2] -= 0.1; q.upload(stream);
    reprime();
    f.values.back() = std::numeric_limits<double>::quiet_NaN(); f.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_NUMERICAL_FAILURE && report.failure == SPACEPDHCG_CUDA_QOCO_FAILURE_NUMERICAL,
        "nonfinite coefficients rejected and correctly reported");
    f.values.back() = 0.02; f.upload(stream);
    require(solve() == SPACEPDHCG_CUDA_SUCCESS, "restored inputs usable after rejected conversions");
    require(report.primal_residual <= 1e-8 && report.dual_residual <= 1e-8, "independent KKT accuracy after updates");
    if (trajectory_test) {
        const auto creations=report.workspace_creations;
        check(cudaStreamDestroy(stream));
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        void* probe=dlopen(std::getenv("SPACEPDHCG_QOCO_LIBRARY"),RTLD_NOW|RTLD_LOCAL);
        require(probe!=nullptr,"load recovery fault probe");
        auto fail_next=reinterpret_cast<void(*)()>(dlsym(probe,"qoco_test_fail_next_solve"));
        require(fail_next!=nullptr,"trajectory lifetime mode requires test-only recovery proxy");
        fail_next();
        require(solve()==SPACEPDHCG_CUDA_NUMERICAL_FAILURE,"forced numerical exit requires fresh solver");
        require(solve()==SPACEPDHCG_CUDA_SUCCESS && report.workspace_creations>creations,
            "recovery setup uses owned metadata after caller index buffers were freed");
        require(report.primal_residual<=1e-8 && report.dual_residual<=1e-8,"recovered KKT accuracy");
        dlclose(probe);
    }
    spacepdhcg_native_qoco_destroy(workspace);
    check(cudaFree(primal)); check(cudaFree(dual)); check(cudaStreamDestroy(stream));
    std::puts("Native GPU conversion: mixed bound/cone maps, duplicate entries, offset views, CPU oracle and mutation contracts PASS");
}
