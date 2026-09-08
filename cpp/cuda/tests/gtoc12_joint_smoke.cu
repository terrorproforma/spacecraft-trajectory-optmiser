// Standalone: nvcc -std=c++17 --fmad=false -Icpp/cuda/include \
//   cpp/cuda/src/gtoc12_joint.cu cpp/cuda/tests/gtoc12_joint_smoke.cu -o joint-smoke
// Requires an idle, explicitly selected CUDA device. No test runs at build time.
#include "spacepdhcg/cuda/gtoc12_joint_c_api.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using Visit = spacepdhcg_gtoc12_joint_visit;
using Stage = spacepdhcg_gtoc12_joint_stage;
using Cost = spacepdhcg_gtoc12_joint_cost;
using Policy = spacepdhcg_gtoc12_joint_policy;
using Result = spacepdhcg_gtoc12_joint_result;

static void close(double actual, double expected) {
    if (!(std::abs(actual - expected) <= 1e-10 * (1.0 + std::abs(expected)))) {
        std::fprintf(stderr, "joint mismatch: %.17g != %.17g\n", actual, expected);
        std::abort();
    }
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    assert(cudaSetDevice(0) == cudaSuccess);
    constexpr int count = 257, n = 3, legs = n - 1;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    Policy policy{0, 2000, 3000, 500, 40, 365.25, 10, 365.25, .6, 30,
        60, .05, nan, nan, 0, 0, 0, 0};
    Visit visits[n]{{0, 0, -2, 0, 0, nan, 0, 1},
        {1, 1, 1, 0, 0, nan, 1200, .5}, {0, 0, -2, 0, 0, nan, 0, 1}};
    Stage stages[legs]{{0, 1, 0, 0, 300, 600, 1, 1, 0, 0, 1},
        {0, 0, 0, 0, 100, 1200, 1, 1, 0, 0, 1}};
    std::vector<double> arr(count * n), dep(count * n);
    std::vector<Cost> costs(count * legs);
    std::vector<Result> results(count);
    std::vector<double> masses(count * legs), inflation(count * legs), proxy(count * legs), mined(count * n);
    for (int i = 0; i < count; ++i) {
        arr[i * n] = dep[i * n] = 0;
        arr[i * n + 1] = 400; dep[i * n + 1] = 1200;
        arr[i * n + 2] = dep[i * n + 2] = 1500;
        costs[i * legs] = Cost{0, 0, 2, 0, 0};
        costs[i * legs + 1] = Cost{0, 0, 1, 0, 0};
    }
    arr[n] = dep[n] = -1;
    dep[2 * n + 1] = 399; // Dwell checked before the now-long return TOF.
    arr[3 * n + 1] = 100;
    dep[4 * n + 1] = 600;
    costs[5 * legs].lambert = inf;
    costs[6 * legs] = Cost{1, 0, inf, 2, 3000};
    costs[7 * legs] = Cost{1, 0, inf, 2, 3060.00001};
    costs[8 * legs].lambert = 100;
    costs[9 * legs] = Cost{1, 0, 100, 100, 3000};
    costs[9 * legs + 1].lambert = 0;
    // A constant five-kg dry-mass deficit cannot be cured by unloading cargo.
    // Four passes leave failure 15 with the fourth pass's diagnostics retained.
    costs[10 * legs] = Cost{1, 0, 100, -30 * std::log(535.0 / 3000), 3000};
    costs[10 * legs + 1].lambert = 0;
    void* workspace = nullptr;
    assert(spacepdhcg_gtoc12_joint_create(0, count, n, &workspace) == 0);
    const auto run = [&](int size) {
        return spacepdhcg_gtoc12_joint_evaluate_host(workspace, size, &policy, visits, stages,
            arr.data(), dep.data(), costs.data(), results.data(), masses.data(),
            inflation.data(), proxy.data(), mined.data());
    };
    assert(run(count) == 0);
    const int expected[] = {0, 1, 4, 7, 11, 12, 0, 12, 14, 16, 15};
    for (int i = 0; i < 11; ++i) assert(results[i].failure == expected[i]);
    assert(results[10].rounds == 4 && results[10].mass_count == 2);
    assert(results[9].mass_count == 0 && std::isinf(results[9].objective));
    const double payload = 10 * 800.0 / 365.25;
    const double first_propellant = 3000 * (1 - std::exp(-2.0 / 30));
    const double return_mass = 3000 - first_propellant - 40 + payload;
    const double return_propellant = return_mass * (1 - std::exp(-1.0 / 30));
    const double final_mass = return_mass - return_propellant;
    for (int i : {0, 6, 128, 256}) {
        assert(results[i].failure == 0);
        close(results[i].weighted, payload * .5);
        close(results[i].collected, payload);
        close(results[i].final_mass, final_mass);
        close(results[i].propellant, first_propellant + return_propellant);
        close(results[i].spare, final_mass - 500 - payload);
        close(results[i].objective, payload * .5 + .05 * (final_mass - 500 - payload));
        close(masses[i * legs + 1], return_mass);
        close(mined[i * n + 1], payload);
    }
    assert(results[6].measured_legs == 1);
    close(proxy[6 * legs], 2); close(inflation[6 * legs], 1);
    // Input refresh and original failure precedence on the retained workspace.
    visits[1].structural_failure = SPACEPDHCG_JOINT_DOUBLE_COLLECT;
    assert(run(2) == 0);
    assert(results[0].failure == 9 && results[1].failure == 1);
    visits[1].structural_failure = 0;
    policy.free_earth_leg = 1; policy.earth_out_tof_floor = 450;
    assert(run(1) == 0 && results[0].failure == 13);
    policy.screen_earth_out = 1; policy.earth_out_inflation = .5;
    assert(run(1) == 0 && results[0].failure == 0);
    close(inflation[0], .5);
    // Unsupported policies fail before touching caller outputs.
    stages[0].model = 99;
    results[0].failure = 1234;
    assert(run(1) == 4 && results[0].failure == 1234);
    assert(spacepdhcg_gtoc12_joint_evaluate_host(workspace, 0, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 0);
    assert(spacepdhcg_gtoc12_joint_destroy(&workspace) == 0 && workspace == nullptr);
    assert(spacepdhcg_gtoc12_joint_destroy(&workspace) == 0);
    std::puts("joint smoke passed (257 candidates, reuse, gates, measured costs, four rounds)");
    return 0;
}
