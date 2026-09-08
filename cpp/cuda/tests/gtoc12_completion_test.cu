// Standalone build (never runs a GPU test during compilation):
// nvcc -std=c++20 --fmad=false -Icpp/cuda/include cpp/cuda/src/gtoc12_completion.cu \
//   cpp/cuda/tests/gtoc12_completion_test.cu -o completion-test
#include "spacepdhcg/cuda/gtoc12_completion_c_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

using Policy = spacepdhcg_gtoc12_completion_policy;
using Candidate = spacepdhcg_gtoc12_completion_candidate;
using Deploy = spacepdhcg_gtoc12_completion_deploy;
using Leg = spacepdhcg_gtoc12_completion_leg;
using Result = spacepdhcg_gtoc12_completion_result;
using Detail = spacepdhcg_gtoc12_completion_leg_result;
using Stats = spacepdhcg_gtoc12_completion_stats;
static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
static constexpr double kInf = std::numeric_limits<double>::infinity();

#define REQUIRE(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "completion check failed at line %d: %s\n", __LINE__, #condition); \
    std::exit(1); } } while (false)
static void close(double actual, double expected) {
    if (!(std::abs(actual - expected) <= 2e-12 * (1.0 + std::abs(expected)))) {
        std::fprintf(stderr, "completion numeric mismatch %.17g != %.17g\n", actual, expected);
        std::exit(1);
    }
}
static Policy physical_policy() {
    return {1, 1, 0, 0, 3000, 500, 40, .6, 4000 * 9.80665 * 1e-3, 10, 365.25, 365.25};
}
static Leg flight(int role = SPACEPDHCG_COMPLETION_EARTH_RETURN, int model = 0,
    double departure = 365.25, double tof = 365.25, double dv = 1) {
    Leg l{};
    l.role = role; l.model = model;
    l.source_deploy = role == 3 || role == 4 ? 0 : -1;
    l.departure = departure; l.arrival = departure + tof; l.dv = dv;
    l.flat = 1.2; l.floor = 1.05; l.slope = .65; l.authority_ratio = .5;
    return l;
}
struct Batch {
    std::vector<Candidate> candidates;
    std::vector<Deploy> deploys;
    std::vector<Leg> legs;
    std::vector<int> expected;
    void add(double mass, std::vector<Deploy> d, std::vector<Leg> l, int failure) {
        candidates.push_back({int(deploys.size()), int(d.size()), int(legs.size()), int(l.size()), mass});
        deploys.insert(deploys.end(), d.begin(), d.end());
        legs.insert(legs.end(), l.begin(), l.end());
        expected.push_back(failure);
    }
};
static Batch normal_batch() {
    Batch b;
    const Deploy d{0, 365.25, 1, 0};
    Leg l = flight();
    b.add(900, {d}, {l}, 0); // 0: original prefix fuel, mass-dependent departure.
    Leg hop = flight(3, 0, 365.25, 100, 0), ret = flight(4, 0, 365.25, 100, 0);
    b.add(900, {d}, {hop, ret}, 0); // 1: duplicate pickup mass vs dictionary semantics.
    Leg camp = flight(0, 0, 0, 365.25, kNaN); camp.flat = kNaN;
    hop.departure = 300; hop.arrival = 400;
    b.add(900, {d}, {camp, hop, ret}, 0); // 2: camp and reposition do not collect.
    l.flat = kNaN;
    b.add(900, {{0, 0, 0, 0}}, {l}, 1); // 3: missing collect precedes flight checks.
    b.add(900, {{0, 300, 1, 0}}, {l}, 2); // 4: short mining stay precedes inflation.
    b.add(900, {{0, 300, 1, 0}, {0, 0, 0, 0}}, {l}, 2); // 5: dict-order preflight.
    l.dv = 1000;
    b.add(900, {d}, {l}, 3); // 6: authority before invalid inflation; pickup already done.
    l.dv = 1;
    b.add(900, {d}, {l}, 4); // 7: nonfinite inflation is its own candidate failure.
    l.flat = -1;
    b.add(900, {d}, {l}, 4); // 8: negative inflation.
    l = flight(3, 1); l.floor = .9; l.slope = .5;
    b.add(900, {d}, {l}, 0); // 9: actual-mass ratio inflation.
    l = flight(3, 2); l.floor = .1; l.fit[0] = .8; l.fit[1] = .2;
    l.fit[2] = .1; l.fit[3] = .03; l.fit[4] = .04;
    l.delta_a_au = -.2; l.delta_longitude_rad = -3.14159265358979323846 / 2;
    b.add(900, {d}, {l}, 0); // 10: all five fit features.
    l = flight(4, 5, 365.25, 420, 5); l.authority_ratio = 1;
    b.add(900, {d}, {l}, 0); // 11: generic DP return model.
    l.model = 4; l.flat = .7;
    b.add(900, {d}, {l}, 0); // 12: certified cell is a supplied inflation, not a bypass.
    l = flight(4, 3, 365.25, 540, 5); l.flat = kNaN;
    b.add(900, {d}, {l}, 0); // 13: return model ignores stored flat inflation.
    b.add(499, {}, {}, 5); // 14: empty forward path still applies final mass gate.
    b.add(500, {}, {}, 0); // 15: equality at final mass gate is accepted.
    b.add(std::nextafter(500.0, -kInf), {}, {}, 5); // 16: one ULP below rejected.
    l = flight(2); l.dv = 0; l.flat = 0;
    b.add(500, {}, {l}, 0); // 17: other forward roles use supplied flat, no pickup.
    l = flight(); l.dv = kNaN;
    b.add(900, {d}, {l}, 3); // 18: NaN DV fails the actual <= gate.
    l = flight(); l.arrival = kNaN;
    b.add(900, {d}, {l}, 3); // 19: NaN TOF is not hidden by a maximum.
    l = flight(3, 2); l.fit[0] = kNaN;
    b.add(900, {d}, {l}, 4); // 20: NaN fit must propagate through floor.
    l = flight(4, 4); l.flat = kNaN; l.dv = 1000;
    b.add(900, {d}, {l}, 3); // 21: certified override cannot defeat authority.
    l = flight(3, 1); l.floor = .75; l.slope = 0; l.authority_ratio = 1;
    const double authority = ((.6 / 910 * 1e-3) * 365.25) * 86400;
    l.dv = authority;
    b.add(900, {d}, {l}, 0); // 22: exact authority equality accepted.
    l.dv = std::nextafter(authority, kInf);
    b.add(900, {d}, {l}, 3); // 23: one ULP above rejected.
    l = flight(4, 5, 365.25, 100, 0);
    b.add(900, {d}, {l}, 0); // 24: return lower endpoint and ratio clip.
    l = flight(4, 5, 365.25, 1000, 0);
    b.add(900, {d}, {l}, 0); // 25: return upper endpoint.
    while (b.candidates.size() < 257) b.add(900, {d}, {flight()}, 0);
    return b;
}
static Batch edge_batch() {
    Batch b;
    Leg l = flight(3, 0, 0, 1, 0); l.flat = 0;
    b.add(500, {{0, 0, 1, 0}}, {l, l}, 0); // zero-gain duplicate pickup exists in dict.
    b.add(500, {{5e-7, 0, 1, 0}}, {l}, 6); // tolerated preflight then mining ValueError.
    b.add(500, {{kNaN, 0, 1, 0}}, {l}, 6); // NaN stay also raises only at pickup.
    b.add(500, {{0, kNaN, 1, 0}}, {l}, 0); // nonmatching NaN collect epoch: no pickup.
    std::vector<Leg> many;
    for (int source : {2, 0, 1}) {
        Leg x = l; x.source_deploy = source;
        x.departure = source == 2 ? 1e16 : 1; x.arrival = x.departure + 1;
        many.push_back(x);
    }
    b.add(500, {{0, 1, 1, 0}, {0, 1, 1, 0}, {0, 1e16, 1, 0}}, many, 5);
    // The high-part mass loses both +1 pickups; Python 3.12's cargo sum retains
    // +2. Ordinary sequential summation would falsely accept this boundary.
    return b;
}
static int evaluate(void* w, const Policy& p, const Batch& b, std::vector<Result>& out,
    std::vector<Detail>& details, std::vector<double>& mined, Stats* stats = nullptr) {
    out.resize(b.candidates.size()); details.resize(b.legs.size()); mined.resize(b.deploys.size());
    return spacepdhcg_gtoc12_completion_evaluate_host(w, int(b.candidates.size()), int(b.deploys.size()),
        int(b.legs.size()), &p, b.candidates.data(), b.deploys.data(), b.legs.data(),
        out.data(), details.data(), mined.data(), stats);
}
static void failures(const Batch& b, const std::vector<Result>& out) {
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i].failure != b.expected[i]) {
            std::fprintf(stderr, "case %zu failure %d != %d\n", i, out[i].failure, b.expected[i]);
            std::exit(1);
        }
    }
}

int main(int argc, char** argv) {
    const bool cpu_only = argc == 2 && std::string_view(argv[1]) == "--cpu-only";
    REQUIRE(argc == 1 || cpu_only);
    const Batch batch = normal_batch(), edges = edge_batch();
    REQUIRE(batch.candidates.size() == 257 && edges.candidates.size() == 5);
    REQUIRE(sizeof(Policy) == 80 && sizeof(Leg) == 128 && sizeof(Detail) == 72);
    const double compensated = double(1e16L + 1.L + 1.L);
    REQUIRE(compensated == 10000000000000002.0);
    REQUIRE(((1e16 + 1.0) + 1.0) != compensated);
    void* invalid = reinterpret_cast<void*>(1);
    REQUIRE(spacepdhcg_gtoc12_completion_create(0, 0, 0, 0, &invalid) == 1 && !invalid);
    if (cpu_only) {
        std::puts("COMPLETION_TEST {\"phase\":\"cpu_only\",\"fixtures\":262,\"GPU_calls\":0}");
        return 0;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    void* workspace = nullptr;
    REQUIRE(spacepdhcg_gtoc12_completion_create(0, 257, int(batch.deploys.size()),
        int(batch.legs.size()), &workspace) == 0);
    const Policy policy = physical_policy();
    std::vector<Result> out;
    std::vector<Detail> detail;
    std::vector<double> mined;
    Stats stats{};
    REQUIRE(evaluate(workspace, policy, batch, out, detail, mined, &stats) == 0);
    failures(batch, out);
    REQUIRE(stats.candidates == 257 && stats.leg_slots == batch.legs.size() && stats.deploy_slots == batch.deploys.size());
    REQUIRE(stats.upload_ms >= 0 && stats.kernel_ms >= 0 && stats.download_ms >= 0);
    const auto at = [&](int row, int leg = 0) -> const Detail& { return detail[batch.candidates[row].leg_begin + leg]; };
    const double first_fuel = 910 * (1 - std::exp(-1.2 / policy.exhaust));
    for (int row : {0, 128, 256}) {
        close(out[row].collected, 10); close(out[row].final_mass, 910 - first_fuel);
        close(out[row].propellant, 2060 + first_fuel);
        REQUIRE(out[row].failed_leg == -1 && out[row].failed_deploy == -1 && out[row].processed_legs == 1);
        close(at(row).mass_before, 900); close(at(row).departure_mass, 910);
        REQUIRE(at(row).pickup == 1 && at(row).stage == 4);
    }
    close(out[1].collected, 10); close(out[1].final_mass, 920);
    REQUIRE(at(1).pickup == 1 && at(1, 1).pickup == 1);
    REQUIRE(at(2).stage == 1 && at(2).pickup == 0 && std::isnan(at(2).inflation));
    REQUIRE(at(2, 1).pickup == 0 && at(2, 2).pickup == 1);
    close(out[2].collected, 10); close(out[2].final_mass, 910);
    REQUIRE(out[5].failed_deploy == 0 && out[5].failed_leg == -1 && at(5).stage == 0);
    REQUIRE(out[6].failed_leg == 0 && out[6].processed_legs == 0 && at(6).pickup == 1 && at(6).stage == 2);
    REQUIRE(std::isnan(at(7).inflation) && at(7).stage == 3);
    const double authority = ((policy.thrust / 910 * 1e-3) * 365.25) * 86400;
    close(at(9).inflation, .9 + .5 / authority);
    close(at(10).inflation, .8 + .2 / authority + .1 + .06 + .02);
    const double return_ratio = 5 / (((.6 / 910 * 1e-3) * 420) * 86400);
    close(at(11).inflation, 1.383 * (1 + .6 * (return_ratio - .33)));
    close(at(12).inflation, .7); close(at(24).inflation, 1.323 * .85);
    close(at(25).inflation, 1.014 * .85);
    REQUIRE(out[15].margin == 0 && out[16].margin < 0);

    // Retained buffers must clear prior seen/detail state, and zero gain still
    // produces explicit pickup records. This second batch has different ragged sizes.
    Policy edge = policy;
    edge.initial_mass = edge.dry_mass = 500; edge.miner_mass = 0;
    edge.minimum_stay = 0; edge.mining_rate = edge.year_days = 1;
    REQUIRE(evaluate(workspace, edge, edges, out, detail, mined) == 0);
    failures(edges, out);
    REQUIRE(detail[0].pickup == 1 && detail[1].pickup == 1 && mined[0] == 0);
    REQUIRE(out[1].failed_leg == 0 && out[1].failed_deploy == 0);
    REQUIRE(detail[edges.candidates[1].leg_begin].stage == 5);
    REQUIRE(detail[edges.candidates[3].leg_begin].pickup == 0);
    REQUIRE(out[4].collected == compensated && out[4].margin == -2);

    // Malformed envelope calls do not launch work or modify output sentinels.
    Result sentinel{}; sentinel.final_mass = -12345;
    Candidate bad{1, 0, 0, 0, 500};
    REQUIRE(spacepdhcg_gtoc12_completion_evaluate_host(workspace, 1, 0, 0, &policy, &bad,
        nullptr, nullptr, &sentinel, nullptr, nullptr, nullptr) == 1);
    REQUIRE(sentinel.final_mass == -12345);
    bad.deploy_begin = 0;
    Policy unsupported = policy; unsupported.sum_mode = 0;
    REQUIRE(spacepdhcg_gtoc12_completion_evaluate_host(workspace, 1, 0, 0, &unsupported,
        &bad, nullptr, nullptr, &sentinel, nullptr, nullptr, nullptr) == 4);
    REQUIRE(sentinel.final_mass == -12345);
    Deploy d{0, 365.25, 1, 0}; Leg l = flight(); bad.deploy_count = bad.leg_count = 1;
    l.source_deploy = 1;
    REQUIRE(spacepdhcg_gtoc12_completion_evaluate_host(workspace, 1, 1, 1, &policy,
        &bad, &d, &l, &sentinel, nullptr, nullptr, nullptr) == 1);
    l.source_deploy = 0; l.model = 1;
    REQUIRE(spacepdhcg_gtoc12_completion_evaluate_host(workspace, 1, 1, 1, &policy,
        &bad, &d, &l, &sentinel, nullptr, nullptr, nullptr) == 4);
    REQUIRE(sentinel.final_mass == -12345);
    REQUIRE(spacepdhcg_gtoc12_completion_evaluate_host(workspace, 0, 0, 0, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &stats) == 0);
    REQUIRE(stats.candidates == 0 && stats.kernel_ms == 0);
    REQUIRE(spacepdhcg_gtoc12_completion_evaluate_host(workspace, 0, 1, 0, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 1);
    if (device_count > 1) {
        REQUIRE(cudaSetDevice(1) == cudaSuccess);
        REQUIRE(spacepdhcg_gtoc12_completion_evaluate_host(workspace, 0, 0, 0, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == 1);
        REQUIRE(cudaSetDevice(0) == cudaSuccess);
    }
    REQUIRE(spacepdhcg_gtoc12_completion_destroy(&workspace) == 0 && !workspace);
    REQUIRE(spacepdhcg_gtoc12_completion_destroy(&workspace) == 0);
    std::puts("COMPLETION_TEST {\"phase\":\"native\",\"kernel_launches\":2,\"candidates\":262,\"passed\":true}");
    return 0;
}
