// Included once by gtoc12_completion.cu; shares its retained workspace and gates.
#include <vector>

namespace {
using ModelPolicy = spacepdhcg_gtoc12_completion_model_policy;
using Orbit = spacepdhcg_gtoc12_completion_orbit;
using ReturnGrid = spacepdhcg_gtoc12_completion_return_grid;
static_assert(sizeof(ModelPolicy) == 176 && sizeof(Orbit) == 40 && sizeof(ReturnGrid) == 16);
static_assert(sizeof(CompactCandidate) == 32 && sizeof(CompactDeploy) == 32 && sizeof(CompactLeg) == 48);
struct Model {
    int device = -1, orbit_count = 0, tof_count = 0;
    ModelPolicy policy{};
    Orbit* orbits{};
    double* tofs{};
    ReturnGrid* grids{};
    int32_t* grid_for_body{};
    int32_t* host_grid_for_body{};
    double* inflation{};
    uint8_t* ok{};
};
bool release_model(Model* m) {
    bool ok = true;
    const auto free_buffer = [&ok](void* p) { if (p && cudaFree(p) != cudaSuccess) ok = false; };
    free_buffer(m->orbits); free_buffer(m->tofs); free_buffer(m->grids);
    free_buffer(m->grid_for_body); free_buffer(m->inflation); free_buffer(m->ok);
    delete[] m->host_grid_for_body;
    m->host_grid_for_body = nullptr;
    return ok;
}
__device__ double positive_mod(double x, double period) {
    const double r = fmod(x, period);
    return r < 0.0 ? r + period : (r == 0.0 ? 0.0 : r);
}
__device__ double longitude(const Orbit& o, double epoch) {
    constexpr double mu = 1.32712440018e11, day = 86400.0;
    constexpr double two_pi = 6.283185307179586476925286766559005768;
    const double a3 = (o.semi_major_axis_km * o.semi_major_axis_km) * o.semi_major_axis_km;
    const double n = sqrt(mu / a3) * day;
    const double anomaly = o.mean_anomaly_rad + n * (epoch - o.epoch_mjd);
    return positive_mod((o.ascending_node_rad + o.argument_of_perihelion_rad) + anomaly, two_pi);
}
__global__ void assemble_candidates(int count, const CompactCandidate* source,
    const CompactDeploy* deploy_source, Candidate* candidates, Deploy* deploys) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    const CompactCandidate c = source[i];
    candidates[i] = {c.deploy_begin, c.deploy_count, c.leg_begin, c.leg_count, c.partial_mass};
    for (int k = 0; k < c.deploy_count; ++k) {
        const int slot = c.deploy_begin + k;
        const CompactDeploy d = deploy_source[slot];
        deploys[slot] = {d.deploy_epoch, d.collect_epoch, d.has_collect, 0};
    }
}
__global__ void assemble_legs(int count, Model m, const CompactCandidate* candidates,
    const CompactDeploy* deploys, const CompactLeg* source, Leg* legs) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    const CompactLeg input = source[i];
    const CompactCandidate c = candidates[input.candidate];
    const ModelPolicy p = m.policy;
    Leg leg{};
    leg.role = input.role; leg.source_deploy = -1;
    leg.departure = input.departure; leg.arrival = input.arrival;
    leg.dv = input.dv; leg.flat = input.input_inflation;
    leg.authority_ratio = p.authority_ratio[input.role];
    if (input.role == SPACEPDHCG_COMPLETION_COLLECT_HOP || input.role == SPACEPDHCG_COMPLETION_EARTH_RETURN) {
        for (int k = 0; k < c.deploy_count; ++k)
            if (deploys[c.deploy_begin + k].body_id == input.from_id) { leg.source_deploy = k; break; }
    }
    if (input.role == SPACEPDHCG_COMPLETION_COLLECT_HOP) {
        leg.model = c.use_table ? p.table_hop_model : p.hop_model;
        // Preserve even unused input fields for exact ABI/readback provenance:
        // the scalar packer writes only the selected model's parameters.
        if (leg.model == SPACEPDHCG_COMPLETION_FLAT) leg.flat = p.hop_flat;
        if (leg.model == SPACEPDHCG_COMPLETION_RATIO) {
            leg.floor = p.hop_floor; leg.slope = p.hop_slope;
        }
        if (leg.model == SPACEPDHCG_COMPLETION_FIT5) {
            leg.floor = p.fit_floor;
            for (int j = 0; j < 5; ++j) leg.fit[j] = p.fit[j];
            const Orbit from = m.orbits[input.from_id - 1], to = m.orbits[input.to_id - 1];
            leg.delta_a_au = (to.semi_major_axis_km - from.semi_major_axis_km) / 1.49597870691e8;
            constexpr double pi = 3.141592653589793238462643383279502884;
            const double delta = longitude(to, input.departure) - longitude(from, input.departure);
            leg.delta_longitude_rad = positive_mod(delta + pi, 2.0 * pi) - pi;
        }
    } else if (input.role == SPACEPDHCG_COMPLETION_EARTH_RETURN) {
        leg.model = c.use_table ? p.table_return_model : p.return_model;
        leg.flat = c.use_table ? p.table_return_flat : p.return_flat;
        const int grid_id = c.use_table ? m.grid_for_body[input.from_id - 1] : -1;
        if (grid_id >= 0) {
            const ReturnGrid grid = m.grids[grid_id];
            const double row = nearbyint((input.departure - p.epoch0) / p.step_days);
            if (row >= 0.0 && row < double(grid.rows)) {
                const double tof = input.arrival - input.departure;
                int nearest = 0;
                double distance = fabs(m.tofs[0] - tof);
                for (int j = 1; j < m.tof_count; ++j) {
                    const double candidate = fabs(m.tofs[j] - tof);
                    if (candidate < distance) { distance = candidate; nearest = j; }
                }
                const int cell = grid.cell_begin + int(row) * m.tof_count + nearest;
                if (m.ok[cell]) { leg.model = SPACEPDHCG_COMPLETION_CERTIFIED_FLAT; leg.flat = m.inflation[cell]; }
            }
        }
    }
    legs[i] = leg;
}

int validate_compact(const Model* m, int count, int nd, int nl, const Policy* p,
    const CompactCandidate* cs, const CompactDeploy* ds, const CompactLeg* ls, const Result* out) {
    if (!cs || !out || (nd && !ds) || (nl && !ls)) return 1;
    const int status = validate_policy(p);
    if (status) return status;
    int64_t d0 = 0, l0 = 0;
    for (int i = 0; i < count; ++i) {
        const CompactCandidate c = cs[i];
        if (c.deploy_begin != d0 || c.leg_begin != l0 || c.deploy_count < 0 || c.leg_count < 0) return 1;
        d0 += c.deploy_count; l0 += c.leg_count;
        if (d0 > nd || l0 > nl) return 1;
        if ((c.use_table != 0 && c.use_table != 1) || c.reserved) return 4;
        for (int k = 0; k < c.deploy_count; ++k) {
            const CompactDeploy d = ds[c.deploy_begin + k];
            if (d.reserved0 || d.reserved1 || (d.has_collect != 0 && d.has_collect != 1)) return 4;
            if (!std::isfinite(d.deploy_epoch) || !std::isfinite(d.collect_epoch) ||
                d.body_id <= 0 || d.body_id > m->orbit_count) return 1;
            for (int j = 0; j < k; ++j) if (ds[c.deploy_begin + j].body_id == d.body_id) return 1;
        }
        for (int k = 0; k < c.leg_count; ++k) {
            const CompactLeg l = ls[c.leg_begin + k];
            if (l.role < 0 || l.role > 4) return 4;
            if (l.candidate != i || l.from_id < 0 || l.to_id < 0 ||
                l.from_id > m->orbit_count || l.to_id > m->orbit_count ||
                !std::isfinite(l.departure) || !std::isfinite(l.arrival)) return 1;
            if (l.role == 3 || l.role == 4) {
                bool found = false;
                for (int j = 0; j < c.deploy_count; ++j) found |= ds[c.deploy_begin + j].body_id == l.from_id;
                if (!found || (l.role == 3 && l.to_id == 0)) return 1;
                // Python round(inf/nan) raises before native evaluation. It must
                // not become an uncertified generic-price fallback on CUDA.
                if (c.use_table && l.role == 4 && m->host_grid_for_body[l.from_id - 1] >= 0 &&
                    !std::isfinite((l.departure - m->policy.epoch0) / m->policy.step_days)) return 1;
            }
        }
    }
    return d0 == nd && l0 == nl ? 0 : 1;
}
} // namespace

extern "C" int spacepdhcg_gtoc12_completion_model_create(int32_t device, const ModelPolicy* p,
    int32_t no, const Orbit* orbits, int32_t nt, const double* tofs,
    int32_t ng, const ReturnGrid* grids, int32_t nc, const double* values, const uint8_t* ok,
    void** output) {
    if (!output) return 1;
    *output = nullptr;
    if (device < 0 || !p || no < 1 || !orbits || nt < 1 || !tofs || ng < 0 || nc < 0 ||
        (ng && !grids) || (nc && (!values || !ok))) return 1;
    if (p->abi_version != 1 || p->reserved0 || p->reserved1 || p->reserved2 ||
        p->hop_model < 0 || p->hop_model > 1 || p->table_hop_model < 0 || p->table_hop_model > 2 ||
        (p->return_model != 0 && p->return_model != 3) ||
        (p->table_return_model != 0 && p->table_return_model != 5)) return 4;
    if (!std::isfinite(p->epoch0) || !std::isfinite(p->step_days) || p->step_days <= 0.0) return 1;
    for (int i = 0; i < no; ++i) {
        const Orbit o = orbits[i];
        if (!std::isfinite(o.semi_major_axis_km) || o.semi_major_axis_km <= 0.0 ||
            !std::isfinite(o.epoch_mjd) || !std::isfinite(o.mean_anomaly_rad) ||
            !std::isfinite(o.ascending_node_rad) || !std::isfinite(o.argument_of_perihelion_rad)) return 1;
    }
    for (int i = 0; i < nt; ++i) if (!std::isfinite(tofs[i])) return 1;
    std::vector<int32_t> mapping;
    try { mapping.assign(size_t(no), -1); } catch (...) { return 2; }
    int64_t end = 0;
    for (int i = 0; i < ng; ++i) {
        const ReturnGrid g = grids[i];
        if (g.reserved) return 4;
        if (g.body_id <= 0 || g.body_id > no || g.rows < 0 || g.cell_begin != end ||
            mapping[size_t(g.body_id - 1)] != -1) return 1;
        end += int64_t(g.rows) * nt;
        if (end > nc) return 1;
        mapping[size_t(g.body_id - 1)] = i;
    }
    if (end != nc) return 1;
    for (int i = 0; i < nc; ++i) if (ok[i] > 1) return 4;
    auto* m = new (std::nothrow) Model;
    if (!m) return 2;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) { delete m; return 2; }
    m->device = device; m->orbit_count = no; m->tof_count = nt; m->policy = *p;
    m->host_grid_for_body = new (std::nothrow) int32_t[size_t(no)];
    if (m->host_grid_for_body) for (int i = 0; i < no; ++i) m->host_grid_for_body[i] = mapping[size_t(i)];
    bool success = m->host_grid_for_body && allocate(m->orbits, no) && allocate(m->tofs, nt) && allocate(m->grids, ng) &&
        allocate(m->grid_for_body, no) && allocate(m->inflation, nc) && allocate(m->ok, nc);
    const auto copy = [&success](void* target, const void* source, size_t bytes) {
        if (success && bytes) success = cudaMemcpy(target, source, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    };
    copy(m->orbits, orbits, size_t(no) * sizeof(Orbit)); copy(m->tofs, tofs, size_t(nt) * sizeof(double));
    copy(m->grids, grids, size_t(ng) * sizeof(ReturnGrid)); copy(m->grid_for_body, mapping.data(), size_t(no) * sizeof(int32_t));
    copy(m->inflation, values, size_t(nc) * sizeof(double)); copy(m->ok, ok, size_t(nc));
    if (!success) release_model(m);
    if (previous != device && cudaSetDevice(previous) != cudaSuccess) {
        if (success) { cudaSetDevice(device); release_model(m); }
        success = false;
    }
    if (!success) { delete m; return 2; }
    *output = m;
    return 0;
}

extern "C" int spacepdhcg_gtoc12_completion_model_destroy(void** opaque) {
    if (!opaque) return 1;
    auto* m = static_cast<Model*>(*opaque);
    if (!m) return 0;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(m->device) != cudaSuccess) return 2;
    bool ok = release_model(m);
    if (previous != m->device && cudaSetDevice(previous) != cudaSuccess) ok = false;
    delete m; *opaque = nullptr;
    return ok ? 0 : 2;
}

extern "C" int spacepdhcg_gtoc12_completion_evaluate_compact_host(void* opaque, const void* model,
    int32_t count, int32_t nd, int32_t nl, const Policy* policy, const CompactCandidate* candidates,
    const CompactDeploy* deploys, const CompactLeg* legs, Result* results, LegResult* leg_results,
    double* collected, Stats* stats, Leg* expanded) {
    auto* w = static_cast<Workspace*>(opaque);
    const auto* m = static_cast<const Model*>(model);
    if (!correct_device(w) || !m || w->device != m->device || count < 0 || count > w->candidate_capacity ||
        nd < 0 || nd > w->deploy_capacity || nl < 0 || nl > w->leg_capacity) return 1;
    std::unique_lock<std::mutex> lock(w->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return 3;
    if (!count) { if (nd || nl) return 1; if (stats) *stats = {}; return 0; }
    const int status = validate_compact(m, count, nd, nl, policy, candidates, deploys, legs, results);
    if (status) return status;
    const auto failed = [&]() { cudaStreamSynchronize(w->stream); return 2; };
    const auto record = [&](int i) { return !stats || cudaEventRecord(w->events[i], w->stream) == cudaSuccess; };
    if (!record(0) || !transfer(w->compact_candidates, candidates, count, cudaMemcpyHostToDevice, w->stream) ||
        !transfer(w->compact_deploys, deploys, nd, cudaMemcpyHostToDevice, w->stream) ||
        !transfer(w->compact_legs, legs, nl, cudaMemcpyHostToDevice, w->stream) || !record(1)) return failed();
    const unsigned blocks = unsigned((int64_t(count) + 127) / 128);
    assemble_candidates<<<blocks, 128, 0, w->stream>>>(count, w->compact_candidates, w->compact_deploys, w->candidates, w->deploys);
    if (cudaGetLastError() != cudaSuccess) return failed();
    if (nl) {
        assemble_legs<<<unsigned((int64_t(nl) + 127) / 128), 128, 0, w->stream>>>(nl, *m,
            w->compact_candidates, w->compact_deploys, w->compact_legs, w->legs);
        if (cudaGetLastError() != cudaSuccess) return failed();
    }
    finish_candidates<<<blocks, 128, 0, w->stream>>>(count, *policy, w->candidates, w->deploys, w->legs,
        w->results, w->leg_results, w->collected, w->seen);
    if (cudaGetLastError() != cudaSuccess || !record(2) ||
        !transfer(results, w->results, count, cudaMemcpyDeviceToHost, w->stream) ||
        (leg_results && !transfer(leg_results, w->leg_results, nl, cudaMemcpyDeviceToHost, w->stream)) ||
        (collected && !transfer(collected, w->collected, nd, cudaMemcpyDeviceToHost, w->stream)) ||
        (expanded && !transfer(expanded, w->legs, nl, cudaMemcpyDeviceToHost, w->stream)) ||
        !record(3) || cudaStreamSynchronize(w->stream) != cudaSuccess) return failed();
    if (stats) {
        float upload = 0, kernel = 0, download = 0;
        if (cudaEventElapsedTime(&upload, w->events[0], w->events[1]) != cudaSuccess ||
            cudaEventElapsedTime(&kernel, w->events[1], w->events[2]) != cudaSuccess ||
            cudaEventElapsedTime(&download, w->events[2], w->events[3]) != cudaSuccess) return 2;
        *stats = {double(upload), double(kernel), double(download), uint64_t(count), uint64_t(nd), uint64_t(nl)};
    }
    return 0;
}
