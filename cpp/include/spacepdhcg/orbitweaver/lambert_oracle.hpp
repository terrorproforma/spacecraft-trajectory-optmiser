#pragma once

#include "spacepdhcg/orbitweaver/lambert.hpp"
#include "spacepdhcg/orbitweaver/lambert_family.hpp"
#include "spacepdhcg/orbitweaver/trajectory_oracle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct CartesianEphemerisState {
    Vector3 position{};
    Vector3 velocity{};
};

using EphemerisProvider = std::function<CartesianEphemerisState(
    std::size_t target,
    double epoch
)>;

struct LambertScreeningConfig {
    double gravitational_parameter{3.986004418e14};
    double specific_impulse_seconds{300.0};
    double cost_per_delta_v{1.0};
    bool long_way{false};
    bool evaluate_both_directions{false};
    std::size_t maximum_revolutions{0U};
    double time_tolerance{1.0e-8};
    std::size_t maximum_iterations{256U};
    std::size_t scan_samples_per_revolution{8'192U};

    void validate() const {
        for (const auto value : {
                 gravitational_parameter,
                 specific_impulse_seconds,
                 cost_per_delta_v,
                 time_tolerance,
             }) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument(
                    "Lambert screening configuration values must be finite and positive"
                );
            }
        }
        if (maximum_iterations == 0U || scan_samples_per_revolution < 16U) {
            throw std::invalid_argument(
                "Lambert screening iteration and scan limits are invalid"
            );
        }
    }
};

namespace lambert_oracle_detail {

inline double velocity_difference_norm(const Vector3& left, const Vector3& right) noexcept {
    const auto dx = left[0U] - right[0U];
    const auto dy = left[1U] - right[1U];
    const auto dz = left[2U] - right[2U];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline void validate_ephemeris(const CartesianEphemerisState& state) {
    for (const auto value : state.position) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("ephemeris provider returned a non-finite position");
        }
    }
    for (const auto value : state.velocity) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("ephemeris provider returned a non-finite velocity");
        }
    }
}

inline std::string branch_name(const LambertParameterBranch branch) {
    switch (branch) {
        case LambertParameterBranch::unique:
            return "unique";
        case LambertParameterBranch::lower_parameter:
            return "lower-parameter";
        case LambertParameterBranch::higher_parameter:
            return "higher-parameter";
    }
    return "unknown";
}

struct ScreenedFamilyMember {
    LambertFamilyMember member{};
    double departure_delta_v{0.0};
    double arrival_delta_v{0.0};
    double total_delta_v{0.0};
    double propellant{0.0};
};

inline ScreenedFamilyMember score(
    const LambertFamilyMember& member,
    const CartesianEphemerisState& departure,
    const CartesianEphemerisState& arrival,
    const double initial_mass,
    const double specific_impulse_seconds
) {
    const auto departure_delta_v = velocity_difference_norm(
        member.solution.departure_velocity,
        departure.velocity
    );
    const auto arrival_delta_v = velocity_difference_norm(
        arrival.velocity,
        member.solution.arrival_velocity
    );
    const auto total_delta_v = departure_delta_v + arrival_delta_v;
    return ScreenedFamilyMember{
        member,
        departure_delta_v,
        arrival_delta_v,
        total_delta_v,
        propellant_required(
            initial_mass,
            total_delta_v,
            specific_impulse_seconds
        ),
    };
}

}  // namespace lambert_oracle_detail

/// Host-executable analytical stage for the OrbitWeaver fidelity pipeline.
///
/// It enumerates the requested zero- and multi-revolution universal-variable Lambert
/// families, evaluates short/long transfer directions when configured, and selects the
/// minimum matching-impulse candidate that closes the rocket equation. The lower bound is
/// deliberately conservative (`0`) until a mission-specific admissible bound is supplied by
/// a stronger screening stage.
class LambertScreeningOracle final : public TrajectoryOracle {
  public:
    LambertScreeningOracle(
        EphemerisProvider ephemeris,
        LambertScreeningConfig config = LambertScreeningConfig{}
    )
        : ephemeris_(std::move(ephemeris)), config_(config) {
        if (!ephemeris_) {
            throw std::invalid_argument("Lambert screening oracle requires an ephemeris provider");
        }
        config_.validate();
    }

    [[nodiscard]] ArcSolution evaluate(const ArcRequest& request) override {
        request.validate();
        if (request.fidelity != ArcFidelity::analytical_screening) {
            throw std::invalid_argument(
                "Lambert screening oracle only implements analytical screening fidelity"
            );
        }
        if (!request.arrival_epoch.has_value()) {
            throw std::invalid_argument(
                "Lambert screening requires an explicit arrival epoch"
            );
        }

        const auto departure = ephemeris_(request.from_target, request.departure_epoch);
        const auto arrival = ephemeris_(request.to_target, *request.arrival_epoch);
        lambert_oracle_detail::validate_ephemeris(departure);
        lambert_oracle_detail::validate_ephemeris(arrival);
        const auto duration = *request.arrival_epoch - request.departure_epoch;

        const auto start = std::chrono::steady_clock::now();
        std::vector<LambertFamilyMember> family{};
        try {
            if (config_.evaluate_both_directions) {
                family = enumerate_lambert_families(
                    departure.position,
                    arrival.position,
                    duration,
                    config_.gravitational_parameter,
                    config_.maximum_revolutions,
                    true,
                    true,
                    config_.time_tolerance,
                    config_.maximum_iterations,
                    config_.scan_samples_per_revolution
                );
            } else {
                family = solve_lambert_revolution_family(
                    departure.position,
                    arrival.position,
                    duration,
                    config_.gravitational_parameter,
                    config_.maximum_revolutions,
                    config_.long_way,
                    config_.time_tolerance,
                    config_.maximum_iterations,
                    config_.scan_samples_per_revolution
                );
            }
        } catch (const std::runtime_error& error) {
            ArcSolution infeasible{};
            infeasible.achieved_fidelity = ArcFidelity::analytical_screening;
            infeasible.diagnostics = std::string{"Lambert family solve failed: "} + error.what();
            return infeasible;
        }
        const auto end = std::chrono::steady_clock::now();

        std::optional<lambert_oracle_detail::ScreenedFamilyMember> best{};
        for (const auto& member : family) {
            const auto scored = lambert_oracle_detail::score(
                member,
                departure,
                arrival,
                request.initial_mass,
                config_.specific_impulse_seconds
            );
            if (scored.propellant >= request.initial_mass) {
                continue;
            }
            if (!best.has_value() || scored.total_delta_v < best->total_delta_v) {
                best = scored;
            }
        }
        if (!best.has_value()) {
            ArcSolution infeasible{};
            infeasible.achieved_fidelity = ArcFidelity::analytical_screening;
            infeasible.diagnostics = "every Lambert family member exceeds the available mass";
            return infeasible;
        }

        const auto residual = std::abs(best->member.solution.time_of_flight_residual);
        const auto solve_seconds = std::chrono::duration<double>(end - start).count();
        const auto direction = best->member.long_way ? "long-way" : "short-way";
        ArcSolution result{
            true,
            ArcFidelity::analytical_screening,
            config_.cost_per_delta_v * best->total_delta_v,
            0.0,
            duration,
            best->total_delta_v,
            best->propellant,
            request.initial_mass - best->propellant,
            residual,
            0.0,
            std::max(residual, std::numeric_limits<double>::epsilon()),
            0U,
            best->member.solution.iterations,
            0.0,
            solve_seconds,
            std::nullopt,
            std::string{"Lambert family screening: "} + direction
                + ", revolutions=" + std::to_string(best->member.revolutions)
                + ", branch="
                + lambert_oracle_detail::branch_name(best->member.branch),
        };
        result.validate(request);
        return result;
    }

    [[nodiscard]] const LambertScreeningConfig& config() const noexcept { return config_; }

  private:
    EphemerisProvider ephemeris_{};
    LambertScreeningConfig config_{};
};

}  // namespace spacepdhcg::orbitweaver
