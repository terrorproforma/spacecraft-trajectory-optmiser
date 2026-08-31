#pragma once

#include "spacepdhcg/orbitweaver/lambert.hpp"
#include "spacepdhcg/orbitweaver/trajectory_oracle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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
    double time_tolerance{1.0e-8};
    std::size_t maximum_iterations{256U};

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
        if (maximum_iterations == 0U) {
            throw std::invalid_argument("Lambert screening iteration limit must be positive");
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

}  // namespace lambert_oracle_detail

/// Host-executable analytical stage for the OrbitWeaver fidelity pipeline.
///
/// It solves the classical zero-revolution Lambert boundary-value problem against target
/// ephemerides, includes departure and arrival matching impulses, and closes propellant mass
/// with the rocket equation. The lower bound is deliberately conservative (`0`) until a
/// mission-specific admissible bound is supplied by a stronger screening stage.
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
        LambertSolution transfer{};
        try {
            transfer = solve_lambert_zero_revolution(
                departure.position,
                arrival.position,
                duration,
                config_.gravitational_parameter,
                config_.long_way,
                config_.time_tolerance,
                config_.maximum_iterations
            );
        } catch (const std::runtime_error& error) {
            ArcSolution infeasible{};
            infeasible.achieved_fidelity = ArcFidelity::analytical_screening;
            infeasible.diagnostics = std::string{"Lambert solve failed: "} + error.what();
            return infeasible;
        }
        const auto end = std::chrono::steady_clock::now();

        const auto departure_delta_v = lambert_oracle_detail::velocity_difference_norm(
            transfer.departure_velocity,
            departure.velocity
        );
        const auto arrival_delta_v = lambert_oracle_detail::velocity_difference_norm(
            arrival.velocity,
            transfer.arrival_velocity
        );
        const auto total_delta_v = departure_delta_v + arrival_delta_v;
        const auto propellant = propellant_required(
            request.initial_mass,
            total_delta_v,
            config_.specific_impulse_seconds
        );
        if (propellant >= request.initial_mass) {
            ArcSolution infeasible{};
            infeasible.achieved_fidelity = ArcFidelity::analytical_screening;
            infeasible.diagnostics = "Lambert transfer exceeds the available mass";
            return infeasible;
        }

        const auto residual = std::abs(transfer.time_of_flight_residual);
        const auto solve_seconds = std::chrono::duration<double>(end - start).count();
        ArcSolution result{
            true,
            ArcFidelity::analytical_screening,
            config_.cost_per_delta_v * total_delta_v,
            0.0,
            duration,
            total_delta_v,
            propellant,
            request.initial_mass - propellant,
            residual,
            0.0,
            std::max(residual, std::numeric_limits<double>::epsilon()),
            0U,
            transfer.iterations,
            0.0,
            solve_seconds,
            std::nullopt,
            "zero-revolution Lambert screening",
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
