#pragma once

#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/orbitweaver/low_thrust_oracle.hpp"
#include "spacepdhcg/orbitweaver/trajectory_oracle.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

struct HighFidelityLowThrustConfig {
    double gravitational_parameter{3.986004418e14};
    double equatorial_radius{6.3781363e6};
    double j2{1.08262668e-3};
    double thrust_to_acceleration{1.0};
    double mass_flow_coefficient{1.0e-4};
    double minimum_mass{1.0};
    double maximum_thrust{1.0};
    double minimum_radius{1.0};
    std::size_t intervals{100U};
    std::size_t substeps_per_interval{8U};
    double cost_per_delta_v{1.0};
    double cost_per_second{0.0};
    double feasibility_tolerance{1.0e-5};
    std::array<double, 7U> state_scales{
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0,
        1.0,
        1.0,
        1.0e-3,
    };

    void validate() const {
        for (const auto value : {
                 gravitational_parameter,
                 equatorial_radius,
                 thrust_to_acceleration,
                 mass_flow_coefficient,
                 minimum_mass,
                 maximum_thrust,
                 minimum_radius,
                 feasibility_tolerance,
             }) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument(
                    "high-fidelity low-thrust physical values must be finite and positive"
                );
            }
        }
        if (!std::isfinite(j2) || j2 < 0.0 || !std::isfinite(cost_per_delta_v)
            || cost_per_delta_v < 0.0 || !std::isfinite(cost_per_second)
            || cost_per_second < 0.0 || intervals < 2U
            || substeps_per_interval == 0U) {
            throw std::invalid_argument(
                "high-fidelity low-thrust certification configuration is invalid"
            );
        }
        if (!std::all_of(state_scales.begin(), state_scales.end(), [](const double value) {
                return std::isfinite(value) && value > 0.0;
            })) {
            throw std::invalid_argument(
                "high-fidelity low-thrust state scales must be finite and positive"
            );
        }
    }
};

struct HighFidelityPathDiagnostics {
    double thrust_epigraph{0.0};
    double throttle_upper{0.0};
    double minimum_mass{0.0};
    double minimum_radius{0.0};

    [[nodiscard]] double maximum_violation() const noexcept {
        return std::max(
            {thrust_epigraph, throttle_upper, minimum_mass, minimum_radius}
        );
    }
};

struct HighFidelityPropagation {
    std::vector<dynamics::LowThrustState> interval_states{};
    std::vector<dynamics::LowThrustState> dense_states{};
    HighFidelityPathDiagnostics path{};
};

struct HighFidelityCertification {
    HighFidelityPropagation propagation{};
    double integration_error{0.0};
    double terminal_error{0.0};
    double path_error{0.0};
    double delta_v{0.0};
};

class J2LowThrustPropagator {
  public:
    explicit J2LowThrustPropagator(HighFidelityLowThrustConfig config = {})
        : config_(config) {
        config_.validate();
    }

    [[nodiscard]] const HighFidelityLowThrustConfig& config() const noexcept {
        return config_;
    }

    [[nodiscard]] dynamics::LowThrustState derivative(
        const dynamics::LowThrustState& state,
        const dynamics::LowThrustControl& control
    ) const {
        validate_state_control(state, control);
        const auto x = state[0U];
        const auto y = state[1U];
        const auto z = state[2U];
        const auto radius_squared = x * x + y * y + z * z;
        const auto radius = std::sqrt(radius_squared);
        const auto inverse_radius_cubed = 1.0 / (radius_squared * radius);
        const auto central = -config_.gravitational_parameter * inverse_radius_cubed;
        const auto z_fraction = z * z / radius_squared;
        const auto j2_factor = 1.5 * config_.j2 * config_.gravitational_parameter
                               * config_.equatorial_radius * config_.equatorial_radius
                               / (radius_squared * radius_squared * radius);
        const std::array<double, 3U> acceleration{
            central * x + j2_factor * x * (5.0 * z_fraction - 1.0)
                + config_.thrust_to_acceleration * control[0U] / state[6U],
            central * y + j2_factor * y * (5.0 * z_fraction - 1.0)
                + config_.thrust_to_acceleration * control[1U] / state[6U],
            central * z + j2_factor * z * (5.0 * z_fraction - 3.0)
                + config_.thrust_to_acceleration * control[2U] / state[6U],
        };
        return dynamics::LowThrustState{
            state[3U],
            state[4U],
            state[5U],
            acceleration[0U],
            acceleration[1U],
            acceleration[2U],
            -config_.mass_flow_coefficient * control[3U],
        };
    }

    [[nodiscard]] dynamics::LowThrustState rk4_step(
        const dynamics::LowThrustState& state,
        const dynamics::LowThrustControl& control,
        const double step_seconds
    ) const {
        if (!std::isfinite(step_seconds) || step_seconds <= 0.0) {
            throw std::invalid_argument(
                "high-fidelity integration step must be finite and positive"
            );
        }
        const auto k1 = derivative(state, control);
        const auto k2 = derivative(add_scaled(state, k1, 0.5 * step_seconds), control);
        const auto k3 = derivative(add_scaled(state, k2, 0.5 * step_seconds), control);
        const auto k4 = derivative(add_scaled(state, k3, step_seconds), control);
        dynamics::LowThrustState next{};
        for (std::size_t component = 0; component < next.size(); ++component) {
            next[component] = state[component]
                              + step_seconds
                                    * (k1[component] + 2.0 * k2[component]
                                       + 2.0 * k3[component] + k4[component])
                                    / 6.0;
        }
        validate_state_control(next, control);
        return next;
    }

    [[nodiscard]] HighFidelityPropagation propagate(
        const dynamics::LowThrustState& initial,
        const std::vector<dynamics::LowThrustControl>& controls,
        const double interval_seconds,
        const std::size_t substeps_per_interval
    ) const {
        if (controls.size() != config_.intervals || substeps_per_interval == 0U
            || !std::isfinite(interval_seconds) || interval_seconds <= 0.0) {
            throw std::invalid_argument(
                "high-fidelity propagation horizon or substep count is invalid"
            );
        }
        const dynamics::LowThrustControl zero{0.0, 0.0, 0.0, 0.0};
        validate_state_control(initial, zero);
        HighFidelityPropagation result{};
        result.interval_states.reserve(controls.size() + 1U);
        result.dense_states.reserve(
            controls.size() * substeps_per_interval + 1U
        );
        result.interval_states.push_back(initial);
        result.dense_states.push_back(initial);
        update_path(result.path, initial, zero);
        auto current = initial;
        const auto substep = interval_seconds
                             / static_cast<double>(substeps_per_interval);
        for (const auto& control : controls) {
            for (std::size_t substep_index = 0;
                 substep_index < substeps_per_interval;
                 ++substep_index) {
                current = rk4_step(current, control, substep);
                result.dense_states.push_back(current);
                update_path(result.path, current, control);
            }
            result.interval_states.push_back(current);
        }
        return result;
    }

    [[nodiscard]] HighFidelityCertification certify(
        const dynamics::LowThrustState& initial,
        const dynamics::LowThrustState& target,
        const std::vector<dynamics::LowThrustControl>& controls,
        const double interval_seconds
    ) const {
        const auto coarse = propagate(
            initial,
            controls,
            interval_seconds,
            config_.substeps_per_interval
        );
        auto fine = propagate(
            initial,
            controls,
            interval_seconds,
            2U * config_.substeps_per_interval
        );
        double integration_error{0.0};
        for (std::size_t node = 0; node < fine.interval_states.size(); ++node) {
            for (std::size_t component = 0; component < 7U; ++component) {
                integration_error = std::max(
                    integration_error,
                    std::abs(
                        (fine.interval_states[node][component]
                         - coarse.interval_states[node][component])
                        * config_.state_scales[component]
                    )
                );
            }
        }
        double terminal_error{0.0};
        for (std::size_t component = 0; component < 6U; ++component) {
            terminal_error = std::max(
                terminal_error,
                std::abs(
                    (fine.interval_states.back()[component] - target[component])
                    * config_.state_scales[component]
                )
            );
        }
        const auto position_scale = std::max(
            {config_.state_scales[0U],
             config_.state_scales[1U],
             config_.state_scales[2U]}
        );
        const auto path_error = std::max(
            {fine.path.thrust_epigraph / config_.maximum_thrust,
             fine.path.throttle_upper / config_.maximum_thrust,
             fine.path.minimum_mass * config_.state_scales[6U],
             fine.path.minimum_radius * position_scale}
        );
        double delta_v{0.0};
        for (std::size_t interval = 0; interval < controls.size(); ++interval) {
            delta_v += interval_seconds * config_.thrust_to_acceleration
                       * controls[interval][3U]
                       / fine.interval_states[interval][6U];
        }
        return HighFidelityCertification{
            std::move(fine),
            integration_error,
            terminal_error,
            path_error,
            delta_v,
        };
    }

  private:
    HighFidelityLowThrustConfig config_{};

    static dynamics::LowThrustState add_scaled(
        const dynamics::LowThrustState& state,
        const dynamics::LowThrustState& derivative_value,
        const double scale
    ) noexcept {
        dynamics::LowThrustState result{};
        for (std::size_t component = 0; component < result.size(); ++component) {
            result[component] = state[component] + scale * derivative_value[component];
        }
        return result;
    }

    void validate_state_control(
        const dynamics::LowThrustState& state,
        const dynamics::LowThrustControl& control
    ) const {
        if (!std::all_of(state.begin(), state.end(), [](const double value) {
                return std::isfinite(value);
            })
            || !std::all_of(control.begin(), control.end(), [](const double value) {
                   return std::isfinite(value);
               })) {
            throw std::invalid_argument(
                "high-fidelity low-thrust state and control must be finite"
            );
        }
        const auto radius = std::sqrt(
            state[0U] * state[0U] + state[1U] * state[1U]
            + state[2U] * state[2U]
        );
        if (radius <= 0.0 || state[6U] <= 0.0) {
            throw std::invalid_argument(
                "high-fidelity low-thrust radius and mass must be positive"
            );
        }
    }

    void update_path(
        HighFidelityPathDiagnostics& diagnostics,
        const dynamics::LowThrustState& state,
        const dynamics::LowThrustControl& control
    ) const noexcept {
        const auto thrust_norm = std::sqrt(
            control[0U] * control[0U] + control[1U] * control[1U]
            + control[2U] * control[2U]
        );
        const auto radius = std::sqrt(
            state[0U] * state[0U] + state[1U] * state[1U]
            + state[2U] * state[2U]
        );
        diagnostics.thrust_epigraph = std::max(
            diagnostics.thrust_epigraph,
            thrust_norm - control[3U]
        );
        diagnostics.throttle_upper = std::max(
            diagnostics.throttle_upper,
            control[3U] - config_.maximum_thrust
        );
        diagnostics.minimum_mass = std::max(
            diagnostics.minimum_mass,
            config_.minimum_mass - state[6U]
        );
        diagnostics.minimum_radius = std::max(
            diagnostics.minimum_radius,
            config_.minimum_radius - radius
        );
    }
};

/// Independent certified-high-fidelity OrbitWeaver stage using dense J2 propagation and
/// step-doubling integration error estimation. Controls are obtained from the preceding
/// low-thrust warm-reference token; no SCvx model state is trusted for certification.
class HighFidelityLowThrustOrbitStage {
  public:
    HighFidelityLowThrustOrbitStage(
        EphemerisProvider ephemeris,
        std::shared_ptr<LowThrustWarmStartStore> store,
        HighFidelityLowThrustConfig config = {}
    )
        : ephemeris_(std::move(ephemeris)),
          store_(std::move(store)),
          config_(config),
          propagator_(config_) {
        if (!ephemeris_ || !store_) {
            throw std::invalid_argument(
                "high-fidelity stage requires ephemeris and warm-reference store"
            );
        }
        config_.validate();
    }

    [[nodiscard]] FidelityPipelineOracle::Stage stage() const {
        return [*this](
                   const ArcRequest& request,
                   const std::optional<ArcSolution>& previous
               ) { return evaluate(request, previous); };
    }

    void register_stage(FidelityPipelineOracle& pipeline) const {
        pipeline.register_stage(ArcFidelity::certified, stage());
    }

    [[nodiscard]] ArcSolution evaluate(
        const ArcRequest& request,
        const std::optional<ArcSolution>& previous = std::nullopt
    ) const {
        request.validate();
        if (request.fidelity != ArcFidelity::certified
            || !request.arrival_epoch.has_value()) {
            throw std::invalid_argument(
                "high-fidelity low-thrust stage requires certified fidelity and arrival epoch"
            );
        }
        const auto token = request.warm_start_token.has_value()
                               ? request.warm_start_token
                               : previous.has_value() ? previous->warm_start_token
                                                      : std::nullopt;
        if (!token.has_value()) {
            return infeasible(
                "high-fidelity certification requires a low-thrust warm-reference token"
            );
        }
        const auto reference = store_->get(*token, request, config_.intervals);
        if (!reference.has_value()) {
            return infeasible(
                "high-fidelity certification token is missing or request-incompatible"
            );
        }
        const auto departure = ephemeris_(request.from_target, request.departure_epoch);
        const auto arrival = ephemeris_(request.to_target, *request.arrival_epoch);
        validate_ephemeris(departure);
        validate_ephemeris(arrival);
        const dynamics::LowThrustState initial{
            departure.position[0U],
            departure.position[1U],
            departure.position[2U],
            departure.velocity[0U],
            departure.velocity[1U],
            departure.velocity[2U],
            request.initial_mass,
        };
        const dynamics::LowThrustState target{
            arrival.position[0U],
            arrival.position[1U],
            arrival.position[2U],
            arrival.velocity[0U],
            arrival.velocity[1U],
            arrival.velocity[2U],
            request.initial_mass,
        };
        const auto duration = *request.arrival_epoch - request.departure_epoch;
        const auto interval_seconds =
            duration / static_cast<double>(config_.intervals);
        const auto start = std::chrono::steady_clock::now();
        HighFidelityCertification certification{};
        try {
            certification = propagator_.certify(
                initial,
                target,
                reference->second,
                interval_seconds
            );
        } catch (const std::exception& error) {
            return infeasible(
                std::string{"high-fidelity propagation failed: "} + error.what()
            );
        }
        const auto stop = std::chrono::steady_clock::now();
        const auto achieved = std::max(
            {certification.integration_error,
             certification.terminal_error,
             certification.path_error,
             std::numeric_limits<double>::epsilon()}
        );
        const auto acceptance = std::max(
            request.requested_tolerance,
            config_.feasibility_tolerance
        );
        if (certification.integration_error > acceptance
            || certification.terminal_error > acceptance
            || certification.path_error > acceptance) {
            return infeasible(
                "high-fidelity J2 certification exceeded the requested acceptance limit"
            );
        }
        const auto final_mass = certification.propagation.interval_states.back()[6U];
        const auto propellant = std::max(0.0, request.initial_mass - final_mass);
        if (propellant >= request.initial_mass) {
            return infeasible("high-fidelity certification consumed all spacecraft mass");
        }
        const auto cost = config_.cost_per_delta_v * certification.delta_v
                          + config_.cost_per_second * duration;
        const auto lower_bound = previous.has_value()
                                     ? std::min(cost, previous->lower_bound)
                                     : 0.0;
        ArcSolution result{
            true,
            ArcFidelity::certified,
            cost,
            lower_bound,
            duration,
            certification.delta_v,
            propellant,
            final_mass,
            certification.terminal_error,
            certification.path_error,
            achieved,
            0U,
            2U * config_.intervals * config_.substeps_per_interval,
            0.0,
            std::chrono::duration<double>(stop - start).count(),
            token,
            std::string{"independent J2 RK4 certification, substeps="}
                + std::to_string(2U * config_.substeps_per_interval)
                + ", integration_error="
                + std::to_string(certification.integration_error),
        };
        result.validate(request);
        return result;
    }

    [[nodiscard]] const HighFidelityLowThrustConfig& config() const noexcept {
        return config_;
    }

  private:
    EphemerisProvider ephemeris_{};
    std::shared_ptr<LowThrustWarmStartStore> store_{};
    HighFidelityLowThrustConfig config_{};
    J2LowThrustPropagator propagator_{};

    static void validate_ephemeris(const CartesianEphemerisState& state) {
        for (const auto value : state.position) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "high-fidelity ephemeris position is non-finite"
                );
            }
        }
        for (const auto value : state.velocity) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "high-fidelity ephemeris velocity is non-finite"
                );
            }
        }
    }

    static ArcSolution infeasible(std::string diagnostics) {
        ArcSolution result{};
        result.achieved_fidelity = ArcFidelity::certified;
        result.diagnostics = std::move(diagnostics);
        return result;
    }
};

}  // namespace spacepdhcg::orbitweaver
