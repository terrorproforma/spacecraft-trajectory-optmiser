#pragma once

// Planner problem document (schema 1.0.0) -> validated native problem model.
//
// The document is authored in canonical family units (see `family_units`):
//   hcw                 : m, m/s, m/s^2, rad/s
//   powered_descent_3dof: m, m/s, kg, N, rad
//   powered_descent_6dof: m, m/s, kg, N, N*m, rad, rad/s
//   low_thrust          : km, km/s, kg, N (thrust_to_acceleration converts N/kg -> km/s^2)
// The Python layer performs unit conversion before handing the document to the
// native code; the native parser only accepts canonical units and fails closed
// on anything the frozen transcriptions cannot represent.

#include "spacepdhcg/planner/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace spacepdhcg::planner {

inline constexpr std::string_view schema_version = "1.0.0";

enum class Family : std::uint8_t {
    hcw,
    powered_descent_3dof,
    powered_descent_6dof,
    low_thrust,
};

enum class Backend : std::uint8_t {
    pure_qoco,
    pdhcg,
    pdhcg_recovery,
    cpu_reference,
};

enum class Preset : std::uint8_t {
    frozen_adaptive_pure_qoco,
    frozen_adaptive_pdhcg,
    fixed_tight_pdhcg,
};

class ProblemError final : public std::invalid_argument {
  public:
    explicit ProblemError(const std::string& message) : std::invalid_argument(message) {}
};

[[nodiscard]] inline Family parse_family(std::string_view name) {
    if (name == "hcw") {
        return Family::hcw;
    }
    if (name == "powered_descent_3dof") {
        return Family::powered_descent_3dof;
    }
    if (name == "powered_descent_6dof") {
        return Family::powered_descent_6dof;
    }
    if (name == "low_thrust") {
        return Family::low_thrust;
    }
    throw ProblemError(
        "unknown family '" + std::string(name)
        + "'; expected hcw, powered_descent_3dof, powered_descent_6dof, or low_thrust"
    );
}

[[nodiscard]] inline std::string_view family_name(Family family) noexcept {
    switch (family) {
        case Family::hcw:
            return "hcw";
        case Family::powered_descent_3dof:
            return "powered_descent_3dof";
        case Family::powered_descent_6dof:
            return "powered_descent_6dof";
        case Family::low_thrust:
            return "low_thrust";
    }
    return "unknown";
}

[[nodiscard]] inline Backend parse_backend(std::string_view name) {
    if (name == "pure_qoco") {
        return Backend::pure_qoco;
    }
    if (name == "pdhcg") {
        return Backend::pdhcg;
    }
    if (name == "pdhcg_recovery") {
        return Backend::pdhcg_recovery;
    }
    if (name == "cpu_reference") {
        return Backend::cpu_reference;
    }
    throw ProblemError(
        "unknown backend '" + std::string(name)
        + "'; expected pure_qoco, pdhcg, pdhcg_recovery, or cpu_reference"
    );
}

[[nodiscard]] inline std::string_view backend_name(Backend backend) noexcept {
    switch (backend) {
        case Backend::pure_qoco:
            return "pure_qoco";
        case Backend::pdhcg:
            return "pdhcg";
        case Backend::pdhcg_recovery:
            return "pdhcg_recovery";
        case Backend::cpu_reference:
            return "cpu_reference";
    }
    return "unknown";
}

[[nodiscard]] inline Preset parse_preset(std::string_view name) {
    if (name == "frozen_adaptive_pure_qoco") {
        return Preset::frozen_adaptive_pure_qoco;
    }
    if (name == "frozen_adaptive_pdhcg") {
        return Preset::frozen_adaptive_pdhcg;
    }
    if (name == "fixed_tight_pdhcg") {
        return Preset::fixed_tight_pdhcg;
    }
    throw ProblemError(
        "unknown solver preset '" + std::string(name)
        + "'; expected frozen_adaptive_pure_qoco, frozen_adaptive_pdhcg, or fixed_tight_pdhcg"
    );
}

[[nodiscard]] inline std::string_view preset_name(Preset preset) noexcept {
    switch (preset) {
        case Preset::frozen_adaptive_pure_qoco:
            return "frozen_adaptive_pure_qoco";
        case Preset::frozen_adaptive_pdhcg:
            return "frozen_adaptive_pdhcg";
        case Preset::fixed_tight_pdhcg:
            return "fixed_tight_pdhcg";
    }
    return "unknown";
}

[[nodiscard]] inline std::size_t state_dimension(Family family) noexcept {
    switch (family) {
        case Family::hcw:
            return 6U;
        case Family::powered_descent_3dof:
        case Family::low_thrust:
            return 7U;
        case Family::powered_descent_6dof:
            return 14U;
    }
    return 0U;
}

[[nodiscard]] inline std::size_t control_dimension(Family family) noexcept {
    switch (family) {
        case Family::hcw:
            return 3U;
        case Family::powered_descent_3dof:
        case Family::low_thrust:
            return 4U;
        case Family::powered_descent_6dof:
            return 7U;
    }
    return 0U;
}

/// Number of leading terminal components pinned by the frozen transcription.
[[nodiscard]] inline std::size_t terminal_dimension(Family family) noexcept {
    switch (family) {
        case Family::hcw:
            return 6U;
        case Family::powered_descent_3dof:
        case Family::low_thrust:
            return 6U;
        case Family::powered_descent_6dof:
            return 13U;
    }
    return 0U;
}

/// Vehicle/environment/constraint parameters after family defaults are applied.
struct VehicleParameters {
    // Shared
    double minimum_mass{0.0};
    double maximum_thrust{0.0};
    double minimum_thrust{0.0};
    double mass_flow_coefficient{0.0};  // kg/s per unit thrust = 1 / exhaust velocity
    // Powered descent
    std::array<double, 3U> gravity{0.0, 0.0, 0.0};
    double maximum_tilt_radians{0.0};
    double glide_slope_radians{0.0};
    double minimum_altitude{0.0};
    // 6-DoF
    std::array<double, 3U> principal_inertia{0.0, 0.0, 0.0};
    double maximum_torque{0.0};
    double maximum_angular_rate{0.0};
    // HCW
    double mean_motion{0.0};
    double maximum_acceleration{0.0};
    bool acceleration_norm_bound{true};
    // Low thrust
    double gravitational_parameter{0.0};
    double thrust_to_acceleration{0.0};
    double minimum_radius{0.0};
};

/// Transcription weights (fixed CQP coefficients that are not solver policy).
struct TranscriptionWeights {
    double virtual_l1_weight{0.0};
    double virtual_quadratic_weight{0.0};
    double virtual_epigraph_regularisation{0.0};
    double fuel_weight{0.0};
    std::vector<double> state_tracking_weights{};
    std::vector<double> control_tracking_weights{};
    std::vector<double> state_trust_scales{};
    std::vector<double> control_trust_scales{};
    // HCW only
    std::vector<double> state_weights{};
    std::vector<double> control_weights{};
};

struct TrustRegionPolicy {
    double initial_radius{1.0};
    double minimum_radius{1.0e-4};
    double maximum_radius{8.0};
    double shrink_factor{0.5};
    double expansion_factor{1.8};
    double acceptance_threshold{0.05};
    double strong_agreement_threshold{0.75};
    double near_boundary_fraction{0.8};
    double restoration_reduction{0.9};
};

struct PenaltyPolicy {
    double feasibility_penalty{100.0};
    double virtual_penalty{0.0};  // defaults to the transcription virtual L1 weight
};

struct ForcingPolicy {
    double epsilon_max{1.0e-3};
    double epsilon_floor{1.0e-8};
    double epsilon_0{1.0e-3};
    double coefficient{0.2};
    double alpha{0.5};
    double gamma{0.6};
    double repair_ceiling{1.0e-2};
    double progress_ceiling{2.0e-3};
    double refinement_ceiling{1.0e-5};
    double polish_ceiling{1.0e-8};
    std::uint64_t repair_iterations{5'000U};
    std::uint64_t progress_iterations{25'000U};
    std::uint64_t refinement_iterations{100'000U};
    std::uint64_t polish_iterations{1'000'000U};
    double resolve_trigger_multiple{5.0};
    double resolve_refinement_factor{0.1};
    double resolve_minimum_tolerance{1.0e-8};
    std::uint32_t maximum_resolves{1U};
    double fixed_inner_tolerance{0.0};
    std::uint64_t fixed_inner_iteration_limit{0U};
    double final_polish_tolerance{1.0e-8};
    std::uint64_t final_polish_iteration_limit{1'000'000U};
};

struct SolverOptions {
    Backend backend{Backend::pure_qoco};
    Preset preset{Preset::frozen_adaptive_pure_qoco};
    double tolerance{1.0e-6};
    double step_tolerance{2.0e-2};
    std::uint32_t maximum_outer_iterations{30U};
    std::uint32_t minimum_outer_iterations{1U};
    double time_limit_seconds{0.0};  // 0 = unlimited
    double certificate_tolerance{1.0e-6};
    double replay_parity_tolerance{1.0e-9};
    TrustRegionPolicy trust{};
    PenaltyPolicy penalty{};
    ForcingPolicy forcing{};
    std::string warm_start_mode{"primal"};  // none | primal | primal_dual | full_retained
};

struct OutputOptions {
    std::size_t dense_replay_substeps{10U};
    bool include_iterations{true};
};

struct WarmStart {
    std::vector<double> states{};    // (N+1) * nx flat
    std::vector<double> controls{};  // N * nu flat
};

struct PlannerProblem {
    std::string name{};
    Family family{Family::powered_descent_3dof};
    std::size_t intervals{0U};
    double final_time{0.0};
    double step_seconds{0.0};
    std::vector<double> initial_state{};
    std::vector<double> target_state{};
    std::vector<bool> terminal_fixed{};
    VehicleParameters vehicle{};
    TranscriptionWeights weights{};
    SolverOptions solver{};
    OutputOptions output{};
    std::optional<WarmStart> warm_start{};
    json::Value document{};  // normalised copy retained for echo/provenance

    [[nodiscard]] std::size_t nx() const noexcept { return state_dimension(family); }
    [[nodiscard]] std::size_t nu() const noexcept { return control_dimension(family); }
};

namespace detail {

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw ProblemError(message);
    }
}

inline double positive(const json::Value& object, std::string_view key, double fallback) {
    const double value = json::number_or(object, key, fallback);
    require(
        std::isfinite(value) && value > 0.0,
        "'" + std::string(key) + "' must be finite and positive"
    );
    return value;
}

inline double non_negative(const json::Value& object, std::string_view key, double fallback) {
    const double value = json::number_or(object, key, fallback);
    require(
        std::isfinite(value) && value >= 0.0,
        "'" + std::string(key) + "' must be finite and non-negative"
    );
    return value;
}

inline std::vector<double> positive_vector(
    const json::Value& object,
    std::string_view key,
    const std::vector<double>& fallback
) {
    const auto* member = object.find(key);
    if (member == nullptr || member->is_null()) {
        return fallback;
    }
    const auto values = json::numbers_at(object, key, fallback.size());
    for (const double value : values) {
        require(
            std::isfinite(value) && value > 0.0,
            "'" + std::string(key) + "' entries must be finite and positive"
        );
    }
    return values;
}

inline std::array<double, 3U> vector3(
    const json::Value& object,
    std::string_view key,
    const std::array<double, 3U>& fallback
) {
    const auto* member = object.find(key);
    if (member == nullptr || member->is_null()) {
        return fallback;
    }
    const auto values = json::numbers_at(object, key, 3U);
    for (const double value : values) {
        require(std::isfinite(value), "'" + std::string(key) + "' entries must be finite");
    }
    return {values[0U], values[1U], values[2U]};
}

inline const json::Value& section(const json::Value& document, std::string_view key) {
    static const json::Value empty = json::Value::object();
    const auto* value = document.find(key);
    if (value == nullptr || value->is_null()) {
        return empty;
    }
    require(value->is_object(), "'" + std::string(key) + "' must be an object");
    return *value;
}

inline double mass_flow_from(const json::Value& vehicle, double fallback) {
    const auto* coefficient = vehicle.find("mass_flow_coefficient");
    const auto* exhaust = vehicle.find("exhaust_velocity");
    const auto* isp = vehicle.find("specific_impulse");
    int supplied = 0;
    supplied += coefficient != nullptr && !coefficient->is_null() ? 1 : 0;
    supplied += exhaust != nullptr && !exhaust->is_null() ? 1 : 0;
    supplied += isp != nullptr && !isp->is_null() ? 1 : 0;
    require(
        supplied <= 1,
        "specify at most one of mass_flow_coefficient, exhaust_velocity, specific_impulse"
    );
    if (coefficient != nullptr && !coefficient->is_null()) {
        return positive(vehicle, "mass_flow_coefficient", fallback);
    }
    if (exhaust != nullptr && !exhaust->is_null()) {
        return 1.0 / positive(vehicle, "exhaust_velocity", 1.0);
    }
    if (isp != nullptr && !isp->is_null()) {
        constexpr double standard_gravity = 9.80665;
        return 1.0 / (positive(vehicle, "specific_impulse", 1.0) * standard_gravity);
    }
    return fallback;
}

}  // namespace detail

/// Family defaults reused by the Python layer and the native executable.
[[nodiscard]] inline VehicleParameters default_vehicle(Family family) {
    VehicleParameters vehicle{};
    switch (family) {
        case Family::hcw:
            vehicle.mean_motion = 1.13e-3;
            vehicle.maximum_acceleration = 5.0e-2;
            vehicle.acceleration_norm_bound = true;
            break;
        case Family::powered_descent_3dof:
            vehicle.gravity = {0.0, 0.0, -3.711};
            vehicle.mass_flow_coefficient = 4.6e-4;
            vehicle.minimum_mass = 1'000.0;
            vehicle.maximum_thrust = 15'000.0;
            vehicle.minimum_thrust = 0.0;
            vehicle.maximum_tilt_radians = 0.5235987755982988;
            vehicle.glide_slope_radians = 1.0471975511965976;
            break;
        case Family::powered_descent_6dof:
            vehicle.gravity = {0.0, 0.0, -3.711};
            vehicle.principal_inertia = {2'500.0, 2'200.0, 1'800.0};
            vehicle.mass_flow_coefficient = 4.6e-4;
            vehicle.minimum_mass = 1'000.0;
            vehicle.maximum_thrust = 15'000.0;
            vehicle.minimum_thrust = 0.0;
            vehicle.maximum_torque = 2'000.0;
            vehicle.maximum_angular_rate = 1.0;
            vehicle.maximum_tilt_radians = 0.5235987755982988;
            vehicle.glide_slope_radians = 1.0471975511965976;
            break;
        case Family::low_thrust:
            vehicle.gravitational_parameter = 398'600.4418;
            vehicle.thrust_to_acceleration = 1.0e-3;
            vehicle.mass_flow_coefficient = 3.4e-5;
            vehicle.minimum_mass = 200.0;
            vehicle.maximum_thrust = 1.0;
            vehicle.minimum_radius = 6'500.0;
            break;
    }
    return vehicle;
}

[[nodiscard]] inline TranscriptionWeights default_weights(
    Family family,
    const VehicleParameters& vehicle
) {
    TranscriptionWeights weights{};
    switch (family) {
        case Family::hcw:
            weights.state_weights = {1.0e-4, 1.0e-4, 1.0e-4, 1.0e-2, 1.0e-2, 1.0e-2};
            weights.control_weights = {1.0, 1.0, 1.0};
            break;
        case Family::powered_descent_3dof: {
            // Qualified production fixture (device_scvx_integration_test run_pd3).
            weights.virtual_l1_weight = 10.0;
            weights.virtual_quadratic_weight = 1.0e-3;
            weights.virtual_epigraph_regularisation = 1.0e-3;
            weights.fuel_weight = 1.0e-3;
            weights.state_tracking_weights =
                {1.0e-4, 1.0e-4, 1.0e-4, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-8};
            weights.control_tracking_weights = {1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8};
            weights.state_trust_scales = {1.0e-3, 1.0e-3, 1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-3};
            const double thrust_scale = 1.0 / vehicle.maximum_thrust;
            weights.control_trust_scales = {thrust_scale, thrust_scale, thrust_scale, thrust_scale};
            break;
        }
        case Family::powered_descent_6dof: {
            weights.virtual_l1_weight = 10.0;
            weights.virtual_quadratic_weight = 1.0e-3;
            weights.virtual_epigraph_regularisation = 1.0e-3;
            weights.fuel_weight = 1.0e-3;
            weights.state_tracking_weights = {
                1.0e-4, 1.0e-4, 1.0e-4, 1.0e-2, 1.0e-2, 1.0e-2, 1.0, 1.0, 1.0, 1.0,
                1.0e-2, 1.0e-2, 1.0e-2, 1.0e-8,
            };
            weights.control_tracking_weights =
                {1.0e-8, 1.0e-8, 1.0e-8, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-8};
            weights.state_trust_scales = {
                1.0e-3, 1.0e-3, 1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0e-3,
            };
            const double thrust_scale = 1.0 / vehicle.maximum_thrust;
            const double torque_scale = 1.0 / vehicle.maximum_torque;
            weights.control_trust_scales = {
                thrust_scale, thrust_scale, thrust_scale,
                torque_scale, torque_scale, torque_scale,
                thrust_scale,
            };
            break;
        }
        case Family::low_thrust:
            // Frozen low-thrust transcription defaults (LowThrustScvxConfig).
            weights.virtual_l1_weight = 1.0e6;
            weights.virtual_quadratic_weight = 1.0e-8;
            weights.virtual_epigraph_regularisation = 1.0e-10;
            weights.fuel_weight = 1.0;
            weights.state_tracking_weights =
                {1.0e-6, 1.0e-6, 1.0e-6, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-8};
            weights.control_tracking_weights = {1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3};
            weights.state_trust_scales = {1.0e-3, 1.0e-3, 1.0e-3, 1.0, 1.0, 1.0, 1.0e-3};
            weights.control_trust_scales = {1.0, 1.0, 1.0, 1.0};
            break;
    }
    return weights;
}

/// Terminal components the frozen transcriptions pin (true) or leave free (false).
[[nodiscard]] inline std::vector<bool> supported_terminal_pattern(Family family) {
    std::vector<bool> pattern(state_dimension(family), false);
    for (std::size_t component = 0U; component < terminal_dimension(family); ++component) {
        pattern[component] = true;
    }
    return pattern;
}

[[nodiscard]] inline std::vector<std::string> state_names(Family family) {
    switch (family) {
        case Family::hcw:
            return {"x", "y", "z", "vx", "vy", "vz"};
        case Family::powered_descent_3dof:
        case Family::low_thrust:
            return {"x", "y", "z", "vx", "vy", "vz", "mass"};
        case Family::powered_descent_6dof:
            return {
                "x", "y", "z", "vx", "vy", "vz", "q0", "q1", "q2", "q3",
                "wx", "wy", "wz", "mass",
            };
    }
    return {};
}

[[nodiscard]] inline std::vector<std::string> control_names(Family family) {
    switch (family) {
        case Family::hcw:
            return {"ax", "ay", "az"};
        case Family::powered_descent_3dof:
        case Family::low_thrust:
            return {"thrust_x", "thrust_y", "thrust_z", "sigma"};
        case Family::powered_descent_6dof:
            return {
                "thrust_x", "thrust_y", "thrust_z", "torque_x", "torque_y", "torque_z", "sigma",
            };
    }
    return {};
}

[[nodiscard]] inline json::Value family_units(Family family) {
    json::Value units = json::Value::object();
    switch (family) {
        case Family::hcw:
            units.set("position", "m");
            units.set("velocity", "m/s");
            units.set("acceleration", "m/s^2");
            units.set("angular_rate", "rad/s");
            units.set("time", "s");
            break;
        case Family::powered_descent_3dof:
            units.set("position", "m");
            units.set("velocity", "m/s");
            units.set("mass", "kg");
            units.set("thrust", "N");
            units.set("angle", "rad");
            units.set("time", "s");
            break;
        case Family::powered_descent_6dof:
            units.set("position", "m");
            units.set("velocity", "m/s");
            units.set("mass", "kg");
            units.set("thrust", "N");
            units.set("torque", "N*m");
            units.set("angle", "rad");
            units.set("angular_rate", "rad/s");
            units.set("inertia", "kg*m^2");
            units.set("time", "s");
            break;
        case Family::low_thrust:
            units.set("position", "km");
            units.set("velocity", "km/s");
            units.set("mass", "kg");
            units.set("thrust", "N");
            units.set("gravitational_parameter", "km^3/s^2");
            units.set("time", "s");
            break;
    }
    return units;
}

namespace detail {

inline VehicleParameters parse_vehicle(Family family, const json::Value& document) {
    auto vehicle = default_vehicle(family);
    const auto& vehicle_section = section(document, "vehicle");
    const auto& environment = section(document, "environment");
    const auto& constraints = section(document, "constraints");
    switch (family) {
        case Family::hcw:
            vehicle.mean_motion = positive(environment, "mean_motion", vehicle.mean_motion);
            vehicle.maximum_acceleration = positive(
                constraints, "maximum_acceleration", vehicle.maximum_acceleration
            );
            {
                const auto bound = json::string_or(constraints, "acceleration_bound", "norm");
                require(
                    bound == "norm" || bound == "box",
                    "'acceleration_bound' must be 'norm' or 'box'"
                );
                vehicle.acceleration_norm_bound = bound == "norm";
            }
            break;
        case Family::powered_descent_3dof:
        case Family::powered_descent_6dof: {
            vehicle.gravity = vector3(environment, "gravity", vehicle.gravity);
            vehicle.minimum_mass = positive(vehicle_section, "dry_mass", vehicle.minimum_mass);
            vehicle.mass_flow_coefficient =
                mass_flow_from(vehicle_section, vehicle.mass_flow_coefficient);
            const auto& thrust = section(vehicle_section, "thrust");
            vehicle.maximum_thrust = positive(thrust, "maximum", vehicle.maximum_thrust);
            vehicle.minimum_thrust = non_negative(thrust, "minimum", vehicle.minimum_thrust);
            require(
                vehicle.minimum_thrust <= vehicle.maximum_thrust,
                "'thrust.minimum' may not exceed 'thrust.maximum'"
            );
            vehicle.maximum_tilt_radians =
                positive(constraints, "maximum_tilt", vehicle.maximum_tilt_radians);
            vehicle.glide_slope_radians =
                positive(constraints, "glide_slope", vehicle.glide_slope_radians);
            constexpr double half_pi = 1.5707963267948966;
            require(
                vehicle.maximum_tilt_radians < half_pi,
                "'maximum_tilt' must lie strictly inside (0, pi/2) radians"
            );
            require(
                vehicle.glide_slope_radians < half_pi,
                "'glide_slope' must lie strictly inside (0, pi/2) radians"
            );
            vehicle.minimum_altitude = non_negative(constraints, "minimum_altitude", 0.0);
            require(
                vehicle.minimum_altitude == 0.0,
                "'minimum_altitude' other than 0 is not supported by the frozen "
                "powered-descent transcription (altitude bound is z >= 0)"
            );
            if (family == Family::powered_descent_6dof) {
                vehicle.principal_inertia =
                    vector3(vehicle_section, "principal_inertia", vehicle.principal_inertia);
                for (const double inertia : vehicle.principal_inertia) {
                    require(inertia > 0.0, "'principal_inertia' entries must be positive");
                }
                vehicle.maximum_torque =
                    positive(vehicle_section, "maximum_torque", vehicle.maximum_torque);
                vehicle.maximum_angular_rate = positive(
                    constraints, "maximum_angular_rate", vehicle.maximum_angular_rate
                );
            } else {
                require(
                    !vehicle_section.contains("principal_inertia")
                        && !vehicle_section.contains("maximum_torque")
                        && !constraints.contains("maximum_angular_rate"),
                    "inertia, torque, and angular-rate parameters apply only to "
                    "powered_descent_6dof"
                );
            }
            break;
        }
        case Family::low_thrust:
            vehicle.gravitational_parameter = positive(
                environment, "gravitational_parameter", vehicle.gravitational_parameter
            );
            vehicle.thrust_to_acceleration = positive(
                vehicle_section, "thrust_to_acceleration", vehicle.thrust_to_acceleration
            );
            vehicle.minimum_mass = positive(vehicle_section, "dry_mass", vehicle.minimum_mass);
            vehicle.mass_flow_coefficient =
                mass_flow_from(vehicle_section, vehicle.mass_flow_coefficient);
            {
                const auto& thrust = section(vehicle_section, "thrust");
                vehicle.maximum_thrust = positive(thrust, "maximum", vehicle.maximum_thrust);
                vehicle.minimum_thrust = non_negative(thrust, "minimum", 0.0);
                require(
                    vehicle.minimum_thrust == 0.0,
                    "a positive 'thrust.minimum' is not supported by the frozen low-thrust "
                    "transcription"
                );
            }
            vehicle.minimum_radius =
                positive(constraints, "minimum_radius", vehicle.minimum_radius);
            break;
    }
    return vehicle;
}

inline TranscriptionWeights parse_weights(
    Family family,
    const VehicleParameters& vehicle,
    const json::Value& document
) {
    auto weights = default_weights(family, vehicle);
    const auto& solver = section(document, "solver");
    const auto& transcription = section(solver, "transcription");
    if (family == Family::hcw) {
        weights.state_weights = positive_vector(transcription, "state_weights", weights.state_weights);
        weights.control_weights =
            positive_vector(transcription, "control_weights", weights.control_weights);
        return weights;
    }
    weights.virtual_l1_weight =
        non_negative(transcription, "virtual_l1_weight", weights.virtual_l1_weight);
    weights.virtual_quadratic_weight = non_negative(
        transcription, "virtual_quadratic_weight", weights.virtual_quadratic_weight
    );
    weights.virtual_epigraph_regularisation = non_negative(
        transcription,
        "virtual_epigraph_regularisation",
        weights.virtual_epigraph_regularisation
    );
    weights.fuel_weight = non_negative(transcription, "fuel_weight", weights.fuel_weight);
    weights.state_tracking_weights = positive_vector(
        transcription, "state_tracking_weights", weights.state_tracking_weights
    );
    weights.control_tracking_weights = positive_vector(
        transcription, "control_tracking_weights", weights.control_tracking_weights
    );
    weights.state_trust_scales =
        positive_vector(transcription, "state_trust_scales", weights.state_trust_scales);
    weights.control_trust_scales =
        positive_vector(transcription, "control_trust_scales", weights.control_trust_scales);
    return weights;
}

inline SolverOptions parse_solver(const json::Value& document, const TranscriptionWeights& weights) {
    SolverOptions options{};
    const auto& solver = section(document, "solver");
    options.backend = parse_backend(json::string_or(solver, "backend", "pure_qoco"));
    options.preset = parse_preset(
        json::string_or(
            solver,
            "preset",
            options.backend == Backend::pure_qoco || options.backend == Backend::cpu_reference
                ? "frozen_adaptive_pure_qoco"
                : options.backend == Backend::pdhcg_recovery
                    ? "fixed_tight_pdhcg"
                    : "frozen_adaptive_pdhcg"
        )
    );
    options.tolerance = positive(solver, "tolerance", options.tolerance);
    options.step_tolerance = positive(solver, "step_tolerance", options.step_tolerance);
    const double maximum_outer = json::number_or(solver, "maximum_outer_iterations", 30.0);
    require(
        std::isfinite(maximum_outer) && maximum_outer >= 1.0 && maximum_outer <= 10'000.0
            && maximum_outer == std::floor(maximum_outer),
        "'maximum_outer_iterations' must be an integer in [1, 10000]"
    );
    options.maximum_outer_iterations = static_cast<std::uint32_t>(maximum_outer);
    const double minimum_outer = json::number_or(solver, "minimum_outer_iterations", 1.0);
    require(
        std::isfinite(minimum_outer) && minimum_outer >= 1.0 && minimum_outer <= maximum_outer
            && minimum_outer == std::floor(minimum_outer),
        "'minimum_outer_iterations' must be an integer in [1, maximum_outer_iterations]"
    );
    options.minimum_outer_iterations = static_cast<std::uint32_t>(minimum_outer);
    options.time_limit_seconds = non_negative(solver, "time_limit_seconds", 0.0);
    options.certificate_tolerance =
        positive(solver, "certificate_tolerance", options.tolerance);
    options.replay_parity_tolerance =
        positive(solver, "replay_parity_tolerance", options.replay_parity_tolerance);
    options.warm_start_mode = json::string_or(
        solver,
        "warm_start_mode",
        options.backend == Backend::pure_qoco ? "primal" : "full_retained"
    );
    require(
        options.warm_start_mode == "none" || options.warm_start_mode == "primal"
            || options.warm_start_mode == "primal_dual"
            || options.warm_start_mode == "full_retained",
        "'warm_start_mode' must be none, primal, primal_dual, or full_retained"
    );

    const auto& trust = section(solver, "trust_region");
    auto& tr = options.trust;
    tr.initial_radius = positive(trust, "initial_radius", tr.initial_radius);
    tr.minimum_radius = positive(trust, "minimum_radius", tr.minimum_radius);
    tr.maximum_radius = positive(trust, "maximum_radius", tr.maximum_radius);
    tr.shrink_factor = positive(trust, "shrink_factor", tr.shrink_factor);
    tr.expansion_factor = positive(trust, "expansion_factor", tr.expansion_factor);
    tr.acceptance_threshold =
        non_negative(trust, "acceptance_threshold", tr.acceptance_threshold);
    tr.strong_agreement_threshold =
        positive(trust, "strong_agreement_threshold", tr.strong_agreement_threshold);
    tr.near_boundary_fraction =
        positive(trust, "near_boundary_fraction", tr.near_boundary_fraction);
    tr.restoration_reduction =
        positive(trust, "restoration_reduction", tr.restoration_reduction);
    require(
        tr.minimum_radius <= tr.initial_radius && tr.initial_radius <= tr.maximum_radius,
        "trust region radii must satisfy minimum <= initial <= maximum"
    );
    require(tr.shrink_factor < 1.0, "'shrink_factor' must be below 1");
    require(tr.expansion_factor > 1.0, "'expansion_factor' must exceed 1");
    require(tr.acceptance_threshold < 1.0, "'acceptance_threshold' must be below 1");
    require(tr.restoration_reduction < 1.0, "'restoration_reduction' must be below 1");

    const auto& penalty = section(solver, "penalty");
    options.penalty.feasibility_penalty =
        positive(penalty, "feasibility_penalty", options.penalty.feasibility_penalty);
    options.penalty.virtual_penalty =
        non_negative(penalty, "virtual_penalty", weights.virtual_l1_weight);

    const auto& forcing = section(solver, "forcing");
    auto& fr = options.forcing;
    fr.epsilon_max = positive(forcing, "epsilon_max", fr.epsilon_max);
    fr.epsilon_floor = positive(forcing, "epsilon_floor", fr.epsilon_floor);
    fr.epsilon_0 = positive(forcing, "epsilon_0", fr.epsilon_0);
    fr.coefficient = positive(forcing, "coefficient", fr.coefficient);
    fr.alpha = positive(forcing, "alpha", fr.alpha);
    fr.gamma = positive(forcing, "gamma", fr.gamma);
    require(fr.gamma < 1.0, "forcing 'gamma' must lie in (0, 1)");
    fr.repair_ceiling = positive(forcing, "repair_ceiling", fr.repair_ceiling);
    fr.progress_ceiling = positive(forcing, "progress_ceiling", fr.progress_ceiling);
    fr.refinement_ceiling = positive(forcing, "refinement_ceiling", fr.refinement_ceiling);
    fr.polish_ceiling = positive(forcing, "polish_ceiling", fr.polish_ceiling);
    const auto iterations = [&](std::string_view key, std::uint64_t fallback) {
        const double value = json::number_or(forcing, key, static_cast<double>(fallback));
        require(
            std::isfinite(value) && value >= 1.0 && value == std::floor(value)
                && value <= 1.0e8,
            "forcing '" + std::string(key) + "' must be a positive integer"
        );
        return static_cast<std::uint64_t>(value);
    };
    fr.repair_iterations = iterations("repair_iterations", fr.repair_iterations);
    fr.progress_iterations = iterations("progress_iterations", fr.progress_iterations);
    fr.refinement_iterations = iterations("refinement_iterations", fr.refinement_iterations);
    fr.polish_iterations = iterations("polish_iterations", fr.polish_iterations);
    fr.resolve_trigger_multiple =
        positive(forcing, "resolve_trigger_multiple", fr.resolve_trigger_multiple);
    fr.resolve_refinement_factor =
        positive(forcing, "resolve_refinement_factor", fr.resolve_refinement_factor);
    fr.resolve_minimum_tolerance =
        positive(forcing, "resolve_minimum_tolerance", fr.resolve_minimum_tolerance);
    {
        const double resolves = json::number_or(forcing, "maximum_resolves", 1.0);
        require(
            std::isfinite(resolves) && resolves >= 0.0 && resolves <= 8.0
                && resolves == std::floor(resolves),
            "forcing 'maximum_resolves' must be an integer in [0, 8]"
        );
        fr.maximum_resolves = static_cast<std::uint32_t>(resolves);
    }
    fr.final_polish_tolerance =
        positive(forcing, "final_polish_tolerance", fr.final_polish_tolerance);
    fr.final_polish_iteration_limit =
        iterations("final_polish_iteration_limit", fr.final_polish_iteration_limit);

    // Preset-driven inner-solve policy.
    switch (options.preset) {
        case Preset::frozen_adaptive_pure_qoco:
            fr.fixed_inner_tolerance = positive(forcing, "fixed_inner_tolerance", 1.0e-8);
            fr.fixed_inner_iteration_limit =
                iterations("fixed_inner_iteration_limit", 200U);
            break;
        case Preset::frozen_adaptive_pdhcg:
            fr.fixed_inner_tolerance = json::number_or(forcing, "fixed_inner_tolerance", 0.0);
            fr.fixed_inner_iteration_limit =
                static_cast<std::uint64_t>(json::number_or(forcing, "fixed_inner_iteration_limit", 0.0));
            break;
        case Preset::fixed_tight_pdhcg:
            fr.fixed_inner_tolerance = positive(
                forcing, "fixed_inner_tolerance", std::min(options.tolerance, 1.0e-6)
            );
            fr.fixed_inner_iteration_limit =
                iterations("fixed_inner_iteration_limit", 1'000'000U);
            break;
    }
    if (options.backend == Backend::pure_qoco) {
        require(
            options.preset == Preset::frozen_adaptive_pure_qoco,
            "backend pure_qoco requires preset frozen_adaptive_pure_qoco"
        );
    }
    if (options.backend == Backend::pdhcg_recovery) {
        require(
            options.preset == Preset::fixed_tight_pdhcg,
            "backend pdhcg_recovery requires preset fixed_tight_pdhcg"
        );
    }
    if (options.backend == Backend::pdhcg) {
        require(
            options.preset != Preset::frozen_adaptive_pure_qoco,
            "backend pdhcg cannot use the pure-QOCO preset; choose frozen_adaptive_pdhcg or "
            "fixed_tight_pdhcg"
        );
    }
    return options;
}

inline OutputOptions parse_output(const json::Value& document) {
    OutputOptions output{};
    const auto& section_value = section(document, "output");
    const double substeps = json::number_or(section_value, "dense_replay_substeps", 10.0);
    require(
        std::isfinite(substeps) && substeps >= 1.0 && substeps <= 1'000.0
            && substeps == std::floor(substeps),
        "'dense_replay_substeps' must be an integer in [1, 1000]"
    );
    output.dense_replay_substeps = static_cast<std::size_t>(substeps);
    output.include_iterations = json::boolean_or(section_value, "include_iterations", true);
    return output;
}

inline std::optional<WarmStart> parse_warm_start(
    const json::Value& document,
    std::size_t intervals,
    std::size_t nx,
    std::size_t nu
) {
    const auto* warm = document.find("warm_start");
    if (warm == nullptr || warm->is_null()) {
        return std::nullopt;
    }
    require(warm->is_object(), "'warm_start' must be an object or null");
    WarmStart result{};
    const auto& states = warm->at("states");
    const auto& controls = warm->at("controls");
    require(
        states.is_array() && states.size() == intervals + 1U,
        "'warm_start.states' must contain intervals + 1 state rows"
    );
    require(
        controls.is_array() && controls.size() == intervals,
        "'warm_start.controls' must contain intervals control rows"
    );
    for (const auto& row : states.as_array()) {
        require(
            row.is_array() && row.size() == nx,
            "'warm_start.states' rows must have the family state dimension"
        );
        for (const auto& item : row.as_array()) {
            require(
                item.is_number() && std::isfinite(item.as_number()),
                "'warm_start.states' entries must be finite numbers"
            );
            result.states.push_back(item.as_number());
        }
    }
    for (const auto& row : controls.as_array()) {
        require(
            row.is_array() && row.size() == nu,
            "'warm_start.controls' rows must have the family control dimension"
        );
        for (const auto& item : row.as_array()) {
            require(
                item.is_number() && std::isfinite(item.as_number()),
                "'warm_start.controls' entries must be finite numbers"
            );
            result.controls.push_back(item.as_number());
        }
    }
    return result;
}

}  // namespace detail

/// Parse and validate a normalised planner problem document.
[[nodiscard]] inline PlannerProblem parse_problem(const json::Value& document) {
    using detail::require;
    require(document.is_object(), "planner problem document must be a JSON object");
    const auto version = json::string_or(document, "schema_version", "");
    require(
        version == schema_version,
        "unsupported planner schema_version '" + version + "' (expected "
        + std::string(schema_version) + ")"
    );
    PlannerProblem problem{};
    problem.document = document;
    problem.name = json::string_or(document, "name", "");
    problem.family = parse_family(json::string_at(document, "family"));
    const std::size_t nx = problem.nx();
    const std::size_t nu = problem.nu();

    const auto& horizon = document.at("horizon");
    require(horizon.is_object(), "'horizon' must be an object");
    require(
        !json::boolean_or(horizon, "free_final_time", false),
        "free final time is not supported by planner schema 1.0.0; set "
        "'horizon.free_final_time' to false and provide 'final_time'"
    );
    const double intervals = json::number_at(horizon, "intervals");
    require(
        std::isfinite(intervals) && intervals >= 2.0 && intervals <= 200'000.0
            && intervals == std::floor(intervals),
        "'horizon.intervals' must be an integer in [2, 200000]"
    );
    problem.intervals = static_cast<std::size_t>(intervals);
    problem.final_time = json::number_at(horizon, "final_time");
    require(
        std::isfinite(problem.final_time) && problem.final_time > 0.0,
        "'horizon.final_time' must be finite and positive"
    );
    problem.step_seconds = problem.final_time / static_cast<double>(problem.intervals);

    problem.initial_state = json::numbers_at(document, "initial_state", nx);
    for (const double value : problem.initial_state) {
        require(std::isfinite(value), "'initial_state' entries must be finite");
    }
    const auto& terminal = document.at("terminal");
    require(terminal.is_object(), "'terminal' must be an object");
    problem.target_state = json::numbers_at(terminal, "state", nx);
    for (const double value : problem.target_state) {
        require(std::isfinite(value), "'terminal.state' entries must be finite");
    }
    const auto supported = supported_terminal_pattern(problem.family);
    problem.terminal_fixed = supported;
    if (const auto* fixed = terminal.find("fixed"); fixed != nullptr && !fixed->is_null()) {
        require(
            fixed->is_array() && fixed->size() == nx,
            "'terminal.fixed' must be a boolean array with the family state dimension"
        );
        for (std::size_t component = 0U; component < nx; ++component) {
            const auto& flag = fixed->as_array()[component];
            require(flag.is_boolean(), "'terminal.fixed' entries must be booleans");
            problem.terminal_fixed[component] = flag.as_boolean();
        }
        const auto names = state_names(problem.family);
        for (std::size_t component = 0U; component < nx; ++component) {
            if (problem.terminal_fixed[component] != supported[component]) {
                throw ProblemError(
                    "terminal component '" + names[component] + "' must be "
                    + (supported[component] ? "fixed" : "free")
                    + " for family " + std::string(family_name(problem.family))
                    + "; the frozen transcription pins exactly the first "
                    + std::to_string(terminal_dimension(problem.family))
                    + " components and leaves the remainder free"
                );
            }
        }
    }

    // Free terminal components carry no target: mirror the initial state so the
    // transcription still receives a physically valid placeholder state.
    for (std::size_t component = 0U; component < nx; ++component) {
        if (!problem.terminal_fixed[component]) {
            problem.target_state[component] = problem.initial_state[component];
        }
    }

    problem.vehicle = detail::parse_vehicle(problem.family, document);
    problem.weights = detail::parse_weights(problem.family, problem.vehicle, document);
    problem.solver = detail::parse_solver(document, problem.weights);
    problem.output = detail::parse_output(document);
    problem.warm_start = detail::parse_warm_start(document, problem.intervals, nx, nu);

    // Family-specific physical sanity checks that would otherwise surface as
    // opaque transcription exceptions.
    switch (problem.family) {
        case Family::hcw:
            break;
        case Family::powered_descent_3dof:
        case Family::powered_descent_6dof: {
            const std::size_t mass_index = nx - 1U;
            require(
                problem.initial_state[mass_index] > problem.vehicle.minimum_mass,
                "initial mass must exceed the vehicle dry mass"
            );
            require(problem.initial_state[2U] >= 0.0, "initial altitude (z) must be non-negative");
            require(problem.target_state[2U] >= 0.0, "target altitude (z) must be non-negative");
            if (problem.family == Family::powered_descent_6dof) {
                double norm_squared = 0.0;
                double target_norm_squared = 0.0;
                for (std::size_t component = 6U; component < 10U; ++component) {
                    norm_squared += problem.initial_state[component] * problem.initial_state[component];
                    target_norm_squared +=
                        problem.target_state[component] * problem.target_state[component];
                }
                require(
                    std::abs(std::sqrt(norm_squared) - 1.0) <= 1.0e-9,
                    "initial quaternion [q0, q1, q2, q3] must have unit norm"
                );
                require(
                    std::abs(std::sqrt(target_norm_squared) - 1.0) <= 1.0e-9,
                    "target quaternion [q0, q1, q2, q3] must have unit norm"
                );
            }
            break;
        }
        case Family::low_thrust: {
            require(
                problem.initial_state[6U] > problem.vehicle.minimum_mass,
                "initial mass must exceed the vehicle dry mass"
            );
            const auto radius = [](const std::vector<double>& state) {
                return std::sqrt(state[0U] * state[0U] + state[1U] * state[1U] + state[2U] * state[2U]);
            };
            require(
                radius(problem.initial_state) >= problem.vehicle.minimum_radius,
                "initial radius must be at least the minimum radius constraint"
            );
            require(
                radius(problem.target_state) >= problem.vehicle.minimum_radius,
                "target radius must be at least the minimum radius constraint"
            );
            break;
        }
    }
    return problem;
}

[[nodiscard]] inline PlannerProblem parse_problem_text(std::string_view text) {
    return parse_problem(json::parse(text));
}

/// Render the fully defaulted document for one family (used by the Python layer to
/// prove its default table matches the native defaults).
[[nodiscard]] inline json::Value default_document(Family family) {
    const auto vehicle = default_vehicle(family);
    const auto weights = default_weights(family, vehicle);
    json::Value vehicle_json = json::Value::object();
    json::Value environment = json::Value::object();
    json::Value constraints = json::Value::object();
    switch (family) {
        case Family::hcw:
            environment.set("mean_motion", vehicle.mean_motion);
            constraints.set("maximum_acceleration", vehicle.maximum_acceleration);
            constraints.set("acceleration_bound", "norm");
            break;
        case Family::powered_descent_3dof:
        case Family::powered_descent_6dof:
            environment.set("gravity", json::Value::numbers(vehicle.gravity));
            vehicle_json.set("dry_mass", vehicle.minimum_mass);
            vehicle_json.set("mass_flow_coefficient", vehicle.mass_flow_coefficient);
            vehicle_json.set(
                "thrust",
                json::Value(json::Object{
                    {"minimum", json::Value(vehicle.minimum_thrust)},
                    {"maximum", json::Value(vehicle.maximum_thrust)},
                })
            );
            constraints.set("maximum_tilt", vehicle.maximum_tilt_radians);
            constraints.set("glide_slope", vehicle.glide_slope_radians);
            constraints.set("minimum_altitude", 0.0);
            if (family == Family::powered_descent_6dof) {
                vehicle_json.set(
                    "principal_inertia", json::Value::numbers(vehicle.principal_inertia)
                );
                vehicle_json.set("maximum_torque", vehicle.maximum_torque);
                constraints.set("maximum_angular_rate", vehicle.maximum_angular_rate);
            }
            break;
        case Family::low_thrust:
            environment.set("gravitational_parameter", vehicle.gravitational_parameter);
            vehicle_json.set("dry_mass", vehicle.minimum_mass);
            vehicle_json.set("mass_flow_coefficient", vehicle.mass_flow_coefficient);
            vehicle_json.set("thrust_to_acceleration", vehicle.thrust_to_acceleration);
            vehicle_json.set(
                "thrust",
                json::Value(json::Object{
                    {"minimum", json::Value(0.0)},
                    {"maximum", json::Value(vehicle.maximum_thrust)},
                })
            );
            constraints.set("minimum_radius", vehicle.minimum_radius);
            break;
    }
    json::Value transcription = json::Value::object();
    if (family == Family::hcw) {
        transcription.set("state_weights", json::Value::numbers(weights.state_weights));
        transcription.set("control_weights", json::Value::numbers(weights.control_weights));
    } else {
        transcription.set("virtual_l1_weight", weights.virtual_l1_weight);
        transcription.set("virtual_quadratic_weight", weights.virtual_quadratic_weight);
        transcription.set(
            "virtual_epigraph_regularisation", weights.virtual_epigraph_regularisation
        );
        transcription.set("fuel_weight", weights.fuel_weight);
        transcription.set(
            "state_tracking_weights", json::Value::numbers(weights.state_tracking_weights)
        );
        transcription.set(
            "control_tracking_weights", json::Value::numbers(weights.control_tracking_weights)
        );
        transcription.set("state_trust_scales", json::Value::numbers(weights.state_trust_scales));
        transcription.set(
            "control_trust_scales", json::Value::numbers(weights.control_trust_scales)
        );
    }
    json::Value fixed = json::Value::array();
    for (const bool flag : supported_terminal_pattern(family)) {
        fixed.push_back(json::Value(flag));
    }
    json::Value result = json::Value::object();
    result.set("schema_version", std::string(schema_version));
    result.set("family", std::string(family_name(family)));
    result.set("units", family_units(family));
    result.set("state_order", json::Value([&] {
        json::Array names;
        for (const auto& name : state_names(family)) {
            names.emplace_back(name);
        }
        return names;
    }()));
    result.set("control_order", json::Value([&] {
        json::Array names;
        for (const auto& name : control_names(family)) {
            names.emplace_back(name);
        }
        return names;
    }()));
    result.set("terminal_fixed", fixed);
    result.set("vehicle", vehicle_json);
    result.set("environment", environment);
    result.set("constraints", constraints);
    result.set("transcription", transcription);
    return result;
}

}  // namespace spacepdhcg::planner
