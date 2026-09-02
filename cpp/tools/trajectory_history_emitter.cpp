#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/orbitweaver/lambert.hpp"
#include "spacepdhcg/scvx/low_thrust_benchmark.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

template <std::size_t Size>
void emit_array(const std::array<double, Size>& values) {
    std::cout << '[';
    for (std::size_t index = 0U; index < Size; ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        std::cout << values[index];
    }
    std::cout << ']';
}

template <std::size_t Size>
void emit_arrays(const std::vector<std::array<double, Size>>& values) {
    std::cout << '[';
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        emit_array(values[index]);
    }
    std::cout << ']';
}

void emit_times(const std::size_t count, const double step_seconds) {
    std::cout << '[';
    for (std::size_t index = 0U; index < count; ++index) {
        if (index != 0U) {
            std::cout << ',';
        }
        std::cout << static_cast<double>(index) * step_seconds;
    }
    std::cout << ']';
}

template <std::size_t Size>
double infinity_distance(
    const std::array<double, Size>& left,
    const std::array<double, Size>& right
) {
    double result = 0.0;
    for (std::size_t index = 0U; index < Size; ++index) {
        result = std::max(result, std::abs(left[index] - right[index]));
    }
    return result;
}

using CartesianState = std::array<double, 6U>;

CartesianState two_body_derivative(
    const CartesianState& state,
    const double gravitational_parameter
) {
    const auto radius =
        std::sqrt(state[0U] * state[0U] + state[1U] * state[1U] + state[2U] * state[2U]);
    const auto scale = -gravitational_parameter / (radius * radius * radius);
    return {
        state[3U],
        state[4U],
        state[5U],
        scale * state[0U],
        scale * state[1U],
        scale * state[2U],
    };
}

CartesianState add_scaled(
    const CartesianState& state,
    const CartesianState& derivative,
    const double scale
) {
    CartesianState result{};
    for (std::size_t index = 0U; index < result.size(); ++index) {
        result[index] = state[index] + scale * derivative[index];
    }
    return result;
}

CartesianState two_body_rk4(
    const CartesianState& state,
    const double step_seconds,
    const double gravitational_parameter
) {
    const auto k1 = two_body_derivative(state, gravitational_parameter);
    const auto k2 = two_body_derivative(
        add_scaled(state, k1, 0.5 * step_seconds),
        gravitational_parameter
    );
    const auto k3 = two_body_derivative(
        add_scaled(state, k2, 0.5 * step_seconds),
        gravitational_parameter
    );
    const auto k4 =
        two_body_derivative(add_scaled(state, k3, step_seconds), gravitational_parameter);
    CartesianState result{};
    for (std::size_t index = 0U; index < result.size(); ++index) {
        result[index] =
            state[index]
            + step_seconds
                  * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
                  / 6.0;
    }
    return result;
}

void emit_lambert(const std::size_t replay_steps) {
    using namespace spacepdhcg::orbitweaver;
    if (replay_steps < 2U) {
        throw std::invalid_argument("Lambert replay requires at least two steps");
    }
    constexpr double time_of_flight = 3'600.0;
    constexpr double gravitational_parameter = 3.986004418e14;
    const Vector3 departure{7.0e6, 0.0, 0.0};
    const Vector3 arrival{0.0, 8.0e6, 0.0};
    const auto solution = solve_lambert_zero_revolution(
        departure,
        arrival,
        time_of_flight,
        gravitational_parameter,
        false,
        1.0e-8,
        256U
    );
    CartesianState initial{
        departure[0U],
        departure[1U],
        departure[2U],
        solution.departure_velocity[0U],
        solution.departure_velocity[1U],
        solution.departure_velocity[2U],
    };
    const auto step_seconds = time_of_flight / static_cast<double>(replay_steps);
    std::vector<CartesianState> states(replay_steps + 1U);
    states.front() = initial;
    for (std::size_t step = 0U; step < replay_steps; ++step) {
        states[step + 1U] =
            two_body_rk4(states[step], step_seconds, gravitational_parameter);
    }
    double terminal_position_inf = 0.0;
    double terminal_velocity_inf = 0.0;
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
        terminal_position_inf =
            std::max(terminal_position_inf, std::abs(states.back()[axis] - arrival[axis]));
        terminal_velocity_inf = std::max(
            terminal_velocity_inf,
            std::abs(states.back()[3U + axis] - solution.arrival_velocity[axis])
        );
    }

    std::cout << "{\"family\":\"P2-Lambert\","
              << "\"frame\":\"Earth-centred inertial Cartesian\","
              << "\"position_units\":\"m\",\"time_units\":\"s\","
              << "\"integration\":{\"source\":\"zero-revolution universal-variable Lambert\","
              << "\"replay\":\"independent two-body RK4\",\"replay_steps\":"
              << replay_steps << "},\"gravitational_parameter\":"
              << gravitational_parameter << ",\"time_of_flight\":" << time_of_flight
              << ",\"departure_position\":";
    emit_array(departure);
    std::cout << ",\"arrival_position\":";
    emit_array(arrival);
    std::cout << ",\"departure_velocity\":";
    emit_array(solution.departure_velocity);
    std::cout << ",\"arrival_velocity\":";
    emit_array(solution.arrival_velocity);
    std::cout << ",\"universal_parameter\":" << solution.universal_parameter
              << ",\"transfer_angle_radians\":" << solution.transfer_angle_radians
              << ",\"iterations\":" << solution.iterations
              << ",\"time_of_flight_residual\":" << solution.time_of_flight_residual
              << ",\"node_times\":[0," << time_of_flight << "],\"states\":[";
    emit_array(initial);
    std::cout << ',';
    CartesianState terminal{
        arrival[0U],
        arrival[1U],
        arrival[2U],
        solution.arrival_velocity[0U],
        solution.arrival_velocity[1U],
        solution.arrival_velocity[2U],
    };
    emit_array(terminal);
    std::cout << "],\"replay_times\":";
    emit_times(states.size(), step_seconds);
    std::cout << ",\"replay_states\":";
    emit_arrays(states);
    std::cout << ",\"validation\":{\"terminal_position_inf\":"
              << terminal_position_inf << ",\"terminal_velocity_inf\":"
              << terminal_velocity_inf << "}}\n";
}

void emit_pd6(
    const std::size_t intervals,
    const double attitude,
    const double rate,
    const std::size_t replay_substeps
) {
    using namespace spacepdhcg::dynamics;
    if (intervals < 2U || replay_substeps < 2U) {
        throw std::invalid_argument("P1-D requires at least two intervals and replay substeps");
    }
    const PoweredDescent6DofModel model{};
    const auto step_seconds = 10.0 / static_cast<double>(intervals);
    PoweredDescent6DofState initial{
        0.0,
        0.0,
        100.0,
        0.0,
        0.0,
        -1.0,
        std::cos(0.5 * attitude),
        std::sin(0.5 * attitude),
        0.0,
        0.0,
        rate,
        0.0,
        0.0,
        2'000.0,
    };
    const PoweredDescent6DofControl control{
        0.0,
        0.0,
        7'500.0,
        0.0,
        0.0,
        0.0,
        7'500.0,
    };
    const std::vector<PoweredDescent6DofControl> controls(intervals, control);
    std::vector<PoweredDescent6DofState> states(intervals + 1U);
    states.front() = initial;
    for (std::size_t interval = 0U; interval < intervals; ++interval) {
        states[interval + 1U] =
            model.euler_step(states[interval], controls[interval], step_seconds);
    }

    const auto replay_step = step_seconds / static_cast<double>(replay_substeps);
    std::vector<PoweredDescent6DofState> replay_states(
        intervals * replay_substeps + 1U
    );
    std::vector<PoweredDescent6DofControl> replay_controls(
        intervals * replay_substeps,
        control
    );
    replay_states.front() = initial;
    for (std::size_t interval = 0U; interval < replay_controls.size(); ++interval) {
        replay_states[interval + 1U] =
            model.rk4_step(replay_states[interval], replay_controls[interval], replay_step);
    }
    const auto source_path = model.path_diagnostics(states, controls);
    const auto replay_path = model.path_diagnostics(replay_states, replay_controls);

    std::cout << "{\"family\":\"P1-D-pd6\",\"frame\":\"local-level inertial\","
              << "\"position_units\":\"m\",\"time_units\":\"s\","
              << "\"integration\":{\"transcription\":\"explicit Euler\","
              << "\"replay\":\"RK4 ZOH\",\"replay_substeps\":" << replay_substeps << "},"
              << "\"model\":{\"gravity\":";
    emit_array(model.config().gravity);
    std::cout << ",\"minimum_mass\":" << model.config().minimum_mass
              << ",\"maximum_thrust\":" << model.config().maximum_thrust
              << ",\"maximum_torque\":" << model.config().maximum_torque
              << ",\"maximum_angular_rate\":" << model.config().maximum_angular_rate
              << ",\"maximum_tilt_radians\":" << model.config().maximum_tilt_radians
              << ",\"glide_slope_radians\":" << model.config().glide_slope_radians
              << ",\"mass_flow_coefficient\":" << model.config().mass_flow_coefficient
              << "},\"node_times\":";
    emit_times(states.size(), step_seconds);
    std::cout << ",\"states\":";
    emit_arrays(states);
    std::cout << ",\"controls\":";
    emit_arrays(controls);
    std::cout << ",\"replay_times\":";
    emit_times(replay_states.size(), replay_step);
    std::cout << ",\"replay_states\":";
    emit_arrays(replay_states);
    std::cout << ",\"validation\":{\"transcription_path_violation\":"
              << source_path.maximum_violation()
              << ",\"replay_path_violation\":" << replay_path.maximum_violation()
              << ",\"replay_terminal_inf\":"
              << infinity_distance(replay_states.back(), states.back()) << "}}\n";
}

void emit_low_thrust(
    const std::size_t intervals,
    const std::string_view transfer_name,
    const std::size_t replay_substeps
) {
    using namespace spacepdhcg::dynamics;
    using namespace spacepdhcg::scvx;
    if (intervals < 2U || replay_substeps < 2U) {
        throw std::invalid_argument("P1-E requires at least two intervals and replay substeps");
    }
    const LowThrustTwoBodyModel model{};
    const LowThrustState initial{
        7'000.0,
        0.0,
        0.0,
        0.0,
        std::sqrt(model.config().gravitational_parameter / 7'000.0),
        0.0,
        500.0,
    };
    constexpr double total_time = 10'000.0;
    const auto step_seconds = total_time / static_cast<double>(intervals);
    const auto transfer = low_thrust_transfer_class(transfer_name);
    const auto reference =
        make_low_thrust_transfer_target(model, initial, intervals, step_seconds, transfer);
    const auto& states = reference.first;
    const auto& controls = reference.second;

    const auto replay_step = step_seconds / static_cast<double>(replay_substeps);
    std::vector<LowThrustState> replay_states(intervals * replay_substeps + 1U);
    std::vector<LowThrustControl> replay_controls;
    replay_controls.reserve(intervals * replay_substeps);
    for (const auto& control : controls) {
        for (std::size_t substep = 0U; substep < replay_substeps; ++substep) {
            replay_controls.push_back(control);
        }
    }
    replay_states.front() = initial;
    for (std::size_t interval = 0U; interval < replay_controls.size(); ++interval) {
        replay_states[interval + 1U] =
            model.rk4_step(replay_states[interval], replay_controls[interval], replay_step);
    }
    const auto source_path = model.path_diagnostics(states, controls);
    const auto replay_path = model.path_diagnostics(replay_states, replay_controls);

    std::cout << "{\"family\":\"P1-E-low-thrust\","
              << "\"frame\":\"central-body inertial Cartesian\","
              << "\"position_units\":\"km\",\"time_units\":\"s\","
              << "\"integration\":{\"transcription\":\"RK4 ZOH\","
              << "\"replay\":\"RK4 ZOH refined step\",\"replay_substeps\":"
              << replay_substeps << "},\"transfer_class\":\"" << transfer_name
              << "\",\"model\":{\"gravitational_parameter\":"
              << model.config().gravitational_parameter
              << ",\"minimum_mass\":" << model.config().minimum_mass
              << ",\"maximum_thrust\":" << model.config().maximum_thrust
              << ",\"minimum_radius\":" << model.config().minimum_radius
              << ",\"mass_flow_coefficient\":" << model.config().mass_flow_coefficient
              << ",\"thrust_to_acceleration\":"
              << model.config().thrust_to_acceleration << "},\"node_times\":";
    emit_times(states.size(), step_seconds);
    std::cout << ",\"states\":";
    emit_arrays(states);
    std::cout << ",\"controls\":";
    emit_arrays(controls);
    std::cout << ",\"replay_times\":";
    emit_times(replay_states.size(), replay_step);
    std::cout << ",\"replay_states\":";
    emit_arrays(replay_states);
    std::cout << ",\"validation\":{\"transcription_path_violation\":"
              << source_path.maximum_violation()
              << ",\"replay_path_violation\":" << replay_path.maximum_violation()
              << ",\"replay_terminal_inf\":"
              << infinity_distance(replay_states.back(), states.back()) << "}}\n";
}

std::size_t parse_size(const char* value) {
    return static_cast<std::size_t>(std::stoull(value));
}

double parse_double(const char* value) {
    return std::stod(value);
}

}  // namespace

int main(const int argc, const char* const* argv) {
    try {
        std::cout << std::setprecision(17);
        if (argc == 6 && std::string_view{argv[1]} == "pd6") {
            emit_pd6(
                parse_size(argv[2]),
                parse_double(argv[3]),
                parse_double(argv[4]),
                parse_size(argv[5])
            );
            return 0;
        }
        if (argc == 5 && std::string_view{argv[1]} == "low-thrust") {
            emit_low_thrust(parse_size(argv[2]), argv[3], parse_size(argv[4]));
            return 0;
        }
        if (argc == 3 && std::string_view{argv[1]} == "lambert") {
            emit_lambert(parse_size(argv[2]));
            return 0;
        }
        throw std::invalid_argument("unsupported family or argument count");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
