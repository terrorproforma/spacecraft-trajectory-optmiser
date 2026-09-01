#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"
#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"
#include "spacepdhcg/scvx/low_thrust_benchmark.hpp"
#include "spacepdhcg/transcription/low_thrust.hpp"
#include "spacepdhcg/transcription/powered_descent_6dof.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double objective(
    const spacepdhcg::core::FixedStructure& structure,
    const spacepdhcg::core::NumericValues& values,
    const std::vector<double>& decision
) {
    double result = 0.0;
    for (std::size_t column = 0U; column < decision.size(); ++column) {
        result += values.linear_objective[column] * decision[column];
        for (auto slot = structure.quadratic.offsets[column];
             slot < structure.quadratic.offsets[column + 1U];
             ++slot) {
            result += 0.5 * decision[column]
                      * values.quadratic[static_cast<std::size_t>(slot)]
                      * decision[static_cast<std::size_t>(
                          structure.quadratic.indices[static_cast<std::size_t>(slot)]
                      )];
        }
    }
    return result;
}

double nonlinear_defect(
    const spacepdhcg::dynamics::LowThrustTwoBodyModel& model,
    const std::vector<spacepdhcg::dynamics::LowThrustState>& states,
    const std::vector<spacepdhcg::dynamics::LowThrustControl>& controls,
    const double step
) {
    double maximum = 0.0;
    for (std::size_t interval = 0U; interval < controls.size(); ++interval) {
        const auto replay = model.rk4_step(states[interval], controls[interval], step);
        for (std::size_t component = 0U; component < replay.size(); ++component) {
            maximum = std::max(
                maximum,
                std::abs(replay[component] - states[interval + 1U][component])
            );
        }
    }
    return maximum;
}

void emit_header(
    const std::string_view family,
    const std::size_t intervals,
    const std::size_t variables,
    const std::size_t scalar_rows,
    const std::size_t affine_rows,
    const std::size_t q_nonzeros,
    const std::size_t a_nonzeros,
    const std::size_t f_nonzeros
) {
    std::cout << std::setprecision(17)
              << "{\"schema_version\":\"1.0.0\",\"family\":\"" << family
              << "\",\"status\":\"reference_feasible\",\"intervals\":" << intervals
              << ",\"variables\":" << variables << ",\"scalar_rows\":" << scalar_rows
              << ",\"affine_rows\":" << affine_rows << ",\"q_nonzeros\":" << q_nonzeros
              << ",\"a_nonzeros\":" << a_nonzeros << ",\"f_nonzeros\":" << f_nonzeros;
}

void emit_pd6(
    const std::size_t intervals,
    const double attitude,
    const double rate,
    const bool final_polish
) {
    using namespace spacepdhcg::dynamics;
    using namespace spacepdhcg::transcription;
    const auto started = Clock::now();
    const PoweredDescent6DofModel model{};
    const PoweredDescent6DofScvxConfig config{
        .intervals = intervals,
        .step_seconds = 0.5,
        .trust_radius = 1.0,
    };
    const PoweredDescent6DofSubproblem subproblem{model, config};
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
            model.euler_step(states[interval], controls[interval], config.step_seconds);
    }
    const auto values = subproblem.values(states, controls, initial, states.back());
    const auto decision = subproblem.reference_decision(states, controls);
    const auto diagnostics = subproblem.diagnostics(decision, values);
    const auto elapsed = std::chrono::duration<double>(Clock::now() - started).count();
    const auto& structure = subproblem.structure();
    emit_header(
        "P1-D-pd6",
        intervals,
        subproblem.layout().variables(),
        subproblem.layout().scalar_rows(),
        subproblem.layout().affine_rows(),
        structure.quadratic.nonzeros(),
        structure.scalar_constraint.nonzeros(),
        structure.affine_cone->nonzeros()
    );
    std::cout << ",\"objective\":" << objective(structure, values, decision)
              << ",\"canonical_primal_residual\":" << diagnostics.maximum_violation()
              << ",\"canonical_dual_residual\":null"
              << ",\"canonical_cone_residual\":" << diagnostics.cone_violation_inf
              << ",\"dynamics_residual\":"
              << diagnostics.linearised_dynamics_defect_inf
              << ",\"path_residual\":"
              << std::max(
                     diagnostics.variable_violation_inf,
                     diagnostics.cone_violation_inf
                 )
              << ",\"terminal_residual\":" << diagnostics.terminal_error_inf
              << ",\"virtual_control_residual\":" << diagnostics.virtual_control_inf
              << ",\"quaternion_residual\":"
              << diagnostics.quaternion_linearisation_error_inf
              << ",\"continuous_time_violation\":null"
              << ",\"attitude_radians\":" << attitude
              << ",\"angular_rate\":" << rate
              << ",\"final_polish\":" << (final_polish ? "true" : "false")
              << ",\"elapsed_seconds\":" << elapsed << "}\n";
}

void emit_low_thrust(
    const std::size_t intervals,
    const double trust_radius,
    const std::string_view transfer_name
) {
    using namespace spacepdhcg::dynamics;
    using namespace spacepdhcg::scvx;
    using namespace spacepdhcg::transcription;
    const auto started = Clock::now();
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
    const auto transfer = low_thrust_transfer_class(transfer_name);
    constexpr double total_time = 10'000.0;
    const auto step = total_time / static_cast<double>(intervals);
    const auto reference =
        make_low_thrust_transfer_target(model, initial, intervals, step, transfer);
    LowThrustScvxConfig config{};
    config.intervals = intervals;
    config.step_seconds = step;
    config.trust_radius = trust_radius;
    config.discretisation = DiscretisationMethod::rk4_variational;
    const LowThrustSubproblem subproblem{model, config};
    const auto& states = reference.first;
    const auto& controls = reference.second;
    const auto values = subproblem.values(states, controls, initial, states.back());
    const auto decision = subproblem.reference_decision(states, controls);
    const auto diagnostics = subproblem.diagnostics(decision, values);
    const auto path = model.path_diagnostics(states, controls).maximum_violation();
    const auto elapsed = std::chrono::duration<double>(Clock::now() - started).count();
    const auto& structure = subproblem.structure();
    emit_header(
        "P1-E-low-thrust",
        intervals,
        subproblem.layout().variables(),
        subproblem.layout().scalar_rows(),
        subproblem.layout().affine_rows(),
        structure.quadratic.nonzeros(),
        structure.scalar_constraint.nonzeros(),
        structure.affine_cone->nonzeros()
    );
    std::cout << ",\"objective\":" << objective(structure, values, decision)
              << ",\"canonical_primal_residual\":" << diagnostics.maximum_violation()
              << ",\"canonical_dual_residual\":null"
              << ",\"canonical_cone_residual\":" << diagnostics.cone_violation_inf
              << ",\"dynamics_residual\":"
              << nonlinear_defect(model, states, controls, step)
              << ",\"path_residual\":" << path
              << ",\"terminal_residual\":" << diagnostics.terminal_error_inf
              << ",\"virtual_control_residual\":" << diagnostics.virtual_control_inf
              << ",\"radial_residual\":"
              << diagnostics.radial_linearisation_error_inf
              << ",\"continuous_time_violation\":" << path
              << ",\"trust_radius\":" << trust_radius
              << ",\"transfer_class\":\"" << transfer_name << "\""
              << ",\"elapsed_seconds\":" << elapsed << "}\n";
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
        if (argc < 2) {
            throw std::invalid_argument("family argument is required");
        }
        const std::string_view family{argv[1]};
        if (family == "pd6" && argc == 6) {
            emit_pd6(
                parse_size(argv[2]),
                parse_double(argv[3]),
                parse_double(argv[4]),
                std::string_view{argv[5]} == "true"
            );
            return 0;
        }
        if (family == "low-thrust" && argc == 5) {
            emit_low_thrust(
                parse_size(argv[2]),
                parse_double(argv[3]),
                argv[4]
            );
            return 0;
        }
        throw std::invalid_argument("unsupported family or argument count");
    } catch (const std::exception& error) {
        std::cerr << "{\"status\":\"failed\",\"diagnostic\":\"" << error.what() << "\"}\n";
        return 2;
    }
}
