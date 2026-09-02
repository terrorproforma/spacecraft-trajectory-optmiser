#pragma once

// JSON descriptions of a parsed planner problem (echoed into results and served
// to the Python layer through the C ABI so both sides share one default table).

#include "spacepdhcg/planner/families.hpp"
#include "spacepdhcg/planner/json.hpp"
#include "spacepdhcg/planner/problem.hpp"

#include <string>
#include <vector>

namespace spacepdhcg::planner {

[[nodiscard]] inline json::Value describe_vehicle(const PlannerProblem& problem) {
    const auto& vehicle = problem.vehicle;
    json::Value result = json::Value::object();
    switch (problem.family) {
        case Family::hcw:
            result.set("mean_motion", vehicle.mean_motion);
            result.set("maximum_acceleration", vehicle.maximum_acceleration);
            result.set("acceleration_bound", vehicle.acceleration_norm_bound ? "norm" : "box");
            break;
        case Family::powered_descent_3dof:
        case Family::powered_descent_6dof:
            result.set("gravity", json::Value::numbers(vehicle.gravity));
            result.set("dry_mass", vehicle.minimum_mass);
            result.set("mass_flow_coefficient", vehicle.mass_flow_coefficient);
            result.set("exhaust_velocity", 1.0 / vehicle.mass_flow_coefficient);
            result.set("minimum_thrust", vehicle.minimum_thrust);
            result.set("maximum_thrust", vehicle.maximum_thrust);
            result.set("maximum_tilt", vehicle.maximum_tilt_radians);
            result.set("glide_slope", vehicle.glide_slope_radians);
            result.set("minimum_altitude", vehicle.minimum_altitude);
            if (problem.family == Family::powered_descent_6dof) {
                result.set("principal_inertia", json::Value::numbers(vehicle.principal_inertia));
                result.set("maximum_torque", vehicle.maximum_torque);
                result.set("maximum_angular_rate", vehicle.maximum_angular_rate);
            }
            break;
        case Family::low_thrust:
            result.set("gravitational_parameter", vehicle.gravitational_parameter);
            result.set("thrust_to_acceleration", vehicle.thrust_to_acceleration);
            result.set("dry_mass", vehicle.minimum_mass);
            result.set("mass_flow_coefficient", vehicle.mass_flow_coefficient);
            result.set("exhaust_velocity", 1.0 / vehicle.mass_flow_coefficient);
            result.set("minimum_thrust", vehicle.minimum_thrust);
            result.set("maximum_thrust", vehicle.maximum_thrust);
            result.set("minimum_radius", vehicle.minimum_radius);
            break;
    }
    return result;
}

[[nodiscard]] inline json::Value describe_weights(const PlannerProblem& problem) {
    const auto& weights = problem.weights;
    json::Value result = json::Value::object();
    if (problem.family == Family::hcw) {
        result.set("state_weights", json::Value::numbers(weights.state_weights));
        result.set("control_weights", json::Value::numbers(weights.control_weights));
        return result;
    }
    result.set("virtual_l1_weight", weights.virtual_l1_weight);
    result.set("virtual_quadratic_weight", weights.virtual_quadratic_weight);
    result.set("virtual_epigraph_regularisation", weights.virtual_epigraph_regularisation);
    result.set("fuel_weight", weights.fuel_weight);
    result.set("state_tracking_weights", json::Value::numbers(weights.state_tracking_weights));
    result.set("control_tracking_weights", json::Value::numbers(weights.control_tracking_weights));
    result.set("state_trust_scales", json::Value::numbers(weights.state_trust_scales));
    result.set("control_trust_scales", json::Value::numbers(weights.control_trust_scales));
    return result;
}

[[nodiscard]] inline json::Value describe_solver(const PlannerProblem& problem) {
    const auto& solver = problem.solver;
    json::Value trust = json::Value::object();
    trust.set("initial_radius", solver.trust.initial_radius);
    trust.set("minimum_radius", solver.trust.minimum_radius);
    trust.set("maximum_radius", solver.trust.maximum_radius);
    trust.set("shrink_factor", solver.trust.shrink_factor);
    trust.set("expansion_factor", solver.trust.expansion_factor);
    trust.set("acceptance_threshold", solver.trust.acceptance_threshold);
    trust.set("strong_agreement_threshold", solver.trust.strong_agreement_threshold);
    trust.set("near_boundary_fraction", solver.trust.near_boundary_fraction);
    trust.set("restoration_reduction", solver.trust.restoration_reduction);

    json::Value penalty = json::Value::object();
    penalty.set("feasibility_penalty", solver.penalty.feasibility_penalty);
    penalty.set("virtual_penalty", solver.penalty.virtual_penalty);

    const auto& fr = solver.forcing;
    json::Value forcing = json::Value::object();
    forcing.set("epsilon_max", fr.epsilon_max);
    forcing.set("epsilon_floor", fr.epsilon_floor);
    forcing.set("epsilon_0", fr.epsilon_0);
    forcing.set("coefficient", fr.coefficient);
    forcing.set("alpha", fr.alpha);
    forcing.set("gamma", fr.gamma);
    forcing.set("repair_ceiling", fr.repair_ceiling);
    forcing.set("progress_ceiling", fr.progress_ceiling);
    forcing.set("refinement_ceiling", fr.refinement_ceiling);
    forcing.set("polish_ceiling", fr.polish_ceiling);
    forcing.set("repair_iterations", static_cast<double>(fr.repair_iterations));
    forcing.set("progress_iterations", static_cast<double>(fr.progress_iterations));
    forcing.set("refinement_iterations", static_cast<double>(fr.refinement_iterations));
    forcing.set("polish_iterations", static_cast<double>(fr.polish_iterations));
    forcing.set("resolve_trigger_multiple", fr.resolve_trigger_multiple);
    forcing.set("resolve_refinement_factor", fr.resolve_refinement_factor);
    forcing.set("resolve_minimum_tolerance", fr.resolve_minimum_tolerance);
    forcing.set("maximum_resolves", static_cast<double>(fr.maximum_resolves));
    forcing.set("fixed_inner_tolerance", fr.fixed_inner_tolerance);
    forcing.set("fixed_inner_iteration_limit", static_cast<double>(fr.fixed_inner_iteration_limit));
    forcing.set("final_polish_tolerance", fr.final_polish_tolerance);
    forcing.set(
        "final_polish_iteration_limit", static_cast<double>(fr.final_polish_iteration_limit)
    );

    json::Value result = json::Value::object();
    result.set("backend", std::string(backend_name(solver.backend)));
    result.set("preset", std::string(preset_name(solver.preset)));
    result.set("tolerance", solver.tolerance);
    result.set("step_tolerance", solver.step_tolerance);
    result.set("maximum_outer_iterations", static_cast<double>(solver.maximum_outer_iterations));
    result.set("minimum_outer_iterations", static_cast<double>(solver.minimum_outer_iterations));
    result.set("time_limit_seconds", solver.time_limit_seconds);
    result.set("certificate_tolerance", solver.certificate_tolerance);
    result.set("replay_parity_tolerance", solver.replay_parity_tolerance);
    result.set("warm_start_mode", solver.warm_start_mode);
    result.set("trust_region", trust);
    result.set("penalty", penalty);
    result.set("forcing", forcing);
    return result;
}

[[nodiscard]] inline json::Value describe_problem(const PlannerProblem& problem) {
    json::Value result = json::Value::object();
    result.set("schema_version", std::string(schema_version));
    result.set("name", problem.name);
    result.set("family", std::string(family_name(problem.family)));
    result.set("units", family_units(problem.family));
    result.set("state_order", json::Value([&] {
        json::Array names;
        for (const auto& name : state_names(problem.family)) {
            names.emplace_back(name);
        }
        return names;
    }()));
    result.set("control_order", json::Value([&] {
        json::Array names;
        for (const auto& name : control_names(problem.family)) {
            names.emplace_back(name);
        }
        return names;
    }()));
    json::Value horizon = json::Value::object();
    horizon.set("intervals", static_cast<double>(problem.intervals));
    horizon.set("final_time", problem.final_time);
    horizon.set("step_seconds", problem.step_seconds);
    horizon.set("free_final_time", false);
    result.set("horizon", horizon);
    result.set("initial_state", json::Value::numbers(problem.initial_state));
    json::Value terminal = json::Value::object();
    terminal.set("state", json::Value::numbers(problem.target_state));
    json::Value fixed = json::Value::array();
    for (const bool flag : problem.terminal_fixed) {
        fixed.push_back(json::Value(flag));
    }
    terminal.set("fixed", fixed);
    result.set("terminal", terminal);
    result.set("vehicle", describe_vehicle(problem));
    result.set("transcription", describe_weights(problem));
    result.set("solver", describe_solver(problem));
    json::Value output = json::Value::object();
    output.set("dense_replay_substeps", static_cast<double>(problem.output.dense_replay_substeps));
    output.set("include_iterations", problem.output.include_iterations);
    result.set("output", output);
    result.set("warm_start_supplied", problem.warm_start.has_value());
    return result;
}

[[nodiscard]] inline json::Value describe_evaluation(const Evaluation& evaluation) {
    json::Value path = json::Value::object();
    json::Value physical = json::Value::object();
    for (const auto& component : evaluation.path) {
        path.set(component.name, component.normalised);
        physical.set(component.name, component.physical);
    }
    json::Value result = json::Value::object();
    result.set("objective", evaluation.objective);
    result.set("objective_definition", evaluation.objective_definition);
    result.set("path_violation", evaluation.path_violation);
    result.set("path_components", path);
    result.set("path_components_physical", physical);
    result.set("terminal_residual", evaluation.terminal_residual);
    result.set("terminal_errors", json::Value::numbers(evaluation.terminal_errors));
    result.set("terminal_position_error", evaluation.terminal_position_error);
    result.set("terminal_velocity_error", evaluation.terminal_velocity_error);
    result.set("propellant_used", evaluation.propellant_used);
    result.set("final_mass", evaluation.final_mass);
    return result;
}

}  // namespace spacepdhcg::planner
