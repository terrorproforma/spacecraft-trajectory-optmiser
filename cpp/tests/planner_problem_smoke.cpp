// Host smoke test for the planner document model and family adapters.
#include "spacepdhcg/planner/describe.hpp"
#include "spacepdhcg/planner/families.hpp"
#include "spacepdhcg/planner/json.hpp"
#include "spacepdhcg/planner/problem.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

namespace planner = spacepdhcg::planner;
namespace json = spacepdhcg::planner::json;

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

template <typename Function>
void require_throws(Function&& function, const char* message) {
    bool threw = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, message);
}

const std::string_view pd3_document = R"({
  "schema_version": "1.0.0",
  "family": "powered_descent_3dof",
  "horizon": {"intervals": 6, "final_time": 3.0},
  "initial_state": [1.0, -0.5, 100.0, 0.0, 0.0, 0.0, 2000.0],
  "terminal": {"state": [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
               "fixed": [true, true, true, true, true, true, false]},
  "vehicle": {"specific_impulse": 221.6},
  "solver": {"backend": "pure_qoco", "time_limit_seconds": 30}
})";

const std::string_view hcw_document = R"({
  "schema_version": "1.0.0",
  "family": "hcw",
  "horizon": {"intervals": 5, "final_time": 100.0},
  "initial_state": [10.0, -5.0, 2.0, 0.0, 0.0, 0.0],
  "terminal": {"state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
  "solver": {"backend": "pdhcg"}
})";

const std::string_view pd6_document = R"({
  "schema_version": "1.0.0",
  "family": "powered_descent_6dof",
  "horizon": {"intervals": 4, "final_time": 1.0},
  "initial_state": [0.0, 0.0, 100.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2000.0],
  "terminal": {"state": [0.0, 0.0, 99.0, 0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
  "solver": {"backend": "pdhcg_recovery"}
})";

const std::string_view low_thrust_document = R"({
  "schema_version": "1.0.0",
  "family": "low_thrust",
  "horizon": {"intervals": 4, "final_time": 240.0},
  "initial_state": [7000.0, 0.0, 0.0, 0.0, 7.5460532901075412, 0.0, 500.0],
  "terminal": {"state": [6995.0, 1810.0, 0.0, -1.95, 7.29, 0.0, 0.0]},
  "solver": {"backend": "cpu_reference"}
})";

void check_json_round_trip() {
    const auto value = json::parse(R"({"a": [1, 2.5, -3e-2, true, null, "x\"y\u00e9"], "b": {}})");
    require(value.at("a").size() == 6U, "array size");
    require(value.at("a").as_array()[2U].as_number() == -3.0e-2, "number parse");
    require(value.at("a").as_array()[5U].as_string() == "x\"y\xC3\xA9", "string escapes");
    const auto text = json::dump(value);
    const auto again = json::parse(text);
    require(json::dump(again) == text, "dump/parse round trip");
    require_throws([] { static_cast<void>(json::parse("{\"a\": 1,}")); }, "trailing comma rejected");
    require_throws([] { static_cast<void>(json::parse("{\"a\": 1, \"a\": 2}")); }, "duplicate key rejected");
    require_throws([] { static_cast<void>(json::parse("[1] x")); }, "trailing garbage rejected");
    json::Value nonfinite = json::Value::object();
    nonfinite.set("inf", std::numeric_limits<double>::infinity());
    require(json::dump(nonfinite) == "{\"inf\":null}", "non-finite renders as null");
}

void check_family(std::string_view document_text, planner::Family family) {
    const auto problem = planner::parse_problem_text(document_text);
    require(problem.family == family, "family parsed");
    const auto adapter = planner::make_adapter(problem);
    const auto& info = adapter->layout();
    require(info.state_dimension == planner::state_dimension(family), "state dimension");
    require(info.control_dimension == planner::control_dimension(family), "control dimension");
    require(info.state_variables.size() == (problem.intervals + 1U) * info.state_dimension, "state variables");
    require(info.control_variables.size() == problem.intervals * info.control_dimension, "control variables");
    require(
        info.state_positions.size() == problem.intervals * info.state_dimension * info.state_dimension,
        "dynamics state positions"
    );
    require(
        info.quadratic_diagonal_positions.size()
            == info.state_variables.size() + info.control_variables.size(),
        "quadratic diagonal positions"
    );
    const bool has_virtual = family != planner::Family::hcw;
    require(info.virtual_variables.empty() != has_virtual, "virtual variables presence");
    const auto reference = adapter->initial_reference();
    const auto replay = adapter->rollout(problem.initial_state, reference.controls, 1U);
    require(
        planner::FamilyAdapter::infinity_distance(reference.states, replay) == 0.0,
        "initial reference is dynamics-consistent"
    );
    const auto dense = adapter->rollout(problem.initial_state, reference.controls, 3U);
    require(dense.size() == (problem.intervals * 3U + 1U) * info.state_dimension, "dense replay size");
    const auto values = adapter->values(reference, problem.solver.trust.initial_radius);
    require(values.linear_objective.size() == info.variables, "values size");
    require(values.scalar_lower.size() == info.scalar_rows, "scalar rows");
    require(values.affine_offset.size() == info.affine_rows, "affine rows");
    const auto evaluation = adapter->evaluate(reference.states, reference.controls);
    require(std::isfinite(evaluation.objective), "objective finite");
    require(evaluation.terminal_errors.size() == info.terminal_dimension, "terminal errors");
    require(!evaluation.path.empty(), "path components present");
    const auto described = planner::describe_problem(problem);
    require(described.at("family").as_string() == std::string(planner::family_name(family)), "describe family");
    const auto defaults = planner::default_document(family);
    require(defaults.at("terminal_fixed").size() == info.state_dimension, "default terminal pattern");
    static_cast<void>(json::dump(planner::describe_evaluation(evaluation)));
}

void check_rejections() {
    require_throws(
        [] {
            static_cast<void>(planner::parse_problem_text(R"({"schema_version": "1.0.0", "family": "hcw",
                "horizon": {"intervals": 4, "final_time": 10.0, "free_final_time": true},
                "initial_state": [0,0,0,0,0,0], "terminal": {"state": [0,0,0,0,0,0]}})"));
        },
        "free final time rejected"
    );
    require_throws(
        [] {
            static_cast<void>(planner::parse_problem_text(R"({"schema_version": "1.0.0",
                "family": "powered_descent_3dof", "horizon": {"intervals": 4, "final_time": 2.0},
                "initial_state": [0,0,100,0,0,0,2000],
                "terminal": {"state": [0,0,0,0,0,0,0], "fixed": [true,true,true,true,true,true,true]}})"));
        },
        "fixed terminal mass rejected"
    );
    require_throws(
        [] {
            static_cast<void>(planner::parse_problem_text(R"({"schema_version": "1.0.0",
                "family": "powered_descent_3dof", "horizon": {"intervals": 4, "final_time": 2.0},
                "initial_state": [0,0,100,0,0,0,2000], "terminal": {"state": [0,0,0,0,0,0,0]},
                "constraints": {"minimum_altitude": 2.0}})"));
        },
        "non-zero minimum altitude rejected"
    );
    require_throws(
        [] {
            static_cast<void>(planner::parse_problem_text(R"({"schema_version": "1.0.0",
                "family": "powered_descent_3dof", "horizon": {"intervals": 4, "final_time": 2.0},
                "initial_state": [0,0,100,0,0,0,900], "terminal": {"state": [0,0,0,0,0,0,0]}})"));
        },
        "initial mass below dry mass rejected"
    );
    require_throws(
        [] {
            static_cast<void>(planner::parse_problem_text(R"({"schema_version": "1.0.0",
                "family": "hcw", "horizon": {"intervals": 4, "final_time": 10.0},
                "initial_state": [0,0,0,0,0,0], "terminal": {"state": [0,0,0,0,0,0]},
                "solver": {"backend": "pure_qoco", "preset": "fixed_tight_pdhcg"}})"));
        },
        "incompatible preset rejected"
    );
}

}  // namespace

int main() {
    check_json_round_trip();
    check_family(pd3_document, planner::Family::powered_descent_3dof);
    check_family(hcw_document, planner::Family::hcw);
    check_family(pd6_document, planner::Family::powered_descent_6dof);
    check_family(low_thrust_document, planner::Family::low_thrust);
    check_rejections();
    std::printf("{\"case\":\"planner_problem_smoke\",\"families\":4,\"status\":\"ok\"}\n");
    return 0;
}
