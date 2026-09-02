// Smoke test for the planner transcription C ABI exported by libspacepdhcg.
#include "spacepdhcg/c_api.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (%s)\n", message, spacepdhcg_last_error());
        std::exit(1);
    }
}

const char* document = R"({
  "schema_version": "1.0.0",
  "family": "low_thrust",
  "horizon": {"intervals": 3, "final_time": 180.0},
  "initial_state": [7000.0, 0.0, 0.0, 0.0, 7.5460532901075412, 0.0, 500.0],
  "terminal": {"state": [6997.0, 1358.0, 0.0, -1.46, 7.40, 0.0, 0.0]},
  "solver": {"backend": "cpu_reference"}
})";

}  // namespace

int main() {
    spacepdhcg_planner* planner = nullptr;
    require(spacepdhcg_planner_create(document, &planner) == SPACEPDHCG_STATUS_OK, "create");
    require(planner != nullptr, "handle");
    spacepdhcg_planner_dimensions dimensions{};
    require(
        spacepdhcg_planner_get_dimensions(planner, &dimensions) == SPACEPDHCG_STATUS_OK,
        "dimensions"
    );
    require(dimensions.state_dimension == 7U && dimensions.control_dimension == 4U, "shape");
    require(dimensions.intervals == 3U && dimensions.terminal_dimension == 6U, "horizon");
    require(dimensions.virtual_variable_count == 3U * 7U, "virtual count");

    std::vector<int32_t> q_offsets(dimensions.variables + 1U);
    std::vector<int32_t> q_indices(dimensions.quadratic_nonzeros);
    std::vector<int32_t> a_offsets(dimensions.variables + 1U);
    std::vector<int32_t> a_indices(dimensions.scalar_nonzeros);
    std::vector<int32_t> f_offsets(dimensions.variables + 1U);
    std::vector<int32_t> f_indices(dimensions.affine_nonzeros);
    std::vector<spacepdhcg_planner_cone> affine_cones(dimensions.affine_cone_count);
    std::vector<spacepdhcg_planner_cone> variable_cones(dimensions.variable_cone_count + 1U);
    std::vector<int32_t> state_variables((dimensions.intervals + 1U) * 7U);
    std::vector<int32_t> control_variables(dimensions.intervals * 4U);
    std::vector<int32_t> virtual_variables(dimensions.virtual_variable_count);
    require(
        spacepdhcg_planner_structure(
            planner,
            q_offsets.data(),
            q_indices.data(),
            a_offsets.data(),
            a_indices.data(),
            f_offsets.data(),
            f_indices.data(),
            affine_cones.data(),
            variable_cones.data(),
            state_variables.data(),
            control_variables.data(),
            virtual_variables.data()
        ) == SPACEPDHCG_STATUS_OK,
        "structure"
    );
    require(
        static_cast<uint64_t>(q_offsets.back()) == dimensions.quadratic_nonzeros,
        "quadratic offsets"
    );
    require(static_cast<uint64_t>(a_offsets.back()) == dimensions.scalar_nonzeros, "scalar offsets");
    require(static_cast<uint64_t>(f_offsets.back()) == dimensions.affine_nonzeros, "affine offsets");
    require(state_variables.front() == 0 && control_variables.front() == 28, "variable order");

    std::vector<double> states((dimensions.intervals + 1U) * 7U);
    std::vector<double> controls(dimensions.intervals * 4U);
    require(
        spacepdhcg_planner_initial_reference(planner, states.data(), controls.data())
            == SPACEPDHCG_STATUS_OK,
        "initial reference"
    );
    std::vector<double> replay(states.size());
    require(
        spacepdhcg_planner_rollout(
            planner, states.data(), controls.data(), dimensions.intervals, 1U, replay.data()
        ) == SPACEPDHCG_STATUS_OK,
        "rollout"
    );
    for (std::size_t index = 0U; index < states.size(); ++index) {
        require(states[index] == replay[index], "reference is dynamics-consistent");
    }
    std::vector<double> dense((dimensions.intervals * 4U + 1U) * 7U);
    require(
        spacepdhcg_planner_rollout(
            planner, states.data(), controls.data(), dimensions.intervals, 4U, dense.data()
        ) == SPACEPDHCG_STATUS_OK,
        "dense rollout"
    );

    std::vector<double> quadratic(dimensions.quadratic_nonzeros);
    std::vector<double> scalar(dimensions.scalar_nonzeros);
    std::vector<double> affine(dimensions.affine_nonzeros);
    std::vector<double> linear(dimensions.variables);
    std::vector<double> lower(dimensions.scalar_rows);
    std::vector<double> upper(dimensions.scalar_rows);
    std::vector<double> offset(dimensions.affine_rows);
    std::vector<double> variable_lower(dimensions.variables);
    std::vector<double> variable_upper(dimensions.variables);
    require(
        spacepdhcg_planner_values(
            planner,
            states.data(),
            controls.data(),
            1.0,
            quadratic.data(),
            scalar.data(),
            affine.data(),
            linear.data(),
            lower.data(),
            upper.data(),
            offset.data(),
            variable_lower.data(),
            variable_upper.data()
        ) == SPACEPDHCG_STATUS_OK,
        "values"
    );
    require(lower[0] == states[0] && upper[0] == states[0], "initial state rows pinned");

    spacepdhcg_planner_evaluation evaluation{};
    require(
        spacepdhcg_planner_evaluate(planner, states.data(), controls.data(), &evaluation)
            == SPACEPDHCG_STATUS_OK,
        "evaluate"
    );
    require(evaluation.path_component_count == 4U, "low-thrust path components");
    require(std::strcmp(evaluation.path_names[3], "minimum_radius") == 0, "path names");
    require(evaluation.terminal_residual > 0.0 && std::isfinite(evaluation.objective), "evaluation");
    spacepdhcg_planner_evaluation dense_evaluation{};
    std::vector<double> dense_controls;
    for (std::size_t interval = 0U; interval < dimensions.intervals; ++interval) {
        for (int repeat = 0; repeat < 4; ++repeat) {
            dense_controls.insert(
                dense_controls.end(),
                controls.begin() + static_cast<std::ptrdiff_t>(interval * 4U),
                controls.begin() + static_cast<std::ptrdiff_t>((interval + 1U) * 4U)
            );
        }
    }
    require(
        spacepdhcg_planner_path_components(
            planner, dense.data(), dense_controls.data(), dimensions.intervals * 4U, &dense_evaluation
        ) == SPACEPDHCG_STATUS_OK,
        "dense path components"
    );

    size_t required = 0U;
    require(
        spacepdhcg_planner_describe(planner, nullptr, 0U, &required) == SPACEPDHCG_STATUS_OK
            && required > 2U,
        "describe size"
    );
    std::string description(required, '\0');
    require(
        spacepdhcg_planner_describe(planner, description.data(), required, &required)
            == SPACEPDHCG_STATUS_OK,
        "describe"
    );
    require(description.find("\"family\":\"low_thrust\"") != std::string::npos, "describe content");
    require(
        spacepdhcg_planner_default_document("hcw", nullptr, 0U, &required) == SPACEPDHCG_STATUS_OK,
        "default document"
    );
    require(
        spacepdhcg_planner_default_document("unknown", nullptr, 0U, &required)
            == SPACEPDHCG_STATUS_INVALID_ARGUMENT,
        "unknown family rejected"
    );
    spacepdhcg_planner* rejected = nullptr;
    require(
        spacepdhcg_planner_create("{\"schema_version\": \"0.1\"}", &rejected)
            == SPACEPDHCG_STATUS_INVALID_ARGUMENT
            && rejected == nullptr,
        "invalid document rejected"
    );
    require(
        spacepdhcg_planner_create("{not json", &rejected) != SPACEPDHCG_STATUS_OK && rejected == nullptr,
        "malformed JSON rejected"
    );
    spacepdhcg_planner_destroy(planner);
    spacepdhcg_planner_destroy(nullptr);
    std::printf("{\"case\":\"planner_c_api_smoke\",\"status\":\"ok\"}\n");
    return 0;
}
