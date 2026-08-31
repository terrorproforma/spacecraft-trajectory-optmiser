#include "spacepdhcg/core/fixed_cqp.hpp"
#include "spacepdhcg/distributed/scenario_layout.hpp"
#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"
#include "spacepdhcg/orbitweaver/lambert.hpp"
#include "spacepdhcg/transcription/powered_descent_3dof.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

namespace {

template <typename Container>
void print_array(const Container& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0U) {
            std::cout << ',';
        }
        std::cout << values[index];
    }
    std::cout << ']';
}

}  // namespace

int main() {
    using spacepdhcg::ConeBlockDescriptor;
    using spacepdhcg::ConeKind;
    using spacepdhcg::core::CscPattern;
    using spacepdhcg::core::FixedStructure;
    using spacepdhcg::distributed::BlockArrowLayout;
    using spacepdhcg::distributed::ScenarioTree;
    using spacepdhcg::dynamics::PoweredDescent3DofModel;
    using spacepdhcg::dynamics::PoweredDescentControl;
    using spacepdhcg::dynamics::PoweredDescentState;
    using spacepdhcg::orbitweaver::Vector3;
    using spacepdhcg::orbitweaver::solve_lambert_zero_revolution;
    using spacepdhcg::transcription::PoweredDescent3DofSubproblem;
    using spacepdhcg::transcription::PoweredDescentScvxConfig;

    const FixedStructure tiny{
        CscPattern{2, 2, {0, 1, 2}, {0, 1}},
        CscPattern{1, 2, {0, 1, 2}, {0, 0}},
        CscPattern{3, 2, {0, 1, 2}, {0, 1}},
        {ConeBlockDescriptor{ConeKind::second_order, 0, 1, 0.0}},
        {},
    };
    tiny.validate();

    const PoweredDescent3DofModel model{};
    const PoweredDescentState state{20.0, -10.0, 120.0, 0.4, -0.2, -7.0, 2'000.0};
    const std::array<double, 3U> thrust{1'200.0, -500.0, 8'000.0};
    const auto sigma = std::sqrt(
        thrust[0U] * thrust[0U] + thrust[1U] * thrust[1U] + thrust[2U] * thrust[2U]
    );
    const PoweredDescentControl control{thrust[0U], thrust[1U], thrust[2U], sigma};
    const auto derivative = model.dynamics(state, control);
    const auto jacobians = model.jacobians(state, control);

    const auto tree = ScenarioTree::common_open_loop(4U, 5U, 3U);
    const BlockArrowLayout layout(tree, 7U, 4U, 12U);

    const auto lambert = solve_lambert_zero_revolution(
        Vector3{5'000.0, 10'000.0, 2'100.0},
        Vector3{-14'600.0, 2'500.0, 7'000.0},
        3'600.0,
        398'600.0,
        false,
        1.0e-9
    );

    const PoweredDescent3DofSubproblem transcription(
        model,
        PoweredDescentScvxConfig{
            .intervals = 4U,
            .step_seconds = 1.0,
            .trust_radius = 1.0,
        }
    );

    std::cout << std::setprecision(17);
    std::cout << '{';
    std::cout << "\"tiny_fingerprint\":" << tiny.fingerprint() << ',';
    std::cout << "\"dynamics\":";
    print_array(derivative);
    std::cout << ",\"state_jacobian\":";
    print_array(jacobians.state);
    std::cout << ",\"control_jacobian\":";
    print_array(jacobians.control);
    std::cout << ",\"scenario\":{";
    std::cout << "\"nodes\":" << tree.nodes().size() << ',';
    std::cout << "\"shared_nodes\":" << tree.shared_nodes().size() << ',';
    std::cout << "\"variables\":" << layout.total_variables() << ',';
    std::cout << "\"consensus_dimension\":" << layout.consensus_dimension() << ',';
    std::cout << "\"nonanticipativity_rows\":" << layout.nonanticipativity_rows();
    std::cout << "},\"lambert_departure\":";
    print_array(lambert.departure_velocity);
    std::cout << ",\"lambert_arrival\":";
    print_array(lambert.arrival_velocity);
    std::cout << ",\"transcription\":{";
    std::cout << "\"variables\":" << transcription.layout().variables() << ',';
    std::cout << "\"scalar_rows\":" << transcription.layout().scalar_rows() << ',';
    std::cout << "\"affine_rows\":" << transcription.layout().affine_rows() << ',';
    std::cout << "\"quadratic_nonzeros\":" << transcription.structure().quadratic.nonzeros()
              << ',';
    std::cout << "\"scalar_nonzeros\":"
              << transcription.structure().scalar_constraint.nonzeros() << ',';
    std::cout << "\"affine_nonzeros\":"
              << transcription.structure().affine_cone->nonzeros() << ',';
    std::cout << "\"cone_blocks\":" << transcription.structure().affine_cones.size() << ',';
    std::cout << "\"fingerprint\":" << transcription.structure().fingerprint();
    std::cout << "}}\n";
    return 0;
}
