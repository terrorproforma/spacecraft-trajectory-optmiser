"""Complete native driver APIs exposed by the full C++ compilation gate."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement site, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    model_path = "cpp/include/spacepdhcg/dynamics/powered_descent_6dof.hpp"
    replace_once(
        model_path,
        "    [[nodiscard]] PoweredDescent6DofState rk4_step(\n",
        "    [[nodiscard]] PoweredDescent6DofState euler_step(\n"
        "        const PoweredDescent6DofState& state,\n"
        "        const PoweredDescent6DofControl& control,\n"
        "        double step_seconds\n"
        "    ) const {\n"
        "        require_step(step_seconds);\n"
        "        const auto derivative = dynamics(state, control);\n"
        "        PoweredDescent6DofState next{};\n"
        "        for (std::size_t component = 0; component < next.size(); ++component) {\n"
        "            next[component] = state[component] + step_seconds * derivative[component];\n"
        "        }\n"
        "        normalise_quaternion(next);\n"
        "        validate_state(next, true);\n"
        "        return next;\n"
        "    }\n\n"
        "    [[nodiscard]] PoweredDescent6DofState rk4_step(\n",
    )

    transcription_path = "cpp/include/spacepdhcg/transcription/powered_descent_3dof.hpp"
    replace_once(
        transcription_path,
        "class PoweredDescent3DofSubproblem {\n",
        "struct PoweredDescentDecodedDecision {\n"
        "    std::vector<PoweredDescentState> states{};\n"
        "    std::vector<PoweredDescentControl> controls{};\n"
        "    std::vector<PoweredDescentState> virtual_controls{};\n"
        "    std::vector<PoweredDescentState> virtual_epigraphs{};\n"
        "};\n\n"
        "class PoweredDescent3DofSubproblem {\n",
    )
    replace_once(
        transcription_path,
        "        return decision;\n"
        "    }\n\n"
        "    [[nodiscard]] PoweredDescentConvexDiagnostics diagnostics(\n",
        "        return decision;\n"
        "    }\n\n"
        "    [[nodiscard]] PoweredDescentDecodedDecision decode(\n"
        "        const std::vector<double>& decision\n"
        "    ) const {\n"
        "        if (decision.size() != layout_.variables()) {\n"
        "            throw std::invalid_argument(\n"
        "                \"powered-descent decision vector has the wrong size\"\n"
        "            );\n"
        "        }\n"
        "        for (const auto value : decision) {\n"
        "            if (!std::isfinite(value)) {\n"
        "                throw std::invalid_argument(\n"
        "                    \"powered-descent decision vector must be finite\"\n"
        "                );\n"
        "            }\n"
        "        }\n"
        "        PoweredDescentDecodedDecision decoded{};\n"
        "        decoded.states.resize(layout_.intervals + 1U);\n"
        "        decoded.controls.resize(layout_.intervals);\n"
        "        decoded.virtual_controls.resize(layout_.intervals);\n"
        "        decoded.virtual_epigraphs.resize(layout_.intervals);\n"
        "        for (std::size_t node = 0; node <= layout_.intervals; ++node) {\n"
        "            const auto range = layout_.state(node);\n"
        "            std::copy_n(\n"
        "                decision.begin() + static_cast<std::ptrdiff_t>(range.start),\n"
        "                7U,\n"
        "                decoded.states[node].begin()\n"
        "            );\n"
        "        }\n"
        "        for (std::size_t interval = 0; interval < layout_.intervals; ++interval) {\n"
        "            const auto control = layout_.control(interval);\n"
        "            const auto virtual_control = layout_.virtual_control(interval);\n"
        "            const auto epigraph = layout_.virtual_epigraph(interval);\n"
        "            std::copy_n(\n"
        "                decision.begin() + static_cast<std::ptrdiff_t>(control.start),\n"
        "                4U,\n"
        "                decoded.controls[interval].begin()\n"
        "            );\n"
        "            std::copy_n(\n"
        "                decision.begin() + static_cast<std::ptrdiff_t>(virtual_control.start),\n"
        "                7U,\n"
        "                decoded.virtual_controls[interval].begin()\n"
        "            );\n"
        "            std::copy_n(\n"
        "                decision.begin() + static_cast<std::ptrdiff_t>(epigraph.start),\n"
        "                7U,\n"
        "                decoded.virtual_epigraphs[interval].begin()\n"
        "            );\n"
        "        }\n"
        "        return decoded;\n"
        "    }\n\n"
        "    [[nodiscard]] PoweredDescentConvexDiagnostics diagnostics(\n",
    )

    driver_path = "cpp/include/spacepdhcg/scvx/powered_descent_3dof_driver.hpp"
    replace_once(driver_path, "TrustAction::hold", "TrustAction::retain")
    replace_once(
        driver_path,
        "                previous_agreement\n"
        "            );",
        "                previous_agreement.value_or(\n"
        "                    std::numeric_limits<double>::quiet_NaN()\n"
        "                )\n"
        "            );",
    )
    replace_once(
        driver_path,
        "forcing_.config().convergence_iteration_limit",
        "forcing_.config().refinement_iteration_limit",
    )
    replace_once(
        driver_path,
        "                subproblem_.config().step_seconds,\n"
        "                false\n"
        "            );",
        "                subproblem_.config().step_seconds,\n"
        "                subproblem_.config().discretisation\n"
        "                    == transcription::DiscretisationMethod::rk4_finite_difference\n"
        "            );",
    )


if __name__ == "__main__":
    main()
