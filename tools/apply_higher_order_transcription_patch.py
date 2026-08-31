"""Apply the native higher-order transcription migration deterministically.

This repository is maintained through GitHub-only writes in the current programme. The script
uses exact, asserted replacements so the migration is auditable and fails rather than silently
editing an unexpected source revision.
"""

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


def patch_transcription(
    path: str,
    dynamics_include: str,
    config_marker: str,
    validation_marker: str,
    constructor_old: str,
    constructor_new: str,
    linearisation_old: str,
    state_dimension: int,
    control_dimension: int,
) -> None:
    replace_once(
        path,
        dynamics_include,
        dynamics_include
        + '#include "spacepdhcg/transcription/discrete_flow_linearisation.hpp"\n'
        + '#include "spacepdhcg/transcription/discretisation.hpp"\n',
    )
    replace_once(
        path,
        config_marker,
        config_marker
        + "    DiscretisationMethod discretisation{DiscretisationMethod::forward_euler};\n"
        + "    double finite_difference_relative_step{1.0e-6};\n",
    )
    replace_once(
        path,
        validation_marker,
        validation_marker
        + "        require_positive(\n"
        + "            finite_difference_relative_step,\n"
        + '            "finite-difference relative step must be positive"\n'
        + "        );\n",
    )
    replace_once(path, constructor_old, constructor_new)
    linearisation_new = (
        f"            const auto linearisation = linearise_discrete_flow<"
        f"{state_dimension}U, {control_dimension}U>(\n"
        "                model_,\n"
        "                states[interval],\n"
        "                controls[interval],\n"
        "                config_.step_seconds,\n"
        "                config_.discretisation,\n"
        "                config_.finite_difference_relative_step\n"
        "            );"
    )
    replace_once(path, linearisation_old, linearisation_new)


def main() -> None:
    patch_transcription(
        "cpp/include/spacepdhcg/transcription/powered_descent_3dof.hpp",
        '#include "spacepdhcg/dynamics/powered_descent_3dof.hpp"\n',
        "    double fuel_weight{1.0e-3};\n",
        '        require_nonnegative(fuel_weight, "fuel weight must be non-negative");\n',
        "        PoweredDescent3DofModel model = {},\n"
        "        PoweredDescentScvxConfig config = {}\n",
        "        PoweredDescent3DofModel model = PoweredDescent3DofModel{},\n"
        "        PoweredDescentScvxConfig config = PoweredDescentScvxConfig{}\n",
        "            const auto linearisation = model_.linearised_euler_dynamics(\n"
        "                states[interval],\n"
        "                controls[interval],\n"
        "                config_.step_seconds\n"
        "            );",
        7,
        4,
    )

    patch_transcription(
        "cpp/include/spacepdhcg/transcription/powered_descent_6dof.hpp",
        '#include "spacepdhcg/dynamics/powered_descent_6dof.hpp"\n',
        "    double fuel_weight{1.0e-3};\n",
        '        require_nonnegative(fuel_weight, "fuel weight must be non-negative");\n',
        "        PoweredDescent6DofModel model = {},\n"
        "        PoweredDescent6DofScvxConfig config = {}\n",
        "        PoweredDescent6DofModel model = PoweredDescent6DofModel{},\n"
        "        PoweredDescent6DofScvxConfig config = PoweredDescent6DofScvxConfig{}\n",
        "            const auto linearisation = model_.linearised_euler_dynamics(\n"
        "                states[interval],\n"
        "                controls[interval],\n"
        "                config_.step_seconds\n"
        "            );",
        14,
        7,
    )

    patch_transcription(
        "cpp/include/spacepdhcg/transcription/low_thrust.hpp",
        '#include "spacepdhcg/dynamics/low_thrust_two_body.hpp"\n',
        "    double fuel_weight{1.0};\n",
        '        require_nonnegative(fuel_weight, "fuel weight must be non-negative");\n',
        "        LowThrustTwoBodyModel model = {},\n"
        "        LowThrustScvxConfig config = {}\n",
        "        LowThrustTwoBodyModel model = LowThrustTwoBodyModel{},\n"
        "        LowThrustScvxConfig config = LowThrustScvxConfig{}\n",
        "            const auto linearisation = model_.linearised_euler_dynamics(\n"
        "                states[interval],\n"
        "                controls[interval],\n"
        "                config_.step_seconds\n"
        "            );",
        7,
        4,
    )

    model_path = "cpp/include/spacepdhcg/dynamics/powered_descent_6dof.hpp"
    replace_once(
        model_path,
        "    double maximum_angular_rate{1.0};\n",
        "    double maximum_angular_rate{1.0};\n"
        "    double maximum_tilt_radians{0.5235987755982988};\n"
        "    double glide_slope_radians{1.0471975511965976};\n",
    )
    replace_once(
        model_path,
        '        require_positive(maximum_angular_rate, "maximum angular rate must be positive");\n',
        '        require_positive(maximum_angular_rate, "maximum angular rate must be positive");\n'
        "        constexpr double half_pi = 1.5707963267948966;\n"
        "        if (!(maximum_tilt_radians > 0.0 && maximum_tilt_radians < half_pi)) {\n"
        '            throw std::invalid_argument("maximum tilt must lie in (0, pi/2)");\n'
        "        }\n"
        "        if (!(glide_slope_radians > 0.0 && glide_slope_radians < half_pi)) {\n"
        '            throw std::invalid_argument("glide slope must lie in (0, pi/2)");\n'
        "        }\n",
    )
    replace_once(
        model_path,
        "\n  private:\n    static void require_positive(double value, const char* message) {\n",
        "\n    [[nodiscard]] double tilt_cosine() const noexcept {\n"
        "        return std::cos(maximum_tilt_radians);\n"
        "    }\n\n"
        "    [[nodiscard]] double glide_slope_tangent() const noexcept {\n"
        "        return std::tan(glide_slope_radians);\n"
        "    }\n\n"
        "  private:\n"
        "    static void require_positive(double value, const char* message) {\n",
    )


if __name__ == "__main__":
    main()
