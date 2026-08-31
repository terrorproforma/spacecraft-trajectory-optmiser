"""Normalise admissible 6-DoF initial attitudes at the rollout boundary."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "cpp/include/spacepdhcg/dynamics/powered_descent_6dof.hpp"
    text = path.read_text(encoding="utf-8")
    old = (
        "        validate_state(initial, true);\n"
        "        std::vector<PoweredDescent6DofState> states(controls.size() + 1U);\n"
        "        states.front() = initial;\n"
    )
    new = (
        "        validate_state(initial, false);\n"
        "        std::vector<PoweredDescent6DofState> states(controls.size() + 1U);\n"
        "        states.front() = initial;\n"
        "        normalise_quaternion(states.front());\n"
        "        validate_state(states.front(), true);\n"
    )
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one 6-DoF rollout replacement site, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
