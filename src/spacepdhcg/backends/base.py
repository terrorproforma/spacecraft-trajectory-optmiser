"""Backend lifecycle shared by CPU references and future GPU solvers."""

from __future__ import annotations

from typing import Protocol

from numpy.typing import NDArray

from spacepdhcg.cqp import CQPSolution, CQPStructure, CQPValues


class PersistentCQPBackend(Protocol):
    """A solver workspace whose symbolic structure is allocated exactly once."""

    structure: CQPStructure
    update_count: int

    def update(self, values: CQPValues) -> None:
        """Replace numerical values without changing symbolic structure."""

    def warm_start(
        self,
        primal: NDArray | None = None,
        dual: NDArray | None = None,
    ) -> None:
        """Set a primal and/or dual warm start."""

    def solve(
        self,
        *,
        tolerance: float | None = None,
        iteration_limit: int | None = None,
    ) -> CQPSolution:
        """Solve the currently loaded numerical problem."""
