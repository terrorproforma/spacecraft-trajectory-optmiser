"""User-facing planner API on top of the validated single-GPU device SCvx stack.

from spacepdhcg.planner import plan, PlanOptions
result = plan("examples/planner/powered_descent_3dof.json", PlanOptions(output_directory="out"))
print(result.status, result.certified, result.objective)
"""

from spacepdhcg.planner.api import (
    GPUUnavailableError,
    PlanExecutionError,
    PlanOptions,
    plan,
)
from spacepdhcg.planner.native_library import PlannerNativeError, PlannerTranscription
from spacepdhcg.planner.problem import (
    BACKENDS,
    FAMILIES,
    FAMILY_INFO,
    PRESETS,
    SCHEMA_VERSION,
    ProblemValidationError,
    load_problem,
    normalise_problem,
    schema_errors,
)
from spacepdhcg.planner.result import PlanResult, load_result

__all__ = [
    "BACKENDS",
    "FAMILIES",
    "FAMILY_INFO",
    "PRESETS",
    "SCHEMA_VERSION",
    "GPUUnavailableError",
    "PlanExecutionError",
    "PlanOptions",
    "PlanResult",
    "PlannerNativeError",
    "PlannerTranscription",
    "ProblemValidationError",
    "load_problem",
    "load_result",
    "normalise_problem",
    "plan",
    "schema_errors",
]
