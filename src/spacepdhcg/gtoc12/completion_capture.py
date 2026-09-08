"""Optional immutable native readbacks for bounded fixed-cargo refinement.

A recorder never accepts a route. Its consumer supplies the fleet objective and
offers an eligible prescription to RefinementAdmissionQueue. The frozen run
launcher supplies the verified loaded-library SHA; this module cannot infer it
from an on-disk library which might have changed since loading.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .refinement_admission import CompletionObservation, FixedCargoRequest


@dataclass(frozen=True, slots=True)
class CompletionEnvelope:
    plan_summary: bytes
    request: FixedCargoRequest
    observation: CompletionObservation
    raw_result: bytes
    raw_legs: bytes
    raw_collected: bytes
    input_sha256: str
    model_sha256: str | None


class CompletionRecorder:
    """Copy complete native outcomes before rejection/workspace reuse loses them.

    ``consume(envelope)`` must return its journal disposition. It may retain at
    most its queue capacity; this recorder keeps only counters, never an unbounded
    route history. Unsupported inventories/nonfinite inputs are journaled through
    ``on_blocked`` and remain ordinary completion outcomes.
    """

    def __init__(self, producer_sha256, consume, *, on_blocked=None):
        if len(producer_sha256) != 64 or any(c not in "0123456789abcdef" for c in producer_sha256):
            raise ValueError("completion capture requires a frozen producer SHA256")
        if not callable(consume):
            raise TypeError("completion consumer must be callable")
        self.producer_sha256 = producer_sha256
        self.consume = consume
        self.on_blocked = on_blocked
        self.dispositions = Counter()

    def __call__(self, row, result, details, collected, inputs, model_sha256=None):
        from .gpu_completion import FAILURES
        from .search import RoutePlan

        failure = int(result["failure"])
        if failure not in (0, 5):
            self.dispositions["earlier_completion_gate"] += 1
            return
        partial, deploy, collect, forward, _ = row
        raw_result, raw_legs, raw_collected = (
            result.tobytes(),
            details.tobytes(),
            collected.tobytes(),
        )
        raw_input = b"".join(a.tobytes() for a in inputs)
        input_sha = hashlib.sha256(raw_input + (model_sha256 or "").encode()).hexdigest()
        readback_sha = hashlib.sha256(
            raw_result + raw_legs + raw_collected + input_sha.encode()
        ).hexdigest()
        cargo = {}
        legs = list(partial.legs)
        for leg, detail in zip(forward, details, strict=True):
            if detail["pickup"]:
                cargo[leg.from_id] = float(detail["gained"])
            legs.append(replace(leg, inflation=float(detail["inflation"])))
        expected_collected = np.asarray([cargo.get(body, 0.0) for body in deploy])
        if collected.shape != expected_collected.shape or not np.array_equal(
            collected, expected_collected
        ):
            self.dispositions["collected_readback_mismatch"] += 1
            if self.on_blocked is not None:
                self.on_blocked("collected_readback_mismatch", "", input_sha, readback_sha)
            return
        # This temporary object supplies the existing serialization only. It is
        # never returned to search or admitted using its proxy feasibility flag.
        plan = RoutePlan(
            tuple(legs),
            dict(deploy),
            dict(collect),
            cargo,
            float(result["propellant"]),
            float(result["final_mass"]),
        )
        try:
            summary = plan.summary()
            prescription = FixedCargoRequest.from_summary(summary)
            encoded = json.dumps(
                summary, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        except (ValueError, KeyError, IndexError, TypeError) as error:
            self.dispositions["unsupported_prescription"] += 1
            if self.on_blocked is not None:
                self.on_blocked("unsupported_prescription", str(error), input_sha, readback_sha)
            return
        costs = details["propellant"]
        factors = details["inflation"][details["stage"] == 4]
        observation = CompletionObservation(
            prescription.sha256,
            self.producer_sha256,
            readback_sha,
            FAILURES[failure],
            len(forward),
            int(result["processed_legs"]),
            tuple(int(x) for x in details["stage"]),
            float(result["final_mass"]),
            float(result["collected"]),
            bool(
                np.all(np.isfinite(costs))
                and np.all(costs >= 0)
                and np.all(np.isfinite(factors))
                and np.all(factors >= 0)
            ),
        )
        blocker = observation.blocker(prescription)
        if blocker:
            self.dispositions[blocker] += 1
            if self.on_blocked is not None:
                self.on_blocked(blocker, "", input_sha, readback_sha)
            return
        envelope = CompletionEnvelope(
            encoded,
            prescription,
            observation,
            raw_result,
            raw_legs,
            raw_collected,
            input_sha,
            model_sha256,
        )
        disposition = self.consume(envelope)
        if not isinstance(disposition, str):
            raise TypeError("completion consumer must return its journal disposition")
        self.dispositions[disposition] += 1
