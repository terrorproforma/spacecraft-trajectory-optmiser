"""GTOC12 official solution-file model, parser and writer.

The file (``GTOC12_Submission_Format.pdf``) is an ASCII table.  Each line starts with the ship ID
and an event ID.  Event lines (launch ``0``, Venus/Earth/Mars flyby ``-2/-3/-4``, or an asteroid ID)
carry ``t r v m`` and always come in pairs (state immediately before and after the event).  Burn
lines (``-1``) carry ``t Tx Ty Tz``; a burning arc opens and closes with zero-thrust lines sharing
the epoch of their neighbouring nonzero sample.  Samples inside an arc are at most one day apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .constants import (
    ASTEROID_COUNT,
    EVENT_BURN,
    EVENT_EARTH_FLYBY,
    EVENT_LAUNCH,
    EVENT_MARS_FLYBY,
    EVENT_VENUS_FLYBY,
)

FloatArray = NDArray[np.float64]
FLYBY_EVENTS = frozenset({EVENT_VENUS_FLYBY, EVENT_EARTH_FLYBY, EVENT_MARS_FLYBY})
EARTH_ID_EVENT = EVENT_LAUNCH


class SolutionFormatError(ValueError):
    """Raised for a structurally invalid solution file (mirrors the official ``ErrorAxx`` codes)."""

    def __init__(self, code: str, line: int | None, message: str) -> None:
        self.code = code
        self.line = line
        location = "" if line is None else f" on line {line}"
        super().__init__(f"{code}{location}: {message}")


@dataclass(frozen=True, slots=True)
class StateLine:
    """One event line: epoch (MJD), position (km), velocity (km/s), mass (kg)."""

    epoch: float
    position: FloatArray
    velocity: FloatArray
    mass: float
    line_number: int = 0


@dataclass(frozen=True, slots=True)
class ThrustSample:
    epoch: float
    thrust: FloatArray  # N
    line_number: int = 0

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.thrust))


@dataclass(frozen=True, slots=True)
class Event:
    """A paired before/after event.  ``event_id`` is 0, -2, -3, -4 or an asteroid ID."""

    event_id: int
    before: StateLine
    after: StateLine

    @property
    def epoch(self) -> float:
        return self.before.epoch

    @property
    def kind(self) -> str:
        if self.event_id == EVENT_LAUNCH:
            return "launch"
        if self.event_id in FLYBY_EVENTS:
            return "flyby"
        return "rendezvous"

    @property
    def is_asteroid(self) -> bool:
        return self.event_id >= 1


@dataclass(frozen=True, slots=True)
class BurnArc:
    """Zero-thrust opener, ``n >= 2`` interior samples, zero-thrust closer."""

    samples: tuple[ThrustSample, ...]

    @property
    def interior(self) -> tuple[ThrustSample, ...]:
        return self.samples[1:-1]

    @property
    def start(self) -> float:
        return self.samples[0].epoch

    @property
    def end(self) -> float:
        return self.samples[-1].epoch

    def interior_arrays(self) -> tuple[FloatArray, FloatArray]:
        epochs = np.asarray([item.epoch for item in self.interior], dtype=np.float64)
        thrust = np.asarray([item.thrust for item in self.interior], dtype=np.float64)
        return epochs, thrust


@dataclass(slots=True)
class ShipTrajectory:
    ship_id: int
    items: list[Event | BurnArc] = field(default_factory=list)

    @property
    def events(self) -> list[Event]:
        return [item for item in self.items if isinstance(item, Event)]

    @property
    def burns(self) -> list[BurnArc]:
        return [item for item in self.items if isinstance(item, BurnArc)]

    @property
    def launch(self) -> Event:
        first = self.items[0] if self.items else None
        if not isinstance(first, Event) or first.event_id != EVENT_LAUNCH:
            raise SolutionFormatError("ErrorA14", None, "ship section must start with a launch")
        return first

    def asteroid_visits(self) -> list[Event]:
        return [item for item in self.events if item.is_asteroid]


@dataclass(slots=True)
class Solution:
    ships: list[ShipTrajectory]

    @property
    def ship_count(self) -> int:
        return len(self.ships)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(format_solution(self), encoding="utf-8", newline="\n")

    @classmethod
    def read(cls, path: str | Path) -> Solution:
        return parse_solution(Path(path).read_text(encoding="utf-8"))


def _tokens(line: str) -> list[str]:
    return line.replace(",", " ").split()


def _is_zero(vector: FloatArray) -> bool:
    return bool(np.all(vector == 0.0))


def parse_solution(text: str) -> Solution:
    """Parse the official ASCII format, raising ``SolutionFormatError`` for structural faults."""

    ships: list[ShipTrajectory] = []
    current: ShipTrajectory | None = None
    pending_event: tuple[int, StateLine] | None = None
    pending_burn: list[ThrustSample] | None = None
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        raise SolutionFormatError("ErrorA02", None, "the solution file is empty")

    def close_burn(line_number: int) -> None:
        nonlocal pending_burn
        if pending_burn is None:
            return
        raise SolutionFormatError("ErrorA05", line_number, "missing the last line of the arc")

    for number, raw in enumerate(lines, start=1):
        tokens = _tokens(raw)
        if not tokens:
            continue
        if len(tokens) not in (6, 10):
            raise SolutionFormatError(
                "ErrorA09", number, "number of data in this line unreasonable"
            )
        try:
            ship_id = int(tokens[0])
            event_id = int(tokens[1])
            values = np.asarray([float(item) for item in tokens[2:]], dtype=np.float64)
        except ValueError as error:
            raise SolutionFormatError("ErrorA09", number, f"unparseable value: {error}") from error
        if not np.all(np.isfinite(values)):
            raise SolutionFormatError("ErrorA09", number, "non-finite value")
        if ship_id < 1:
            raise SolutionFormatError("ErrorA12", number, "the ID of the ship is wrong")
        if current is None or ship_id != current.ship_id:
            if pending_event is not None:
                raise SolutionFormatError("ErrorA23", number, "unpaired event line")
            close_burn(number)
            if ship_id != len(ships) + 1:
                raise SolutionFormatError("ErrorA12", number, "ship IDs must increase 1..N")
            current = ShipTrajectory(ship_id)
            ships.append(current)

        if event_id == EVENT_BURN:
            if len(tokens) != 6:
                raise SolutionFormatError("ErrorA09", number, "burn line needs t Tx Ty Tz")
            if pending_event is not None:
                raise SolutionFormatError("ErrorA23", number, "event pair interrupted by a burn")
            sample = ThrustSample(float(values[0]), values[1:4].copy(), number)
            if pending_burn is None:
                if not _is_zero(sample.thrust):
                    raise SolutionFormatError(
                        "ErrorA04", number, "thrust magnitude of the first line is not 0"
                    )
                pending_burn = [sample]
                continue
            previous = pending_burn[-1]
            if sample.epoch < previous.epoch:
                raise SolutionFormatError("ErrorA07", number, "time interval less than 0")
            pending_burn.append(sample)
            closes = (
                len(pending_burn) >= 3
                and _is_zero(sample.thrust)
                and sample.epoch == previous.epoch
                and not _is_zero(previous.thrust)
            )
            if closes:
                current.items.append(BurnArc(tuple(pending_burn)))
                pending_burn = None
            continue

        if len(tokens) != 10:
            raise SolutionFormatError("ErrorA09", number, "event line needs t r v m")
        if not (
            event_id == EVENT_LAUNCH or event_id in FLYBY_EVENTS or 1 <= event_id <= ASTEROID_COUNT
        ):
            raise SolutionFormatError("ErrorA10", number, f"unknown event ID {event_id}")
        close_burn(number)
        state = StateLine(
            float(values[0]), values[1:4].copy(), values[4:7].copy(), float(values[7]), number
        )
        if pending_event is None:
            pending_event = (event_id, state)
            continue
        if pending_event[0] != event_id:
            raise SolutionFormatError(
                "ErrorA23", number, "event IDs of one event do not appear in two adjacent lines"
            )
        if not current.items and event_id != EVENT_LAUNCH:
            raise SolutionFormatError("ErrorA14", number, "the event ID of the launch is wrong")
        if current.items and event_id == EVENT_LAUNCH:
            raise SolutionFormatError("ErrorA13", number, "second launch inside one ship section")
        current.items.append(Event(event_id, pending_event[1], state))
        pending_event = None

    if pending_event is not None:
        raise SolutionFormatError("ErrorA23", len(lines), "unpaired trailing event line")
    close_burn(len(lines))
    for ship in ships:
        ship.launch  # noqa: B018 - raises ErrorA14 when the section does not start with a launch
    return Solution(ships)


def _format_state(ship: int, event: int, state: StateLine) -> str:
    r = state.position
    v = state.velocity
    return (
        f"{ship:5d} {event:6d} {state.epoch:20.10f} "
        f"{r[0]:24.9f} {r[1]:24.9f} {r[2]:24.9f} "
        f"{v[0]:22.13f} {v[1]:22.13f} {v[2]:22.13f} {state.mass:22.10f}"
    )


def _format_thrust(ship: int, sample: ThrustSample) -> str:
    t = sample.thrust
    return (
        f"{ship:5d} {EVENT_BURN:6d} {sample.epoch:20.10f} {t[0]:22.14f} {t[1]:22.14f} {t[2]:22.14f}"
    )


def format_solution(solution: Solution) -> str:
    """Serialise with fixed-point fields fine enough for the 1000 km / 1 m/s / 1 g tolerances."""

    lines: list[str] = []
    for ship in solution.ships:
        for item in ship.items:
            if isinstance(item, Event):
                lines.append(_format_state(ship.ship_id, item.event_id, item.before))
                lines.append(_format_state(ship.ship_id, item.event_id, item.after))
            else:
                lines.extend(_format_thrust(ship.ship_id, sample) for sample in item.samples)
    # No trailing newline: the official verifier reports the resulting empty last line as
    # ``ErrorA09 ... This line is empty!``.
    return "\n".join(lines)


def make_burn_arc(epochs: FloatArray, thrust: FloatArray) -> BurnArc:
    """Wrap interior samples with the mandatory zero-thrust opener and closer."""

    epochs = np.asarray(epochs, dtype=np.float64)
    thrust = np.asarray(thrust, dtype=np.float64)
    if epochs.ndim != 1 or thrust.shape != (epochs.shape[0], 3) or epochs.shape[0] < 2:
        raise ValueError("a burn arc needs at least two interior samples with 3-vector thrust")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError("burn sample epochs must strictly increase")
    zero = np.zeros(3)
    samples = [ThrustSample(float(epochs[0]), zero.copy())]
    samples.extend(
        ThrustSample(float(t), np.asarray(f, dtype=np.float64))
        for t, f in zip(epochs, thrust, strict=True)
    )
    samples.append(ThrustSample(float(epochs[-1]), zero.copy()))
    return BurnArc(tuple(samples))


def make_event(
    event_id: int, epoch: float, position, velocity_before, velocity_after, mass_before, mass_after
) -> Event:
    position = np.asarray(position, dtype=np.float64)
    return Event(
        int(event_id),
        StateLine(
            float(epoch),
            position.copy(),
            np.asarray(velocity_before, dtype=np.float64),
            float(mass_before),
        ),
        StateLine(
            float(epoch),
            position.copy(),
            np.asarray(velocity_after, dtype=np.float64),
            float(mass_after),
        ),
    )
