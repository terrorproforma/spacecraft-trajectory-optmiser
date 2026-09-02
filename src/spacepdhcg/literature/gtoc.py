"""GTOC5 / GTOC9 / GTOC12 data loaders, evaluators, and reduced-subset rules (P2-F, Phase 0/1).

* GTOC12: the official verification program (binary, pinned by checksum inside the pinned zip)
  is executed on the bundled ``Result.txt`` example and on the published solution files.
* GTOC9: no official offline evaluator was published; the Kelvins validation rules 4-19, the
  J2 ephemeris procedure, and the mission cost function are implemented here verbatim and
  exercised on the two official example submissions.  This is a *verified-local* evaluator,
  labelled accordingly.
* GTOC5: problem data and statement are pinned; the only public evaluator (Simoes et al. beam
  P-ACO code) depends on ``pykep``, which is not installed, so scoring is recorded as blocked.

Reduced deterministic subsets are defined by metadata rules only (never by solver scores) and
frozen in ``benchmarks/literature/gtoc_reduced_subsets.json``.
"""

from __future__ import annotations

import math
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.literature import external_sources

FloatArray = NDArray[np.float64]

# --------------------------------------------------------------------------- GTOC9 constants
GTOC9_CONSTANTS: dict[str, float] = {
    "alpha_meur_per_kg2": 2.0e-6,
    "c_m_meur": 45.0,
    "c_M_meur": 55.0,
    "unremoved_debris_cost_meur": 55.0018,
    "delta_t_R_days": 30.0,
    "delta_t_M_days": 30.0,
    "t_w_days": 5.0,
    "tof_years": 8.0,
    "m_de_kg": 30.0,
    "m_dry_kg": 2000.0,
    "m_p_max_kg": 5000.0,
    "r_p_min_m": 6_600_000.0,
    "mu_m3_s2": 398600.4418e9,
    "j2": 1.08262668e-3,
    "r_eq_m": 6_378_137.0,
    "isp_s": 340.0,
    "g0_m_s2": 9.80665,
    "eps_r_m": 100.0,
    "eps_v_m_s": 1.0,
    "eps_m_kg": 1.0e-8,
    "day_s": 86400.0,
    "year_days": 365.25,
    "window_start_mjd2000": 23467.0,
    "window_end_mjd2000": 26419.0,
    "max_lines": 856,
}


@dataclass(frozen=True, slots=True)
class Debris:
    id: int
    t0: float
    a: float
    e: float
    i: float
    raan0: float
    argp0: float
    m0: float


def load_gtoc9_debris(path: Path | None = None) -> list[Debris]:
    path = path or external_sources.fetch("gtoc9.debris")
    rows: list[Debris] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("id"):
            continue
        parts = [part.strip() for part in stripped.split(",")]
        rows.append(
            Debris(
                id=int(parts[0]),
                t0=float(parts[1]),
                a=float(parts[2]),
                e=float(parts[3]),
                i=float(parts[4]),
                raan0=float(parts[5]),
                argp0=float(parts[6]),
                m0=float(parts[7]),
            )
        )
    return rows


def _solve_kepler(M: float, e: float) -> float:
    E = M if e < 0.8 else math.pi
    for _ in range(60):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        step = f / fp
        E -= step
        if abs(step) < 1.0e-14:
            break
    return E


def gtoc9_debris_state(debris: Debris, t_mjd2000: float) -> tuple[FloatArray, FloatArray]:
    """Kelvins 'Equations' page: secular J2 precession then Keplerian conversion (SI units)."""

    c = GTOC9_CONSTANTS
    mu = c["mu_m3_s2"]
    n = math.sqrt(mu / debris.a**3)
    p = debris.a * (1.0 - debris.e**2)
    factor = (c["r_eq_m"] / p) ** 2 * n
    raan_dot = -1.5 * c["j2"] * factor * math.cos(debris.i)
    argp_dot = 0.75 * c["j2"] * factor * (5.0 * math.cos(debris.i) ** 2 - 1.0)
    dt = (t_mjd2000 - debris.t0) * c["day_s"]
    raan = debris.raan0 + raan_dot * dt
    argp = debris.argp0 + argp_dot * dt
    M = debris.m0 + n * dt
    M = math.fmod(M, 2.0 * math.pi)
    E = _solve_kepler(M, debris.e)
    theta = 2.0 * math.atan2(
        math.sqrt(1.0 + debris.e) * math.sin(E / 2.0), math.sqrt(1.0 - debris.e) * math.cos(E / 2.0)
    )
    gamma = math.atan2(debris.e * math.sin(theta), 1.0 + debris.e * math.cos(theta))
    r = debris.a * (1.0 - debris.e**2) / (1.0 + debris.e * math.cos(theta))
    v = math.sqrt(2.0 * mu / r - mu / debris.a)
    u = theta + argp
    ci, si = math.cos(debris.i), math.sin(debris.i)
    cO, sO = math.cos(raan), math.sin(raan)
    position = np.array(
        [
            r * (math.cos(u) * cO - math.sin(u) * ci * sO),
            r * (math.cos(u) * sO + math.sin(u) * ci * cO),
            r * (math.sin(u) * si),
        ]
    )
    ug = u - gamma
    velocity = np.array(
        [
            v * (-math.sin(ug) * cO - math.cos(ug) * ci * sO),
            v * (-math.sin(ug) * sO + math.cos(ug) * ci * cO),
            v * (math.cos(ug) * si),
        ]
    )
    return position, velocity


def gtoc9_j2_acceleration(state: FloatArray) -> FloatArray:
    c = GTOC9_CONSTANTS
    mu = c["mu_m3_s2"]
    x, y, z = state[0:3]
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    coefficient = 1.5 * c["j2"] * (c["r_eq_m"] ** 2) / r2
    z2r2 = z * z / r2
    base = -mu / (r2 * r)
    ax = base * x * (1.0 + coefficient * (1.0 - 5.0 * z2r2))
    ay = base * y * (1.0 + coefficient * (1.0 - 5.0 * z2r2))
    az = base * z * (1.0 + coefficient * (3.0 - 5.0 * z2r2))
    return np.array([state[3], state[4], state[5], ax, ay, az])


def propagate_j2(state: FloatArray, duration_s: float) -> FloatArray:
    from scipy.integrate import solve_ivp

    solution = solve_ivp(
        lambda _t, y: gtoc9_j2_acceleration(y),
        (0.0, duration_s),
        np.asarray(state, dtype=np.float64),
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    if not solution.success:
        raise RuntimeError(f"J2 propagation failed: {solution.message}")
    return solution.y[:, -1]


def osculating_pericentre(position: FloatArray, velocity: FloatArray) -> float:
    mu = GTOC9_CONSTANTS["mu_m3_s2"]
    r = float(np.linalg.norm(position))
    v2 = float(velocity @ velocity)
    energy = v2 / 2.0 - mu / r
    a = -mu / (2.0 * energy)
    h = np.cross(position, velocity)
    e_vec = np.cross(velocity, h) / mu - position / r
    return a * (1.0 - float(np.linalg.norm(e_vec)))


@dataclass(slots=True)
class GTOC9Event:
    t: float
    r: FloatArray
    v: FloatArray
    m: float
    dv: FloatArray
    event_id: int


def parse_gtoc9_submission(path: Path) -> list[GTOC9Event]:
    events: list[GTOC9Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part for part in stripped.replace(",", " ").split()]
        if len(parts) != 12:
            raise ValueError(f"{path}: expected 12 values per line, got {len(parts)}")
        values = [float(part) for part in parts[:11]]
        events.append(
            GTOC9Event(
                t=values[0],
                r=np.array(values[1:4]),
                v=np.array(values[4:7]),
                m=values[7],
                dv=np.array(values[8:11]),
                event_id=int(float(parts[11])),
            )
        )
    return events


@dataclass(slots=True)
class GTOC9MissionEvaluation:
    valid: bool
    violations: list[str]
    debris_removed: list[int]
    initial_mass_kg: float
    final_mass_kg: float
    propellant_mass_kg: float
    mass_penalty_meur: float
    mission_cost_min_meur: float
    mission_cost_max_meur: float
    max_rendezvous_position_error_m: float
    max_rendezvous_velocity_error_m_s: float
    max_propagation_position_error_m: float
    max_propagation_velocity_error_m_s: float
    lines: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_gtoc9_mission(
    path: Path, debris: list[Debris] | None = None
) -> GTOC9MissionEvaluation:
    """Apply the Kelvins preliminary-validation rules 4-19 and compute the mission cost."""

    c = GTOC9_CONSTANTS
    debris_table = {item.id: item for item in (debris or load_gtoc9_debris())}
    events = parse_gtoc9_submission(path)
    violations: list[str] = []
    n_lines = len(events)
    if n_lines > c["max_lines"] or n_lines < 2:
        violations.append(f"rule 3: {n_lines} lines outside [2, {c['max_lines']}]")
    ve = c["isp_s"] * c["g0_m_s2"]
    ids = [event.event_id for event in events]
    for k, event in enumerate(events):
        if not -1 <= event.event_id <= 122:
            violations.append(f"rule 4: line {k} id {event.event_id}")
        if osculating_pericentre(event.r, event.v) <= c["r_p_min_m"]:
            violations.append(f"rule 5: line {k} pericentre below {c['r_p_min_m']} m")
        if not (c["window_start_mjd2000"] <= event.t <= c["window_end_mjd2000"]):
            violations.append(f"rule 19: line {k} epoch {event.t} outside window")
    distinct = sorted({i for i in ids if i != -1})
    N = len(distinct)
    m0 = events[0].m
    mf = events[-1].m
    propellant = m0 - c["m_dry_kg"] - c["m_de_kg"] * N
    if m0 < c["m_dry_kg"] + c["m_de_kg"]:
        violations.append("rule 6: initial mass below 2030 kg")
    if propellant > c["m_p_max_kg"]:
        violations.append(f"rule 6: propellant {propellant:.3f} kg exceeds 5000 kg")
    if mf < c["m_dry_kg"]:
        violations.append("rule 6: final mass below dry mass")
    for k in range(1, n_lines):
        if events[k].t <= events[k - 1].t:
            violations.append(f"rule 7: epochs not increasing at line {k}")
    if np.linalg.norm(events[0].dv) != 0.0 or np.linalg.norm(events[-1].dv) != 0.0:
        violations.append("rule 8: first/last velocity increments must be zero")
    if not (ids[0] == ids[1] != -1 and ids[-1] == ids[-2] != -1):
        violations.append("rule 9: first/last event pairs must be arrival/departure at one debris")
    for k in range(2, n_lines - 2):
        if ids[k] != -1 and not (ids[k - 1] == ids[k] or ids[k + 1] == ids[k]):
            violations.append(f"rule 10: isolated debris event at line {k}")
    for debris_id in distinct:
        if ids.count(debris_id) != 2:
            violations.append(f"rule 11: debris {debris_id} has {ids.count(debris_id)} events")
    max_rdv_r = max_rdv_v = max_prop_r = max_prop_v = 0.0
    arrival_lines: list[int] = []
    seen: set[int] = set()
    for k, event in enumerate(events):
        if event.event_id == -1:
            continue
        if event.event_id not in seen:
            seen.add(event.event_id)
            arrival_lines.append(k)
    arrival_set = set(arrival_lines)
    for k in arrival_lines:
        event = events[k]
        r_d, v_d = gtoc9_debris_state(debris_table[event.event_id], event.t)
        err_r = float(np.linalg.norm(event.r - r_d))
        err_v = float(np.linalg.norm(event.v - v_d + event.dv))
        max_rdv_r = max(max_rdv_r, err_r)
        max_rdv_v = max(max_rdv_v, err_v)
        if err_r >= c["eps_r_m"] or err_v >= c["eps_v_m_s"]:
            violations.append(f"rule 12: arrival rendezvous error at line {k}")
        if k + 1 < n_lines and events[k + 1].t - event.t < c["t_w_days"]:
            violations.append(f"rule 14: waiting time below 5 days at line {k}")
    for index in range(1, len(arrival_lines)):
        if (
            events[arrival_lines[index]].t - events[arrival_lines[index - 1]].t
            > c["delta_t_R_days"]
        ):
            violations.append(
                f"rule 15: more than 30 days between rendezvous at line {arrival_lines[index]}"
            )
    for k in range(1, n_lines):
        event = events[k]
        previous = events[k - 1]
        expected_mass = previous.m * math.exp(-float(np.linalg.norm(previous.dv)) / ve)
        is_departure = event.event_id != -1 and k not in arrival_set
        if is_departure:
            if abs(event.m - expected_mass + c["m_de_kg"]) > c["eps_m_kg"]:
                violations.append(f"rule 17: departure mass update at line {k}")
            r_d, v_d = gtoc9_debris_state(debris_table[event.event_id], event.t)
            err_r = float(np.linalg.norm(event.r - r_d))
            err_v = float(np.linalg.norm(event.v - v_d))
            max_rdv_r = max(max_rdv_r, err_r)
            max_rdv_v = max(max_rdv_v, err_v)
            if err_r >= c["eps_r_m"] or err_v >= c["eps_v_m_s"]:
                violations.append(f"rule 16: departure rendezvous error at line {k}")
        else:
            if abs(event.m - expected_mass) > c["eps_m_kg"]:
                violations.append(f"rule 13: mass update at line {k}")
            state = np.concatenate([previous.r, previous.v + previous.dv])
            propagated = propagate_j2(state, (event.t - previous.t) * c["day_s"])
            err_r = float(np.linalg.norm(propagated[0:3] - event.r))
            err_v = float(np.linalg.norm(propagated[3:6] - event.v))
            max_prop_r = max(max_prop_r, err_r)
            max_prop_v = max(max_prop_v, err_v)
            if err_r >= c["eps_r_m"] or err_v >= c["eps_v_m_s"]:
                violations.append(f"rule 18: propagation mismatch at line {k}")
    penalty = c["alpha_meur_per_kg2"] * (m0 - c["m_dry_kg"]) ** 2
    return GTOC9MissionEvaluation(
        valid=not violations,
        violations=violations,
        debris_removed=distinct,
        initial_mass_kg=m0,
        final_mass_kg=mf,
        propellant_mass_kg=propellant,
        mass_penalty_meur=penalty,
        mission_cost_min_meur=c["c_m_meur"] + penalty,
        mission_cost_max_meur=c["c_M_meur"] + penalty,
        max_rendezvous_position_error_m=max_rdv_r,
        max_rendezvous_velocity_error_m_s=max_rdv_v,
        max_propagation_position_error_m=max_prop_r,
        max_propagation_velocity_error_m_s=max_prop_v,
        lines=n_lines,
    )


# --------------------------------------------------------------------------- GTOC12
GTOC12_VERIFIER_SHA256 = "d4e4bc81129266420b27c9bde038bce9eda1960e7de9c695772fbfdb1cc82cd6"
GTOC12_RESULT_EXAMPLE_SHA256 = "6352147cfbc70def3563eb61804ad13e64891bc3537fbb76de50169c39d07ab5"


def extract_gtoc12_verifier() -> tuple[Path, Path, Path]:
    """Extract the Linux verifier, asteroid data, and bundled Result.txt from the pinned zip."""

    archive = external_sources.fetch("gtoc12.verification_program")
    root = external_sources.cache_root() / "gtoc12" / "verifier"
    root.mkdir(parents=True, exist_ok=True)
    names = {
        "binary": "GTOC12_Verification/Linux/GTOC12_Verify",
        "data": "GTOC12_Verification/Linux/GTOC12_Asteroids_Data.txt",
        "example": "GTOC12_Verification/Linux/Result.txt",
        "readme": "GTOC12_Verification/README.md",
    }
    with zipfile.ZipFile(archive) as handle:
        for member in names.values():
            if not (root / member).is_file():
                handle.extract(member, root)
    binary = root / names["binary"]
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    digest = external_sources.sha256_of(binary)
    if digest != GTOC12_VERIFIER_SHA256:
        raise external_sources.ChecksumMismatch(f"GTOC12 verifier sha256 {digest} unexpected")
    return binary, root / names["data"], root / names["example"]


@dataclass(slots=True)
class GTOC12Verification:
    solution: str
    accepted: bool
    ships: int | None
    mined_asteroids: int | None
    total_resource_mass_kg: float | None
    stdout: str
    wall_seconds: float
    score_data_rows: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_gtoc12_verifier(solution: Path, *, timeout_s: float = 900.0) -> GTOC12Verification:
    import time

    binary, data, _ = extract_gtoc12_verifier()
    if os.name != "posix":
        raise RuntimeError("the pinned GTOC12 verifier binary is Linux x86-64 only")
    with tempfile.TemporaryDirectory(prefix="gtoc12-verify-") as tmp:
        workdir = Path(tmp)
        shutil.copy(binary, workdir / "GTOC12_Verify")
        shutil.copy(data, workdir / "GTOC12_Asteroids_Data.txt")
        shutil.copy(solution, workdir / "Result.txt")
        start = time.perf_counter()
        completed = subprocess.run(
            ["./GTOC12_Verify"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        wall = time.perf_counter() - start
        stdout = completed.stdout + completed.stderr
        score_rows = None
        score_file = workdir / "ScoreData.txt"
        if score_file.is_file():
            lines = [line for line in score_file.read_text().splitlines() if line.strip()]
            score_rows = max(len(lines) - 1, 0)
    accepted = "Check successfully" in stdout
    ships = mined = None
    mass = None
    for line in stdout.splitlines():
        if "mining ships" in line:
            ships = int(line.split("is")[1].split(";")[0])
        elif "mined asteroids" in line:
            mined = int(line.split("is")[1].split(";")[0])
        elif "total resource mass" in line:
            mass = float(line.split("is")[1].split("kg")[0])
    return GTOC12Verification(
        solution=solution.name,
        accepted=accepted,
        ships=ships,
        mined_asteroids=mined,
        total_resource_mass_kg=mass,
        stdout=stdout.strip(),
        wall_seconds=wall,
        score_data_rows=score_rows,
    )


@dataclass(frozen=True, slots=True)
class Asteroid12:
    id: int
    epoch_mjd: float
    a_au: float
    e: float
    i_deg: float
    raan_deg: float
    argp_deg: float
    m_deg: float


def load_gtoc12_asteroids(path: Path | None = None) -> list[Asteroid12]:
    path = path or external_sources.fetch("gtoc12.asteroids")
    rows: list[Asteroid12] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 8 or not parts[0].isdigit():
            continue
        rows.append(
            Asteroid12(
                id=int(parts[0]),
                epoch_mjd=float(parts[1]),
                a_au=float(parts[2]),
                e=float(parts[3]),
                i_deg=float(parts[4]),
                raan_deg=float(parts[5]),
                argp_deg=float(parts[6]),
                m_deg=float(parts[7]),
            )
        )
    return rows


# --------------------------------------------------------------------------- GTOC5
@dataclass(frozen=True, slots=True)
class Asteroid5:
    id: int
    name: str
    epoch_mjd: float
    a_au: float
    e: float
    i_deg: float
    argp_deg: float
    node_deg: float
    m_deg: float


def load_gtoc5_asteroids(path: Path | None = None) -> list[Asteroid5]:
    """Rows of ``gtoc5_problem_data.txt``; ``id`` is the 1-based row order (the file's
    designation column mixes numbered asteroids and provisional designations)."""

    path = path or external_sources.fetch("gtoc5.problem_data")
    rows: list[Asteroid5] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        try:
            epoch = float(parts[0])
        except ValueError:
            continue
        rows.append(
            Asteroid5(
                id=len(rows) + 1,
                name=" ".join(parts[7:]),
                epoch_mjd=epoch,
                a_au=float(parts[1]),
                e=float(parts[2]),
                i_deg=float(parts[3]),
                argp_deg=float(parts[4]),
                node_deg=float(parts[5]),
                m_deg=float(parts[6]),
            )
        )
    return rows


# --------------------------------------------------------------------------- reduced subsets
def gtoc9_reduced_subset(debris: list[Debris], *, count: int = 15) -> dict[str, Any]:
    """Metadata rule: the ``count`` debris whose J2-propagated RAAN at the window start is
    closest to the population median RAAN (transfers within a RAAN cluster are cheapest)."""

    t0 = GTOC9_CONSTANTS["window_start_mjd2000"]
    raans = {}
    for item in debris:
        c = GTOC9_CONSTANTS
        n = math.sqrt(c["mu_m3_s2"] / item.a**3)
        p = item.a * (1.0 - item.e**2)
        raan_dot = -1.5 * c["j2"] * (c["r_eq_m"] / p) ** 2 * n * math.cos(item.i)
        raans[item.id] = (item.raan0 + raan_dot * (t0 - item.t0) * c["day_s"]) % (2.0 * math.pi)
    median = float(np.median(list(raans.values())))

    def distance(value: float) -> float:
        d = abs(value - median) % (2.0 * math.pi)
        return min(d, 2.0 * math.pi - d)

    ordered = sorted(raans, key=lambda i: (distance(raans[i]), i))
    return {
        "rule": (
            "count debris with RAAN (J2-propagated to MJD2000 23467) closest to the "
            "population median RAAN; ties by id"
        ),
        "count": count,
        "median_raan_rad": median,
        "debris_ids": sorted(ordered[:count]),
    }


def gtoc12_reduced_subset(asteroids: list[Asteroid12], *, count: int = 500) -> dict[str, Any]:
    """Metadata rule: low-inclination, low-eccentricity main-belt asteroids, first ``count`` ids."""

    selected = [
        item.id
        for item in asteroids
        if item.i_deg < 3.0 and item.e < 0.15 and 2.5 <= item.a_au <= 3.0
    ]
    return {
        "rule": "i < 3 deg, e < 0.15, 2.5 AU <= a <= 3.0 AU; first count ids ascending",
        "count": count,
        "candidates_matching_rule": len(selected),
        "asteroid_ids": sorted(selected)[:count],
    }


def gtoc5_reduced_subset(asteroids: list[Asteroid5], *, count: int = 200) -> dict[str, Any]:
    """Metadata rule: Earth-like orbits (a in [0.9, 1.5] AU, e < 0.2, i < 5 deg) plus Beletskij."""

    beletskij = [item.id for item in asteroids if "Beletskij" in item.name]
    selected = sorted(
        item.id
        for item in asteroids
        if 0.9 <= item.a_au <= 1.5 and item.e < 0.2 and item.i_deg < 5.0
    )[:count]
    return {
        "rule": (
            "0.9 AU <= a <= 1.5 AU, e < 0.2, i < 5 deg; first count ids ascending; "
            "Beletskij always included"
        ),
        "count": count,
        "asteroid_ids": sorted(set(selected) | set(beletskij)),
        "beletskij_id": beletskij[0] if beletskij else None,
    }


def build_reduced_subsets() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "selection_policy": (
            "frozen from problem metadata before any solver score is observed "
            "(docs/COMPARATIVE_SOLVER_CAMPAIGN.md, Layer E)"
        ),
        "gtoc9": gtoc9_reduced_subset(load_gtoc9_debris()),
        "gtoc12": gtoc12_reduced_subset(load_gtoc12_asteroids()),
        "gtoc5": gtoc5_reduced_subset(load_gtoc5_asteroids()),
    }


# --------------------------------------------------------------------------- target runners
def _blocked(document: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "target_id": document["id"],
        "status": "blocked",
        "published": {},
        "measured": {},
        "gap": {},
        "labels": {},
        "envelope": {},
        "commands": [f"spacepdhcg literature run {document['id']}"],
        "notes": [f"blocked: {reason}"],
    }


def run_gtoc12_target(document: dict[str, Any], *, options: dict[str, Any]) -> dict[str, Any]:
    try:
        _, _, example = extract_gtoc12_verifier()
        results = {"bundled_result_example": run_gtoc12_verifier(example).as_dict()}
        for artifact_id in document.get("published_solutions", []):
            path = external_sources.fetch(artifact_id)
            results[artifact_id] = run_gtoc12_verifier(path).as_dict()
    except external_sources.ArtifactUnavailable as error:
        return _blocked(document, str(error))
    accepted = all(item["accepted"] for item in results.values())
    return {
        "target_id": document["id"],
        "status": "reproduced" if accepted else "gap",
        "published": document.get("published", {}),
        "measured": {
            key: {
                "accepted": item["accepted"],
                "ships": item["ships"],
                "mined_asteroids": item["mined_asteroids"],
                "total_resource_mass_kg": item["total_resource_mass_kg"],
            }
            for key, item in results.items()
        },
        "gap": {},
        "labels": {f"measured.{key}": "reproduced-external" for key in results},
        "envelope": {"evaluator": "official GTOC12 verification program (Linux binary, pinned)"},
        "commands": [f"spacepdhcg literature run {document['id']}"],
        "notes": ["no route search run; official verifier accepted every file"],
        "details": results,
    }


def run_gtoc9_target(document: dict[str, Any], *, options: dict[str, Any]) -> dict[str, Any]:
    try:
        debris = load_gtoc9_debris()
        evaluations = {
            "example1": evaluate_gtoc9_mission(
                external_sources.fetch("gtoc9.example1"), debris
            ).as_dict(),
            "example2": evaluate_gtoc9_mission(
                external_sources.fetch("gtoc9.example2"), debris
            ).as_dict(),
        }
    except external_sources.ArtifactUnavailable as error:
        return _blocked(document, str(error))
    all_valid = all(item["valid"] for item in evaluations.values())
    return {
        "target_id": document["id"],
        "status": "reproduced" if all_valid else "gap",
        "published": document.get("published", {}),
        "measured": {
            key: {
                "valid": item["valid"],
                "debris_removed": item["debris_removed"],
                "mission_cost_min_meur": item["mission_cost_min_meur"],
                "mission_cost_max_meur": item["mission_cost_max_meur"],
            }
            for key, item in evaluations.items()
        },
        "gap": {},
        "labels": {f"measured.{key}": "measured-local" for key in evaluations},
        "envelope": {
            "evaluator": (
                "local implementation of Kelvins rules 4-19 (no official offline evaluator exists)"
            ),
            "propagator": "scipy DOP853 rtol 1e-12 on the official J2 equations",
        },
        "commands": [f"spacepdhcg literature run {document['id']}"],
        "notes": [
            "the official GTOC9 scoring ran on the Kelvins server; both official example "
            "submissions must validate under the re-implemented rules",
        ],
        "details": evaluations,
    }


def run_gtoc5_target(document: dict[str, Any], *, options: dict[str, Any]) -> dict[str, Any]:
    try:
        asteroids = load_gtoc5_asteroids()
    except external_sources.ArtifactUnavailable as error:
        return _blocked(document, str(error))
    return {
        "target_id": document["id"],
        "status": "blocked",
        "published": document.get("published", {}),
        "measured": {"asteroid_count": len(asteroids)},
        "gap": {},
        "labels": {"measured.asteroid_count": "measured-local"},
        "envelope": {},
        "commands": [f"spacepdhcg literature run {document['id']}"],
        "notes": [
            "blocked: no official offline GTOC5 evaluator or example solution file is published; "
            "the pinned Simoes et al. beam P-ACO implementation requires pykep, which is not "
            "installed in the campaign environment",
        ],
        "details": {"asteroid_count": len(asteroids)},
    }
