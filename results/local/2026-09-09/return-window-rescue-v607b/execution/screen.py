"""Bounded paired GPU Lambert screening; deterministic approximate ranking only."""

# Frozen source activation must precede every production import.
# ruff: noqa: E402

import math

import numpy as np
import support

support.activate()
from spacepdhcg.gtoc12.gpu_lambert import GpuLambert
from spacepdhcg.gtoc12.screening import (
    propellant_for_delta_v,
    return_inflation_model,
    thrust_authority_km_s,
)


def coarse_windows():
    waits = sorted(set(range(0, 589, 5)) | {588})
    arrivals = sorted(set(range(5, 590, 5)) | {589})
    return [
        (69218.0 + wait, 69218.0 + arrival)
        for wait in waits
        for arrival in arrivals
        if arrival > wait
    ]


def fine_windows(centres, already):
    if len(centres) > 32:
        raise ValueError("at most 32 fine-grid centres")
    points = set()
    for centre in centres:
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                pair = (centre["departure"] + dx, centre["arrival"] + dy)
                if 69218 <= pair[0] < pair[1] <= 69807 and pair not in already:
                    points.add(pair)
    if len(points) > 2592:
        raise AssertionError("fine-grid budget exceeded")
    return sorted(points)


def rank_key(row):
    return (
        row["predicted_propellant_kg"],
        row["authority_ratio"],
        -row["tof_days"],
        row["departure"],
        row["arrival"],
    )


def select_basins(rows, count, separation=20):
    if not 0 <= count <= 32:
        raise ValueError("bounded basin count required")
    valid = sorted(
        (
            row
            for row in rows
            if row["geometry_feasible"]
            and all(
                math.isfinite(row[key])
                for key in ("predicted_propellant_kg", "authority_ratio", "departure", "arrival")
            )
        ),
        key=rank_key,
    )
    selected, used = [], set()
    for diverse in (True, False):
        for row in valid:
            pair = (row["departure"], row["arrival"])
            if pair in used:
                continue
            if diverse and any(
                max(abs(pair[0] - chosen["departure"]), abs(pair[1] - chosen["arrival"]))
                < separation
                for chosen in selected
            ):
                continue
            if len(selected) >= count:
                return selected
            selected.append(row)
            used.add(pair)
    return selected


def evaluate(engine, catalogue, windows, mass, minimum, directory, stage):
    points = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return []
    hop = engine.paired_hops(catalogue, 59653, 0, points[:, 0], points[:, 1] - points[:, 0])
    dv = hop.total_delta_v
    authority = thrust_authority_km_s(mass, hop.tof_days, 1.0)
    ratio = dv / np.maximum(authority, 1e-12)
    inflation = return_inflation_model(hop.tof_days, ratio)
    propellant = propellant_for_delta_v(mass, inflation * dv)
    feasible = (
        hop.feasible & np.isfinite(dv) & np.isfinite(propellant) & (dv >= 0) & (propellant >= 0)
    )
    raw = directory / (stage + ".npz")
    np.savez_compressed(
        raw,
        departure=points[:, 0],
        arrival=points[:, 1],
        tof_days=hop.tof_days,
        departure_delta_v=hop.departure_delta_v,
        arrival_delta_v=hop.arrival_delta_v,
        departure_velocity=hop.departure_velocity,
        arrival_velocity=hop.arrival_velocity,
        geometry_feasible=hop.feasible,
        ranking_valid=feasible,
        authority_ratio=ratio,
        inflation=inflation,
        predicted_propellant_kg=propellant,
    )
    rows = [
        {
            "stage": stage,
            "row": index,
            "departure": float(pair[0]),
            "arrival": float(pair[1]),
            "tof_days": float(pair[1] - pair[0]),
            "lambert_delta_v_km_s": float(dv[index]),
            "authority_ratio": float(ratio[index]),
            "inflation": float(inflation[index]),
            "predicted_propellant_kg": float(propellant[index]),
            "predicted_reserve_kg": float(mass - minimum - propellant[index]),
            "geometry_feasible": bool(feasible[index]),
            "scope": "uncertified_ranking_only",
        }
        for index, pair in enumerate(points)
    ]
    support.write(
        directory / (stage + ".json"),
        {"rows": rows, "raw_sha256": support.sha(raw), "raw": raw.name},
    )
    return rows


def screen_windows(prefix, catalogue, directory, *, engine_factory=GpuLambert):
    directory.mkdir(parents=True, exist_ok=False)
    coarse = coarse_windows()
    if len(coarse) != 7022:
        raise AssertionError("reviewed coarse grid changed")
    minimum = 500 + sum(prefix.cargo.values())
    with engine_factory(maximum_batch_size=16384) as engine:
        first = evaluate(engine, catalogue, coarse, prefix.after_mass, minimum, directory, "coarse")
        centres = select_basins(first, 32)
        fine = fine_windows(centres, set(coarse))
        second = evaluate(engine, catalogue, fine, prefix.after_mass, minimum, directory, "fine")
        chosen = select_basins(first + second, 4)
        telemetry = dict(engine.telemetry)
    total = len(first) + len(second)
    if total > 9614 or len(chosen) > 4:
        raise AssertionError("reviewed screening/refinement budget exceeded")
    result = {
        "coarse_rows": len(first),
        "fine_rows": len(second),
        "total_rows": total,
        "logical_branch_requests": 2 * total,
        "fine_centres": centres,
        "selected": chosen,
        "native_telemetry": telemetry,
        "scope": "GPU geometry with host ranking; no low-thrust certificate",
    }
    support.write(directory / "report.json", result)
    return result
