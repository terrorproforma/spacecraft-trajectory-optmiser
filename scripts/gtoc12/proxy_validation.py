"""Proxy-vs-truth validation for GTOC12 hop screening.

Two data sets:

1. Our certified legs (``results/gtoc12/runs/**/refinements.json``): the search's inflated
   zero-revolution Lambert propellant proxy against the refined, certified SCvx propellant.
2. The archived reference hops (``results/gtoc12/references/*.itinerary.json``): true low-thrust
   ΔV (from the mass drop) against zero-rev Lambert and the Lambert-free phasing/Edelbaum proxy.

Writes ``results/gtoc12/proxy_validation.json`` and prints the error distributions.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spacepdhcg.gtoc12 import constants as C  # noqa: E402
from spacepdhcg.gtoc12.data import load_catalogue  # noqa: E402
from spacepdhcg.gtoc12.ephemeris import asteroid_state  # noqa: E402
from spacepdhcg.gtoc12.proxies import phasing_edelbaum_proxy  # noqa: E402
from spacepdhcg.gtoc12.screening import lambert_hops, propellant_for_delta_v  # noqa: E402

EXHAUST = C.ISP_S * C.G0_M_S2 * 1e-3


def percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)])
    if arr.size == 0:
        return {}
    keys = [5, 10, 25, 50, 75, 90, 95]
    out = {f"p{k}": float(np.percentile(arr, k)) for k in keys}
    out["mean"] = float(arr.mean())
    out["count"] = int(arr.size)
    return out


def certified_legs() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "results/gtoc12/runs").glob("**/refinements.json")):
        run_name = path.relative_to(ROOT / "results/gtoc12/runs").parts[0]
        for entry in json.loads(path.read_text()):
            if "plan" not in entry:  # skipped candidate (contained an already-failed leg)
                continue
            plan_legs = {
                (leg["from"], leg["to"], round(leg["t0"], 3)): leg
                for leg in entry["plan"]["legs"]
                if leg["role"] != "camp"
            }
            for leg in entry["refined"]["legs"]:
                if not leg["certified"] or leg["propellant_kg"] is None:
                    continue
                planned = plan_legs.get((leg["from"], leg["to"], round(leg["t0"], 3)))
                if planned is None:
                    continue
                inflation = 1.6 if 0 in (leg["from"], leg["to"]) else 1.2
                proxy_prop = float(
                    propellant_for_delta_v(leg["mass_before"], planned["dv_proxy_km_s"] * inflation)
                )
                rows.append(
                    {
                        "run": run_name,
                        "from": leg["from"],
                        "to": leg["to"],
                        "role": planned["role"],
                        "tof_days": leg["tf"] - leg["t0"],
                        "mass_before": leg["mass_before"],
                        "dv_lambert_km_s": planned["dv_proxy_km_s"],
                        "dv_refined_km_s": leg["delta_v_km_s"],
                        "propellant_proxy_kg": proxy_prop,
                        "propellant_refined_kg": leg["propellant_kg"],
                    }
                )
    return rows


def reference_hops(catalogue) -> list[dict]:
    rows = []
    for path in sorted((ROOT / "results/gtoc12/references").glob("*.itinerary.json")):
        data = json.loads(path.read_text())
        for ship in data["ships"]:
            for leg in ship["legs"]:
                if leg["role"] not in ("deploy_hop", "collect_hop"):
                    continue
                s, t = leg["from_body"], leg["to_body"]
                if s <= 0 or t <= 0 or s == t:
                    continue
                t0, tof = leg["departure_epoch"], leg["tof_days"]
                m0, dm = leg["mass_departure"], leg["propellant_kg"]
                if dm <= 0.0 or tof <= 0.0:
                    continue
                dv_true = EXHAUST * math.log(m0 / (m0 - dm))
                rs, vs = asteroid_state(catalogue, s, t0)
                rt, vt = asteroid_state(catalogue, t, t0 + tof)
                hop = lambert_hops(
                    rs[None], vs[None], rt[None], vt[None], np.array([t0]), np.array([tof])
                )
                dv_lambert = float(hop.total_delta_v[0]) if hop.feasible[0] else float("nan")
                proxy = phasing_edelbaum_proxy(catalogue, s, np.array([t]), t0, np.array([tof]))
                rows.append(
                    {
                        "file": path.name.split(".")[0],
                        "tof_days": tof,
                        "mass_departure": m0,
                        "dv_true_km_s": dv_true,
                        "dv_lambert_km_s": dv_lambert,
                        "dv_edelbaum_phasing_km_s": float(proxy["delta_v"][0, 0]),
                    }
                )
    return rows


def main() -> None:
    catalogue = load_catalogue()
    legs = certified_legs()
    refs = reference_hops(catalogue)
    ours = {
        "legs": len(legs),
        "propellant_ratio_refined_over_proxy": percentiles(
            [
                r["propellant_refined_kg"] / r["propellant_proxy_kg"]
                for r in legs
                if r["propellant_proxy_kg"] > 0
            ]
        ),
        "propellant_error_kg_refined_minus_proxy": percentiles(
            [r["propellant_refined_kg"] - r["propellant_proxy_kg"] for r in legs]
        ),
        "dv_ratio_refined_over_lambert": percentiles(
            [
                r["dv_refined_km_s"] / r["dv_lambert_km_s"]
                for r in legs
                if r["dv_lambert_km_s"] > 0.05
            ]
        ),
        "by_role": {
            role: percentiles(
                [
                    r["dv_refined_km_s"] / r["dv_lambert_km_s"]
                    for r in legs
                    if r["role"] == role and r["dv_lambert_km_s"] > 0.05
                ]
            )
            for role in ("earth_out", "deploy_hop", "collect_hop", "earth_return")
        },
    }
    lam = [r["dv_true_km_s"] / r["dv_lambert_km_s"] for r in refs if r["dv_lambert_km_s"] > 0.05]
    edl = [
        r["dv_true_km_s"] / r["dv_edelbaum_phasing_km_s"]
        for r in refs
        if r["dv_edelbaum_phasing_km_s"] > 0.05
    ]
    lam_abs = [r["dv_true_km_s"] - r["dv_lambert_km_s"] for r in refs]
    edl_abs = [r["dv_true_km_s"] - r["dv_edelbaum_phasing_km_s"] for r in refs]
    true_dv = np.array([r["dv_true_km_s"] for r in refs])
    lam_dv = np.array([r["dv_lambert_km_s"] for r in refs])
    edl_dv = np.array([r["dv_edelbaum_phasing_km_s"] for r in refs])
    ok = np.isfinite(lam_dv)
    references = {
        "hops": len(refs),
        "true_over_lambert_ratio": percentiles(lam),
        "true_minus_lambert_km_s": percentiles(lam_abs),
        "true_over_edelbaum_phasing_ratio": percentiles(edl),
        "true_minus_edelbaum_phasing_km_s": percentiles(edl_abs),
        "spearman_true_vs_lambert": float(_spearman(true_dv[ok], lam_dv[ok])),
        "spearman_true_vs_edelbaum_phasing": float(_spearman(true_dv, edl_dv)),
        "authority_fraction_true_dv_over_full_thrust": percentiles(
            [
                r["dv_true_km_s"]
                / (C.THRUST_MAX_N / r["mass_departure"] * 1e-3 * r["tof_days"] * C.DAY_S)
                for r in refs
            ]
        ),
    }
    out = {"certified_legs": ours, "reference_hops": references}
    (ROOT / "results/gtoc12/proxy_validation.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=1, sort_keys=True))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
