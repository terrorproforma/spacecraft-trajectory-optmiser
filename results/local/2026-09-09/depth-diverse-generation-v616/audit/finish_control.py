"""Two scalar completion checks on saved epochs/DVs; no generation or native work."""

import ctypes
import dataclasses
import json
import math
import os
import sys
from unittest.mock import patch

from audit import KIT, ROOT, read, sha


def main():
    ready = read(KIT / "ready-manifest.json")
    for name, digest in ready["files"].items():
        assert sha(KIT / name) == digest
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(KIT / "source/src"))
    os.environ["SPACEPDHCG_GTOC12_DATA"] = read(KIT / "profile.json")["data"]
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("No native loading")):
        import numpy as np

        from spacepdhcg.gtoc12 import constants as C
        from spacepdhcg.gtoc12.data import load_catalogue
        from spacepdhcg.gtoc12.search import RoutePlan, RouteSearch, SearchSettings, _Partial

        defaults = {field.name: field.default for field in dataclasses.fields(SearchSettings)}
        values = read(KIT / "output/report.json")["settings"]
        values = {
            key: (
                defaults[key]
                if value is None
                and isinstance(defaults[key], float)
                and not math.isfinite(defaults[key])
                else value
            )
            for key, value in values.items()
        }
        settings = SearchSettings(**values)
        inputs = read(KIT / "generation-input.json")
        catalogue = load_catalogue()
        original = read(KIT / "inputs/ship-23.json")
        plan = RoutePlan.from_summary(original["plan"])
        split = max(i for i, leg in enumerate(plan.legs) if leg.role == "deploy_hop") + 1
        last = plan.legs[split - 1]
        actual = next(
            leg
            for leg in original["legs"]
            if (leg["from"], leg["to"], leg["t0"], leg["tf"])
            == (last.from_id, last.to_id, last.departure_epoch, last.arrival_epoch)
        )
        diagnosis = read(ROOT / "diagnosis-v2.json")
        proxy_mass = diagnosis["incumbent_saved_dv_scalar_proxy_deploy_checks"][-1][
            "sequential_proxy_mass_after_deploy_kg"
        ]
        measured_mass = actual["mass_after"] - C.MINER_MASS_KG
        rows = []
        for name, mass in [
            ("current_flat_deploy_proxy", proxy_mass),
            ("measured_deploy_prefix_mass_control", measured_mass),
        ]:
            search = RouteSearch(catalogue, np.asarray(inputs["allowed_ids"]), settings)
            partial = _Partial(
                list(plan.legs[:split]),
                last.to_id,
                last.arrival_epoch,
                mass,
                sorted(plan.deploy_epochs.items(), key=lambda x: (x[1], x[0])),
            )
            mass_guess = mass + sum(plan.collected_mass.values())
            forward = [
                dataclasses.replace(
                    leg,
                    inflation=search.collect_table.return_inflation_at(
                        leg.from_id,
                        leg.departure_epoch,
                        leg.tof_days,
                        leg.delta_v_proxy_km_s,
                        mass_guess,
                    ),
                )
                if leg.role == "earth_return"
                else leg
                for leg in plan.legs[split:]
            ]
            result = search._finish(
                partial, dict(plan.deploy_epochs), dict(plan.collect_epochs), forward
            )
            rows.append(
                {
                    "control": name,
                    "mass_after_deploys_kg": mass,
                    "current_completion_return_inflation": forward[-1].inflation,
                    "accepted_by_frozen_scalar_finish": result is not None,
                    "failure": search.last_failure if result is None else None,
                    "cargo_preserved_kg": sum(plan.collected_mass.values()),
                    "final_cargo_margin_kg": None
                    if result is None
                    else result.final_mass_proxy_kg
                    - C.DRY_MASS_KG
                    - sum(result.collected_mass.values()),
                    "no_physical_certificate": True,
                }
            )
            if result is not None:
                assert result.collected_mass == plan.collected_mass
                assert (
                    result.deploy_epochs == plan.deploy_epochs
                    and result.collect_epochs == plan.collect_epochs
                )
        report = {
            "GPU_calls": 0,
            "fresh_lambert_evaluations": 0,
            "generations": 0,
            "refinements": 0,
            "scalar_finish_checks": 2,
            "controls": rows,
            "measured_minus_proxy_deploy_mass_kg": measured_mass - proxy_mass,
            "scope": (
                "Exact frozen _finish on saved incumbent epochs and Lambert DVs; this bypasses "
                "grid generation for diagnosis only and emits no candidate or fleet"
            ),
        }
        with (ROOT / "finish-control.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
