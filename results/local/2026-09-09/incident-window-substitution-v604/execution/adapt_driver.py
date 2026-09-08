"""Construct the bounded v604 driver from its exact retained v599 predecessor."""

from pathlib import Path

root = Path(__file__).resolve().parent
source = root / "reference/v599-run.py"
text = source.read_text()


def replace(old, new):
    global text
    if old not in text:
        raise ValueError(f"missing original driver fragment: {old[:100]}")
    text = text.replace(old, new)


replace("v599", "v604")
replace("from pathlib import Path\n", "from pathlib import Path\n\nimport incident\n")
start = text.index("def epoch_seeds(")
end = text.index("\ndef objective_gate(", start)
text = text[:start] + "epoch_seeds = incident.epoch_seeds\n\n" + text[end:]
start = text.index("def make_plan(")
end = text.index("\ndef arguments(", start)
text = (
    text[:start]
    + """def make_plan(summaries, audit, weights, catalogue, solution):
    from spacepdhcg.gtoc12.jointopt import JointItinerary, route_from_summary
    from spacepdhcg.gtoc12.retiming import visits_of

    inventory(summaries, audit, weights, solution)
    reference = ROOT / "reference/v599-plan.json"
    if sha256(reference) != "c58c60f081b389d37a95bdec79cf406ec98a363197d1ba1d3de2cc459675dff2":
        raise ValueError("exact 496-case v599 inventory required")
    history = ROOT / "reference/v599-report.json"
    if sha256(history) != "0f8e3d2078ad001c446ca80491cfd506bbcb261d7261a21e47e583370cee29d2":
        raise ValueError("historical control changed")
    plan = json.loads(reference.read_text())
    if catalogue.source_sha256 != DATA_SHA:
        raise ValueError("catalogue pin mismatch")
    cases = plan["replacement_cases"]
    occupied = set(plan["occupied_asteroids"])
    count, structures = 0, []
    max_polish = 0
    for selected in plan["selected_ships"]:
        route = route_from_summary(summaries[selected["ship"]])
        visits, arr, dep = visits_of(route.plan)
        max_polish = max(max_polish, 1 + 4 * len(list(JointItinerary.moves(len(visits), 8))))
        for case in cases:
            if case["ship"] != selected["ship"]:
                continue
            new_visits = replacement_visits(visits, case["old"], case["new"], occupied)
            seeds = epoch_seeds(new_visits, arr, dep, case["new"])
            indices = incident.incident_indices(new_visits, case["new"])
            count += len(seeds)
            structures.append({"case": case["case"], "seeds": len(seeds), "incident_legs": indices})
    if len(cases) != 496 or count > 12400 or 4 * max_polish > incident.MAX_POLISH_ROWS:
        raise ValueError("prepared search exceeds approved candidate budget")
    controls = [r["case"] for r in json.loads(history.read_text())["retimings"]]
    incident.choose_arms(cases, {}, controls)
    plan.update(
        kind="CPU_prepared_controlled_incident_window_search_not_physics_certificates",
        reference_case_plan_sha256=sha256(reference),
        reference_run_report_sha256=sha256(history),
        epoch_shift_days=list(incident.SHIFTS_DAYS),
        construction=structures,
        control_case_ids=controls,
        maximum_initial_epoch_candidates=count,
        maximum_total_joint_candidates=count + 4 * max_polish,
        maximum_retime_calls=24,
        maximum_retime_calls_per_arm=12,
        maximum_polish_seeds=4,
        maximum_full_refinements=4,
        comparison_scope=("Shared expanded grid; twelve historical-priority versus twelve "
                          "complete-incidence choices, fresh equivalent retimer states. "
                          "Grid expansion versus v599 is a separate changed budget."),
    )
    return plan

"""
    + text[end:]
)
replace(
    '"maximum_joint_candidates": 6000,',
    '"maximum_joint_candidates": plan["maximum_total_joint_candidates"],',
)
replace(
    '"gpu_neighbor_queries": [],',
    """"gpu_neighbor_queries": [],
        "controlled_comparison": {
            "shared_grid_rows_planned": plan["maximum_initial_epoch_candidates"],
            "retimings_per_arm": 12,
            "maximum_refinements_total": 4,
            "arms": {},
            "scope": plan["comparison_scope"],
        },
        "ranking_module_sha256": sha256(ROOT / "incident.py"),""",
)
replace("screened_best = {}", "screened_best = {}\n    diagnostics = {}")
replace(
    '"predicted_weighted_gain_kg": weighted - row["verified_weighted_kg"],',
    '"predicted_weighted_gain_kg": weighted - row["verified_weighted_kg"],\n'
    '            "passes_unchanged_native_forward": True,',
)
replace(
    "            neighbor_cache = {}",
    """            def fresh_state(ship):
                route = states[ship]["route"]
                retimer = Retimer(
                    catalogue,
                    cluster_search_settings(policy, len(catalogue.ids)),
                    dataclasses.replace(cluster_retime_settings(policy, last=True), step_days=15),
                    weights=weights,
                )
                retimer.protect_earth_leg(route.plan)
                joint = JointItinerary(
                    catalogue, retimer, weights=weights,
                    settings=JointSettings(
                        insert=False, max_moves_per_mesh=2, time_budget_seconds=15),
                )
                joint.learn(route)
                return retimer, joint

            neighbor_cache = {}""",
)
replace(
    "for (mode, _, _), value in zip(seeds, values, strict=True):",
    "for (mode, seed_arr, seed_dep), value in zip(seeds, values, strict=True):",
)
replace(
    """                        if value.feasible:
                            sample["proxy"] = remember(value.plan, case, mode)
                        record["samples"].append(sample)""",
    """                        diagnostic = incident.ranking_diagnostic(
                            state["joint"], state["route"], visits, seed_arr, seed_dep,
                            case, weights, ship_rows[case["ship"]],
                            baseline["total_mass_kg"], MIN_FLEET_RAW_KG,
                        )
                        sample["ranking_estimate"] = diagnostic
                        previous = diagnostics.get(case["case"])
                        if previous is None or (
                            incident.diagnostic_key(diagnostic, case["case"], mode)
                            < incident.diagnostic_key(previous, case["case"], previous["mode"])
                        ):
                            diagnostics[case["case"]] = diagnostic | {"mode": mode}
                        if value.feasible:
                            sample["proxy"] = remember(value.plan, case, mode)
                        record["samples"].append(sample)""",
)
start = text.index("            # Give each selected ship up to four diverse retimings")
end = text.index('            report["stage"] = "gpu_substitution_joint_polish"', start)
text = (
    text[:start]
    + """            # Shared screening evidence, independently initialized arm state, and
            # equal counts. No time-to-first-failure ratio is a complete-incidence score.
            arms = incident.choose_arms(cases, diagnostics, plan["control_case_ids"])
            arm_states = {name: {ship: fresh_state(ship) for ship in states} for name in arms}
            report["controlled_comparison"]["shared_grid_complete"] = (
                report["all_prepared_cases_screened"])
            for name, selected in arms.items():
                report["controlled_comparison"]["arms"][name] = {
                    "selected_cases": [c["case"] for c in selected],
                    "retimings_planned": len(selected),
                    "retimings_completed": 0,
                    "feasible_native_forward_plans": 0,
                    "objective_eligible_native_forward_plans": 0,
                    "retiming_seconds": 0.0,
                    "telemetry_delta": {},
                }
            report["stage"] = "gpu_substitution_retiming"
            save()
            # Alternate which arm goes first per slot; counts and complete state
            # remain visible if the shared soft time budget is reached.
            for slot in range(12):
                names = list(arms) if slot % 2 == 0 else list(reversed(arms))
                for name in names:
                    if time.perf_counter() >= proxy_deadline:
                        break
                    case = arms[name][slot]
                    began = time.perf_counter()
                    telemetry_before = dict(gpu.telemetry)
                    original = states[case["ship"]]["route"].plan
                    retimer, _arm_joint = arm_states[name][case["ship"]]
                    deploy, collect = (
                        [case["new"] if a == case["old"] else a for a in order]
                        for order in orders_of(original)
                    )
                    profile = profile_for_orders(original, retimer, deploy, collect, None)
                    result = retimer.retime_order(
                        deploy, collect, profile,
                        before=ship_rows[case["ship"]]["verified_weighted_kg"],
                        original=original,
                    )
                    record = {
                        "arm": name, "slot": slot, "case": case["case"],
                        "result": result.summary(), "seconds": time.perf_counter() - began,
                        "ranking_estimate": diagnostics.get(case["case"]),
                        "telemetry_delta": {
                            k: v - telemetry_before.get(k, 0)
                            for k, v in gpu.telemetry.items()
                            if isinstance(v, (int, float))
                            and isinstance(telemetry_before.get(k, 0), (int, float))
                        },
                    }
                    arm = report["controlled_comparison"]["arms"][name]
                    arm["retimings_completed"] += 1
                    arm["retiming_seconds"] += record["seconds"]
                    for key, value in record["telemetry_delta"].items():
                        arm["telemetry_delta"][key] = arm["telemetry_delta"].get(key, 0) + value
                    if result.plan is not None:
                        record["proxy"] = remember(result.plan, case, f"{name}_retimed")
                        arm["feasible_native_forward_plans"] += 1
                        arm["objective_eligible_native_forward_plans"] += int(
                            record["proxy"]["eligible_for_refinement"])
                    report["retimings"].append(record)
                    save()
                    retimer.release_caches()
            compared = report["controlled_comparison"]["arms"]
            report["controlled_comparison"]["equal_full_budgets_completed"] = (
                report["all_prepared_cases_screened"]
                and all(a["retimings_completed"] == 12 for a in compared.values())
            )
"""
    + text[end:]
)
replace("> 6000:", '> plan["maximum_total_joint_candidates"]:')
replace(
    '"ranking_module_sha256": sha256(ROOT / "incident.py"),',
    """"ranking_module_sha256": sha256(ROOT / "incident.py"),
        "joint_screen_call_seconds": 0.0,
        "CPU_ranking_diagnostic_seconds": 0.0,""",
)
replace(
    "                    values = evaluate_joint(",
    "                    native_began = time.perf_counter()\n"
    "                    values = evaluate_joint(",
)
replace(
    "                    if values is None or len(values) != len(seeds):",
    """                    report["joint_screen_call_seconds"] += (
                        time.perf_counter() - native_began)
                    if values is None or len(values) != len(seeds):""",
)
replace(
    "                        diagnostic = incident.ranking_diagnostic(",
    "                        diagnostic_began = time.perf_counter()\n"
    "                        diagnostic = incident.ranking_diagnostic(",
)
replace(
    '                        sample["ranking_estimate"] = diagnostic',
    """                        report["CPU_ranking_diagnostic_seconds"] += (
                            time.perf_counter() - diagnostic_began)
                        sample["ranking_estimate"] = diagnostic""",
)
replace(
    '            report["all_prepared_cases_screened"] = '
    'len(report["screened_cases"]) == len(cases)',
    """            report["shared_grid"] = incident.screen_progress(
                report["screened_cases"], cases, plan["maximum_initial_epoch_candidates"])
            report["all_prepared_cases_screened"] = report["shared_grid"]["complete"]""",
)
(root / "run.py").write_text(text)
