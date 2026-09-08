"""CPU construction/inventory checks; no optimization or GPU access."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("next_score_driver", HERE / "run.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
sys.path.insert(0, str(REPO / "src"))


class OrderConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summaries = {i: json.loads((HERE / "inputs" / f"ship-{i:02d}.json").read_text()) for i in range(1, 24)}

    def test_all_cases_preserve_camp_footprint_and_leg_count(self):
        from spacepdhcg.gtoc12.pipeline import plan_from_route_summary
        from spacepdhcg.gtoc12.retiming import build_visits, orders_of

        cases = driver.make_cases(self.summaries)
        plan = plan_from_route_summary(self.summaries[15])
        deploy, collect = orders_of(plan)
        actual = {tuple(c["collect_order"]) for c in cases}
        self.assertEqual(61, len(cases))
        self.assertEqual(len(cases), len(actual))
        self.assertNotIn(tuple(collect), actual)
        original_legs = sum(leg.role != "camp" for leg in plan.legs)
        for case in cases:
            self.assertEqual(deploy, case["deploy_order"])
            self.assertEqual(set(collect), set(case["collect_order"]))
            self.assertEqual([19102, 13077], [case["collect_order"][0], case["collect_order"][-1]])
            visits = build_visits(deploy, case["collect_order"])
            self.assertEqual(original_legs, len(visits) - 1)
            camp = [v for v in visits if v.body == 19102]
            self.assertEqual(1, len(camp))
            self.assertTrue(camp[0].deploy and camp[0].collect)

    def test_new_incumbent_summary_matches_actual_result(self):
        from spacepdhcg.gtoc12.solution import Solution

        solution = Solution.read(HERE / "inputs/Result.txt")
        self.assertEqual(23, solution.ship_count)
        plan = self.summaries[15]["plan"]
        expected = Counter((int(a), float(t)) for phase in ("deploy_epochs", "collect_epochs") for a, t in plan[phase].items())
        selected = next(s for s in solution.ships if s.ship_id == 15)
        actual = Counter((e.event_id, e.before.epoch) for e in selected.asteroid_visits())
        self.assertEqual(expected, actual)

    def test_reject_stale_eight_collect_input(self):
        changed = copy.deepcopy(self.summaries)
        del changed[15]["plan"]["collect_epochs"]["19102"]
        with self.assertRaisesRegex(ValueError, "nine deployed and nine collected"):
            driver.make_cases(changed)

    def test_reject_shared_footprint(self):
        changed = copy.deepcopy(self.summaries)
        changed[1]["plan"]["collect_epochs"]["19102"] = 69000
        with self.assertRaisesRegex(ValueError, "shares"):
            driver.make_cases(changed)


if __name__ == "__main__":
    unittest.main()
