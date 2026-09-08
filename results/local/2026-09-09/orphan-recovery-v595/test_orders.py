"""Cheap construction checks; no GPU access or trajectory optimization."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("orphan_recovery_driver", HERE / "run.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
sys.path.insert(0, str(ROOT / "src"))


class OrderConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources"
        cls.summaries = {i: json.loads((source / f"ship-{i:02d}.json").read_text()) for i in range(1, 24)}

    def test_cases_preserve_deploys_and_collect_own_orphan_once(self):
        from spacepdhcg.gtoc12.pipeline import plan_from_route_summary
        from spacepdhcg.gtoc12.retiming import build_visits

        cases = driver.make_cases(self.summaries, [15, 3])
        self.assertEqual(18, len(cases))
        self.assertEqual({19102, 44233}, {c["orphan"] for c in cases})
        for case in cases:
            plan = plan_from_route_summary(self.summaries[case["ship"]])
            original_legs = sum(leg.role != "camp" for leg in plan.legs)
            visits = build_visits(case["deploy_order"], case["collect_order"])
            self.assertEqual(set(plan.deploy_epochs), set(case["deploy_order"]))
            self.assertEqual(set(plan.collect_epochs) | {case["orphan"]}, set(case["collect_order"]))
            self.assertEqual(original_legs + case["additional_low_thrust_legs"], len(visits) - 1)
            matching = [v for v in visits if v.body == case["orphan"]]
            if case["position"] == 0:
                self.assertEqual(1, len(matching))
                self.assertTrue(matching[0].deploy and matching[0].collect)
            else:
                self.assertEqual(2, len(matching))
                self.assertEqual(1, sum(v.deploy for v in matching))
                self.assertEqual(1, sum(v.collect for v in matching))

    def test_reject_orphan_claimed_by_another_incumbent(self):
        changed = copy.deepcopy(self.summaries)
        changed[1]["plan"]["collect_epochs"]["19102"] = 69000
        with self.assertRaisesRegex(ValueError, "already used"):
            driver.make_cases(changed, [15])

    def test_reject_stale_orphan_metadata(self):
        changed = copy.deepcopy(self.summaries)
        changed[15]["plan"]["orphaned"] = []
        with self.assertRaisesRegex(ValueError, "exactly one own orphan"):
            driver.make_cases(changed, [15])


if __name__ == "__main__":
    unittest.main()
