from pathlib import Path

tests = Path("/home/ubuntu/spacepdhcg/gtoc12/tests/test_gtoc12_cooperative.py")
t = tests.read_text()
old = """    try:
        result = solve_fleet_master(columns, lp_bound=False)
        assert sys.getrecursionlimit() == lowered
    finally:
        sys.setrecursionlimit(previous)
    assert result.selected and fleet_feasible(result.selected) == ""
    assert observed and max(observed) >= 2 * len(columns) + 200
"""
new = """    try:
        result = solve_fleet_master(columns, lp_bound=False)
        # captured before fleet_feasible() below calls ship_count again at the restored limit
        peak = max(observed) if observed else 0
        assert sys.getrecursionlimit() == lowered
    finally:
        sys.setrecursionlimit(previous)
    assert result.selected and fleet_feasible(result.selected) == ""
    assert peak >= 2 * len(columns) + 200
"""
assert old in t
tests.write_text(t.replace(old, new))
print("test fixed")
