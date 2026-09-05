from pathlib import Path

tests = Path("/home/ubuntu/spacepdhcg/gtoc12/tests/test_gtoc12_cooperative.py")
t = tests.read_text()
start = t.index("def _stack_depth() -> int:")
end = t.index("def test_master_packs_each_asteroid_once_and_prefers_value() -> None:")
new_test = '''def test_master_widens_the_recursion_limit_for_the_column_count(monkeypatch) -> None:
    # The depth-first search recurses once per usable column; the archive-wide H100 fleet master
    # (>1000 columns) died with RecursionError under CPython's default limit. The master must
    # widen the limit to cover its columns while searching and restore it afterwards.
    import spacepdhcg.gtoc12.cooperative as cooperative

    columns = [_column(i, {100 + i: 100.0}, {100 + i: 3000.0}, 10.0 + i) for i in range(60)]
    observed: list[int] = []
    original_ship_count = cooperative.ship_count

    def recording_ship_count(selected):
        observed.append(sys.getrecursionlimit())
        return original_ship_count(selected)

    monkeypatch.setattr(cooperative, "ship_count", recording_ship_count)
    previous = sys.getrecursionlimit()
    lowered = min(previous, 300)
    sys.setrecursionlimit(lowered)
    try:
        result = solve_fleet_master(columns, lp_bound=False)
        assert sys.getrecursionlimit() == lowered
    finally:
        sys.setrecursionlimit(previous)
    assert result.selected and fleet_feasible(result.selected) == ""
    assert observed and max(observed) >= 2 * len(columns) + 200


'''
t = t[:start] + new_test + t[end:]
tests.write_text(t)
print("test rewritten")
