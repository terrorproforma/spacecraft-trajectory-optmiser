import sys


def test_dbg(monkeypatch):
    import spacepdhcg.gtoc12.cooperative as cooperative
    from spacepdhcg.gtoc12.cooperative import FleetColumn, solve_fleet_master

    print("module", cooperative.__file__, "has_fix", "required_depth" in open(cooperative.__file__).read())
    cols = [FleetColumn(i, 1, f"c{i}", {100 + i: 100.0}, {100 + i: 3000.0}, {}, {100 + i: 10.0 + i}, True) for i in range(60)]
    observed = []
    orig = cooperative.ship_count

    def rec(selected):
        observed.append(sys.getrecursionlimit())
        return orig(selected)

    monkeypatch.setattr(cooperative, "ship_count", rec)
    prev = sys.getrecursionlimit()
    print("prev", prev)
    sys.setrecursionlimit(min(prev, 300))
    try:
        r = solve_fleet_master(cols, lp_bound=False)
    finally:
        sys.setrecursionlimit(prev)
    print("observed", len(observed), max(observed) if observed else None, "selected", len(r.selected))
    assert observed and max(observed) >= 320
