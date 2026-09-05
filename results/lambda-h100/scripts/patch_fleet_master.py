"""Make the fleet-master depth-first search survive archive-wide column pools.

`solve_fleet_master.search` recurses once per usable column (the skip branch always advances
`index` to `len(usable)`), so an archive-wide master with more columns than CPython's default
1000-frame recursion limit dies with RecursionError after the (expensive) recertification stage.
Raise the limit for the duration of the search, scaled to the column count, and restore it after.
"""
from pathlib import Path

root = Path("/home/ubuntu/spacepdhcg/gtoc12")
src = root / "src/spacepdhcg/gtoc12/cooperative.py"
text = src.read_text()

old_imports = "import math\nfrom dataclasses import dataclass, field\n"
new_imports = "import math\nimport sys\nfrom dataclasses import dataclass, field\n"
assert old_imports in text
text = text.replace(old_imports, new_imports, 1)

old_call = """    search(0, (), 0.0, 0.0, {}, set(), zero_dual, np.ones(n_usable, dtype=bool))
    # LP-based branch and bound closes (or bounds) what the combinatorial search left open:"""
new_call = """    # ``search`` recurses once per usable column (the skip branch always walks ``index`` to
    # ``len(usable)``), so an archive-wide master offers more columns than CPython's default
    # 1000-frame limit; widen it for the search and restore it afterwards.
    required_depth = 2 * n_usable + 200
    previous_limit = sys.getrecursionlimit()
    if required_depth > previous_limit:
        sys.setrecursionlimit(required_depth)
    try:
        search(0, (), 0.0, 0.0, {}, set(), zero_dual, np.ones(n_usable, dtype=bool))
    finally:
        if required_depth > previous_limit:
            sys.setrecursionlimit(previous_limit)
    # LP-based branch and bound closes (or bounds) what the combinatorial search left open:"""
assert old_call in text
text = text.replace(old_call, new_call, 1)
src.write_text(text)

tests = root / "tests/test_gtoc12_cooperative.py"
ttext = tests.read_text()
anchor = "def test_master_packs_each_asteroid_once_and_prefers_value() -> None:"
assert anchor in ttext
new_test = '''def _stack_depth() -> int:
    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def test_master_search_depth_scales_with_the_column_count() -> None:
    # 60 mutually compatible columns: the depth-first search recurses once per column, so with a
    # recursion limit only slightly above the current depth the master used to die with
    # RecursionError (seen on the archive-wide H100 fleet master with >1000 columns).
    columns = [_column(i, {100 + i: 100.0}, {100 + i: 3000.0}, 10.0 + i) for i in range(60)]
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(_stack_depth() + 30)
    try:
        result = solve_fleet_master(columns, lp_bound=False)
    finally:
        sys.setrecursionlimit(previous)
    assert result.selected and fleet_feasible(result.selected) == ""
    assert sys.getrecursionlimit() == previous


'''
ttext = ttext.replace(anchor, new_test + anchor, 1)
if "\nimport sys\n" not in ttext:
    # add the import after the last top-level import line of the module header
    lines = ttext.split("\n")
    last_import = max(i for i, line in enumerate(lines[:60]) if line.startswith(("import ", "from ")))
    lines.insert(last_import + 1, "import sys")
    ttext = "\n".join(lines)
tests.write_text(ttext)
print("patched", src, tests)
