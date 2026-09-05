"""Resolve the recursion-limit conflict between ba9b764 (WSL) and c4e2c31 (H100) in cooperative.py."""
import pathlib, re, sys
path = pathlib.Path("src/spacepdhcg/gtoc12/cooperative.py")
text = path.read_text()
pattern = re.compile(r"<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", re.S)
blocks = pattern.findall(text)
assert len(blocks) == 1, f"expected exactly one conflict, found {len(blocks)}"
ours, theirs = blocks[0]
assert "required_depth" in ours and "n_usable + 500" in theirs, (ours, theirs)
resolved = '''    # ``search`` recurses once per usable column (the skip branch always walks ``index`` to
    # ``len(usable)``), so an archive-wide master offers more columns than CPython's default
    # 1000-frame limit (1019 columns in fleet_master_v6, 1055 on the H100 host); widen it for
    # the search and restore it afterwards.  Both fixes' margins are honoured.
    required_depth = max(2 * n_usable + 200, n_usable + 500)
    previous_limit = sys.getrecursionlimit()
    if required_depth > previous_limit:
        sys.setrecursionlimit(required_depth)
    try:
        search(0, (), 0.0, 0.0, {}, set(), zero_dual, np.ones(n_usable, dtype=bool))
    finally:
        if required_depth > previous_limit:
            sys.setrecursionlimit(previous_limit)
'''
text = pattern.sub(lambda m: resolved, text, count=1)
assert "<<<<<<<" not in text and ">>>>>>>" not in text
path.write_text(text)
print("resolved cooperative.py")