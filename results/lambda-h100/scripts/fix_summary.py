"""Complete the deferred-sweep summary.json: ctest counts, supplement items, literature re-run, triage."""
import json, re, sys
from pathlib import Path

out = Path.home() / "spacepdhcg/v2/results/gpu/h100-deferred-3373988"
s = json.loads((out / "summary.json").read_text())
for key in list(s.get("ctest_counts", {})):
    log = out / f"{key}.log"
    if log.exists():
        text = log.read_text()
        m = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", text)
        if m:
            s["ctest_counts"][key] = {"percent": int(m[1]), "failed": int(m[2]), "total": int(m[3])}
        else:
            m = re.search(r"(\d+)% tests passed out of (\d+)", text)
            if m:
                s["ctest_counts"][key] = {"percent": int(m[1]), "failed": 0, "total": int(m[2])}


def read_items(tsv: Path) -> dict:
    items = {}
    for line in tsv.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            items[parts[0]] = {
                "verdict": parts[1],
                "exit": int(parts[2].split("=")[1]),
                "expected_exit": int(parts[3].split("=")[1]),
                "seconds": int(parts[4].rstrip("s")),
            }
    return items


sup = out / "supplement/items.tsv"
if sup.exists():
    s["supplement"] = read_items(sup)
    gdb = (out / "supplement/pd6-parity-magnitude-gdb.log").read_text()
    vals = re.findall(r"\$\d+ = \{coefficients = ([0-9.e+-]+), sigma_column = ([0-9.e+-]+), sigma_finite_difference = ([0-9.e+-]+), quaternion_radial = ([0-9.e+-]+), reconstruction = ([0-9.e+-]+)\}", gdb)
    if len(vals) >= 5:
        names = ["failing_parity", "pd6_one", "pd6_four", "pd3_one", "pd3_four"]
        s["device_time_dilated_parity"] = {
            n: dict(zip(["coefficients", "sigma_column", "sigma_finite_difference", "quaternion_radial", "reconstruction"], map(float, v)))
            for n, v in zip(names, vals)
        }
        s["device_time_dilated_parity"]["limits"] = {"pd3_coefficients": 5e-11, "pd6_coefficients": 2e-9, "pd6_sigma_column": 2e-9}
for rerun in sorted(out.glob("literature-rerun-*")):
    st = dict(line.split("=", 1) for line in (rerun / "status.txt").read_text().splitlines() if "=" in line)
    s.setdefault("literature_rerun", {})[rerun.name] = st
    for name in ("acikmese-ploen-2007-pd3", "blackmore-2010-pd3-case1", "chari-2024-pd6-monte-carlo"):
        p = rerun / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            m = d.get("measured", {})
            s["literature_rerun"][rerun.name][name] = {
                "status": d.get("status"),
                **{k: v for k, v in m.items() if "gpu" in k and not isinstance(v, (dict, list))},
            }
s["triage"] = "triage.md"
s["source_fixes_on_instance"] = ["5aabbfc fix(literature): GPU preflight must not count its own process as a foreign device holder"]
(out / "summary.json").write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: s[k] for k in ("all_pass", "ctest_counts", "supplement", "device_time_dilated_parity", "literature_rerun") if k in s}, indent=1)[:3000])
