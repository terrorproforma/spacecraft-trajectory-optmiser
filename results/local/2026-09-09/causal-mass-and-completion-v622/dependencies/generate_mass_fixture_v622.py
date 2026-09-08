"""Generate only native-test constants from the independently reviewed CPU oracle."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "build/performance/mass-native-oracle-v622/fixtures.json"
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == "7312df3f38291175e76df66459da86c68fdb71cb8c1895a67b68d72790291169"
j = json.loads(source.read_text())
def literal(value):
    if isinstance(value, list):
        return "{"+",".join(literal(x) for x in value)+"}"
    return repr(float(value))
lines = ["// Generated from the independent v622 oracle; no production algorithm code.",
         "// fixtures.json SHA256 "+digest, "#pragma once", "#include <vector>", "namespace mass_fixture {"]
for name in ("A", "F"):
    lines.append("inline const std::vector<std::vector<double>> "+name+"="+literal(j["original"][name])+";")
for name in ("c", "scalar_upper", "affine_offset"):
    lines.append("inline const std::vector<double> "+name+"="+literal(j["original"][name])+";")
for prefix,key in (("initial", "initial_point"), ("seed", "qualified_near_zero_seed")):
    v=j[key]
    lines.append("inline const std::vector<double> "+prefix+"_x="+literal(v["x"])+";")
    lines.append("inline const std::vector<double> "+prefix+"_y="+literal(v["scalar_dual"]+v["affine_dual"])+";")
lines.append("inline const std::vector<double> seed_c="+literal(j["qualified_near_zero_seed"]["c"])+";")
lines.append("inline const std::vector<std::vector<double>> expected_x="+literal([x["x"] for x in j["iterations"]])+";")
lines.append("inline const std::vector<std::vector<double>> expected_y="+literal([x["scalar_dual"]+x["affine_dual"] for x in j["iterations"]])+";")
lines.append("inline const std::vector<double> tau="+literal(j["reduced"]["tau"])+";")
lines.append("inline const std::vector<double> sigma="+literal(j["reduced"]["sigma"])+";")
lines.append("inline const std::vector<double> row_sums="+literal(j["reduced"]["row_abs_sums"])+";")
lines.append("inline const std::vector<double> column_sums="+literal(j["reduced"]["column_abs_sums"])+";")
lines.append("}")
out = ROOT / "cpp/cuda/tests/persistent_mass_fixture.hpp"
out.write_text("\n".join(lines)+"\n")
print(json.dumps({"output": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "GPU_calls": 0}))
