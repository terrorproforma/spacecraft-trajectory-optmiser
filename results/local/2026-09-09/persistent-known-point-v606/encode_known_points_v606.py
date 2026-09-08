"""Encode explicitly translated, hash-bound reference points without altering their values."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent / "known-point-replay-v606"
spec = importlib.util.spec_from_file_location("audit", root / "source/audit_persistent_snapshot.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
manifest = json.loads((root / "preparation.json").read_text())
assert manifest["complete"] and manifest["gpu_calls"] == 0
report = {"complete": False, "gpu_calls": 0, "coordinates": "translated", "points": []}
for row in manifest["points"]:
    label = row["label"]
    source = root / "inputs" / (label + "-point.json")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row["point_sha256"]
    point = json.loads(source.read_text())
    lines = ["SPACEPDHCG_QOCO_INITIAL_POINT_V1",
             "snapshot_sha256 " + row["snapshot_sha256"], "coordinates translated"]
    for name, key in (("x", "x_translated"), ("y", "y_original"),
                      ("z", "z_original"), ("s", "s_original")):
        values = point[key]
        assert np.all(np.isfinite(values))
        tokens = [format(value, ".17g") for value in values]
        assert [float(token) for token in tokens] == values
        lines.append(f"{name} {len(values)} " + " ".join(tokens))
    target = root / "inputs" / (label + "-initial.txt")
    with target.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    problem = audit.load_snapshot(root / "inputs" / (label + ".txt"))
    record = {"x": point["x_original"], "y": point["y_original"],
              "z": point["z_original"], "s": point["s_original"],
              "status": point["reference_status"]}
    supplied = audit.audit(problem, record, backend="qoco", coordinates="original")
    assert supplied["qualified"]
    reconstructed_slack = (problem["h"] - problem["G"] @ np.asarray(
        record["x"], dtype=np.longdouble)).astype(np.float64).tolist()
    reconstructed = audit.audit(problem, {**record, "s": reconstructed_slack},
                                backend="qoco", coordinates="original")
    report["points"].append({
        "label": label,
        "encoded_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "snapshot_sha256": row["snapshot_sha256"],
        "supplied_point_audit": supplied,
        "generic_reconstructed_slack_audit": reconstructed,
        "note": "CPU audit only; native mapping and actual solver behavior are still untested",
    })
report["complete"] = True
with (root / "encoded-points.json").open("x") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps(report))
