"""Independent saved-byte/array subtraction audit; no propagation or project imports."""

import ast
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile

READY_SHA = "3f7eb4886ad4d7eccc9a8498ea380706c37ab278c723adc81456b2dc400094e8"
REPORT_SHA = "2c71e38eb21ce0135ce9efa2f4969de6a055f2a456ed261cdce47fd594810145"
ANALYSIS_SHA = "46b9cc8bfea592834022506b73a1d8a025fb17ceca18c9cf2b5be4a17508f5de"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def arrays(payload):
    """Read this evidence's bounded little-endian FP64 NPY members, without NumPy."""
    result = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 10:
            raise ValueError("unexpected NPZ members")
        for name in names:
            member = archive.getinfo(name)
            if member.file_size > 200000 or not name.endswith(".npy"):
                raise ValueError("unexpected NPZ member size/type")
            raw = archive.read(name)
            if raw[:6] != b"\x93NUMPY" or raw[6:8] != b"\x01\x00":
                raise ValueError("unexpected NPY version")
            size = struct.unpack_from("<H", raw, 8)[0]
            header = ast.literal_eval(raw[10:10 + size].decode("latin1").strip())
            if header["descr"] != "<f8" or header["fortran_order"]:
                raise ValueError("only contiguous FP64 evidence is supported")
            shape = tuple(header["shape"])
            count = math.prod(shape)
            data = raw[10 + size:]
            if len(data) != count * 8:
                raise ValueError("invalid NPY payload length")
            values = struct.unpack(f"<{count}d", data)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite terminal readback")
            result[name[:-4]] = {"shape": shape, "values": values, "bytes": data}
    return result


def audit(get):
    if not __debug__:
        raise RuntimeError("assertions required")
    read = lambda name: json.loads(get(name))
    assert digest(get("ready.json")) == READY_SHA
    ready = read("ready.json")
    for key in ("files", "external_files"):
        for name, expected in ready[key].items():
            assert digest(get(name)) == expected, name
    report, analysis, launch = [read("output/" + name) for name in (
        "report.json", "analysis.json", "launch-report.json")]
    assert digest(get("output/report.json")) == REPORT_SHA
    assert digest(get("output/analysis.json")) == ANALYSIS_SHA
    assert report["complete"] and report["passed"] and report["comparison_passed"]
    assert launch["passed"] and launch["exit_code"] == 0 and not launch["failures"]
    assert launch["owned_descendant_cleanup"]["verified_empty"]
    assert launch["compute_processes_after_cleanup"] == ""
    assert report["evaluate_calls_started"] == report["evaluate_calls_returned"] == 1
    assert report["create_calls_started"] == report["create_calls_returned"] == report["destroy_calls"] == 1
    assert report["intervals"] == 225 and report["substeps"] == 8 and report["linearise"] == 0
    assert all(report[k] == 0 for k in ("optimizer_calls", "certificate_calls", "full_fleet_checks", "mission_score_changes"))
    raw = arrays(get("output/raw-replay.npz"))
    saved = arrays(get("output/residuals.npz"))["normalized"]
    original = arrays(get("../mass-merit-return-v632/output/candidate_mass_return/legs/00/native-raw.npz"))
    plan = read("plan.json")
    names = {"states": "states_scaled", "controls": "controls_scaled", "times": "times_scaled",
             "epochs": "epochs", "thrust_n": "thrust_n"}
    for key, name in names.items():
        assert digest(raw[name]["bytes"]) == report["input_array_sha256"][key] == plan["input_array_sha256"][key]
    for key in ("states_scaled", "epochs", "thrust_n"):
        assert raw[key]["bytes"] == original[key]["bytes"]
    assert raw["propagated_scaled"]["shape"] == saved["shape"] == (225, 7)
    states, propagated = raw["states_scaled"]["values"], raw["propagated_scaled"]["values"]
    residuals = tuple(states[i + 7] - propagated[i] for i in range(1575))
    assert residuals == saved["values"]
    largest = max(range(1575), key=lambda i: abs(residuals[i]))
    failed = [divmod(i, 7) for i, value in enumerate(residuals) if abs(value) > 5e-9]
    assert largest == 4 and failed == [(0, 4)]
    assert abs(residuals[largest]) == analysis["max_defect_normalized"] == analysis["original_max_defect_normalized"]
    assert analysis["absolute_difference_from_original_max"] == 0
    mass = plan["normalization"]["initial_mass_kg"]
    du = 1.49597870691e8
    vu = du / math.sqrt(du**3 / 1.32712440018e11)
    scales = (du, du, du, vu, vu, vu, mass)
    maxima = []
    for axis in range(7):
        interval = max(range(225), key=lambda i: abs(residuals[i * 7 + axis]))
        value = residuals[interval * 7 + axis]
        expected = analysis["per_axis_maxima"][axis]
        assert expected["interval_index_zero_based"] == interval
        assert expected["residual_normalized_signed"] == value
        assert expected["residual_physical_signed"] == value * scales[axis]
        maxima.append({"axis": axis, "interval": interval, "normalized_signed": value,
                       "physical_signed": value * scales[axis]})
    assert not analysis["virtual_vector_available"] and not analysis["is_trajectory_certificate"]
    first_thrust = raw["thrust_n"]["values"][:3]
    norm = math.sqrt(sum(value * value for value in first_thrust))
    dv_m_s = residuals[4] * vu * 1000
    return {
        "passed": True, "scope": "Saved bytes and independent scalar readback subtraction only",
        "prepared_files_checked": len(ready["files"]), "external_files_checked": len(ready["external_files"]),
        "residual_components_checked": 1575, "components_over_original_gate": failed,
        "max_defect_difference": 0.0, "per_axis_maxima": maxima,
        "first_interval_thrust_norm_N": norm,
        "missing_vy_m_s": dv_m_s,
        "constant_initial_mass_thrust_time_scale_days": mass * abs(dv_m_s) / 0.6 / 86400,
        "impulsive_rocket_equivalent_propellant_scale_kg": mass * (-math.expm1(-abs(dv_m_s) / (4000 * 9.80665))),
        "scale_estimates_are_not_a_feasible_repair_or_optimizer_result": True,
        "worker_seconds": report["worker_seconds"], "create_seconds": report["create_seconds"],
        "host_evaluate_seconds": report["host_evaluate_seconds"],
        "new_GPU_calls": 0, "new_optimizer_calls": 0, "new_orbital_propagations": 0,
        "evidence_of_global_infeasibility": False,
        "report_sha256": REPORT_SHA, "analysis_sha256": ANALYSIS_SHA,
        "raw_replay_sha256": digest(get("output/raw-replay.npz")),
        "residuals_sha256": digest(get("output/residuals.npz")),
        "launch_sha256": digest(get("output/launch-report.json")),
    }


if __name__ == "__main__":
    kit = Path(__file__).resolve().parent
    findings = audit(lambda name: (kit / name).read_bytes())
    with (kit / "saved-audit.json").open("x") as stream:
        json.dump(findings, stream, indent=2)
        stream.write("\n")
    print(json.dumps(findings, indent=2))
