"""Reference-reproduction report: machine-readable JSON plus the Markdown rendering."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from spacepdhcg.literature.registry import (
    REPOSITORY_ROOT,
    LiteratureTarget,
    load_target_registry,
    run_target,
)

REPORT_JSON = REPOSITORY_ROOT / "benchmarks" / "literature" / "reference_reproduction.json"
REPORT_MD = REPOSITORY_ROOT / "docs" / "REFERENCE_REPRODUCTION_REPORT.md"
DETAILS_DIR = REPOSITORY_ROOT / "results" / "literature"


def _git_commit(root: Path = REPOSITORY_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dirty(root: Path = REPOSITORY_ROOT) -> bool:
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(output.strip())


def _sanitize(value: Any) -> Any:
    """Replace NaN/inf with None so the JSON twin stays strictly valid."""

    import math

    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _compact(record: dict[str, Any]) -> dict[str, Any]:
    return _sanitize({key: value for key, value in record.items() if key != "details"})


def run_targets(
    target_ids: list[str] | None = None,
    *,
    options: dict[str, Any] | None = None,
    write_details: bool = True,
) -> list[dict[str, Any]]:
    registry = load_target_registry()
    selected = [registry[t] for t in target_ids] if target_ids else list(registry.targets)
    records: list[dict[str, Any]] = []
    for target in selected:
        record = _execute(target, options or {})
        records.append(record)
        if write_details:
            DETAILS_DIR.mkdir(parents=True, exist_ok=True)
            (DETAILS_DIR / f"{target.id}.json").write_text(
                json.dumps(_sanitize(record), indent=1, default=_json_default) + "\n",
                encoding="utf-8",
            )
    return records


def _execute(target: LiteratureTarget, options: dict[str, Any]) -> dict[str, Any]:
    import time

    start = time.perf_counter()
    try:
        record = run_target(target, options=options.get(target.id, options.get("*", {})))
    except Exception as error:
        record = {
            "target_id": target.id,
            "family": target.family,
            "status": "blocked",
            "published": {},
            "measured": {},
            "gap": {},
            "labels": {},
            "envelope": {},
            "commands": [f"spacepdhcg literature run {target.id}"],
            "notes": [f"runner raised {type(error).__name__}: {error}"],
        }
    record["wall_seconds"] = time.perf_counter() - start
    record["support"] = target.support
    if target.unsupported_reason:
        record["unsupported_reason"] = target.unsupported_reason
    return record


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and value != value:
        return None
    return str(value)


def merge_records(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {record["target_id"]: record for record in existing}
    for record in new:
        by_id[record["target_id"]] = record
    return [by_id[key] for key in sorted(by_id)]


def write_report(records: list[dict[str, Any]], *, commit: str | None = None) -> dict[str, Any]:
    document = {
        "schema_version": "1.0.0",
        "generated_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
        "repository_commit": commit or _git_commit(),
        "working_tree_dirty": _dirty(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "targets": [_compact(record) for record in records],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(document, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    REPORT_MD.write_text(render_markdown(document), encoding="utf-8")
    return document


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return "; ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


def _headline(record: dict[str, Any]) -> tuple[str, str, str]:
    """Pick the headline published/measured/gap triple for the summary table."""

    published = record.get("published", {})
    measured = record.get("measured", {})
    gap = record.get("gap", {})
    tid = record["target_id"]
    if tid.endswith("pd3") or "pd3" in tid:
        gpu = measured.get("scvx_qoco_gpu_fuel_used_kg")
        if gpu is not None:
            gpu_text = f"{_fmt(gpu)} kg (repo SCvx QOCO-GPU)"
        elif measured.get("scvx_qoco_gpu_status") == "deferred":
            gpu_text = "GPU leg deferred (G4 owns the device; `literature gpu-run`)"
        else:
            gpu_text = "GPU leg not run"
        return (
            f"{published.get('fuel_used_text', published.get('fuel_used_kg', '-'))} kg propellant",
            f"{_fmt(measured.get('lossless_fuel_used_kg', '-'))} kg (lossless SOCP); "
            f"{_fmt(measured.get('scvx_cpu_fuel_used_kg', '-'))} kg (repo SCvx CPU, "
            f"{measured.get('scvx_cpu_status', '-')}); {gpu_text}",
            f"{_fmt(gap.get('lossless_minus_published_kg', '-'))} kg (lossless); "
            f"{_fmt(gap.get('scvx_cpu_minus_published_kg', '-'))} kg (SCvx CPU)",
        )
    if "szmuk" in tid:
        ext = measured.get("extended_run", {})
        spread = _fmt(measured.get("time_of_flight_spread_ut", "-"))
        spread_gap = _fmt(gap.get("time_of_flight_spread_minus_published_ut", "-"))
        native = measured.get("native_pd6_fft", {})
        if "time_of_flight" in native:
            native_text = (
                f"; native pd6_fft t_f = {_fmt(native['time_of_flight'])} UT "
                f"({native.get('status')}, gap vs core {_fmt(native.get('gap_vs_cpu_core_ut'))} UT)"
            )
        else:
            native_text = f"; native pd6_fft {native.get('status', 'not run')}"
        return (
            f"t_f: figure-only; sweep spread <= {published.get('tf_guess_sweep_spread_ut')} UT; "
            f"{published.get('iterations_to_converge')} iterations",
            f"t_f = {_fmt(ext.get('time_of_flight', '-'))} UT; sweep spread {spread} UT"
            + native_text,
            f"spread - published = {spread_gap} UT; t_f descriptive-only",
        )
    if "chari" in tid:
        batches = measured.get("cpu_independent_batch", {})
        summary = "; ".join(
            f"N={k}: conv {v['convergence_probability']:.2f}, "
            f"{v['accepted_trajectories_per_second']:.2f} traj/s"
            for k, v in batches.items()
        )
        gpu = measured.get("gpu_persistent_batch", {})
        pure = gpu.get("pure_qoco_native_pd6_fft", {}) if isinstance(gpu, dict) else {}
        if pure.get("status") == "measured":
            gpu_text = "GPU pure-QOCO pd6_fft batch measured"
        elif pure.get("status") == "deferred":
            pre = pure.get("preflight") or {}
            why = "G4 owns the device" if pre.get("g4_owned") else "preflight refused"
            gpu_text = (
                f"GPU pure-QOCO pd6_fft batch deferred ({why}; `literature gpu-run`); "
                "device SCvx blocked"
            )
        else:
            gpu_text = "GPU batch blocked / not run"
        return (
            "batch 256 demonstrated (no objective printed)",
            summary or "-",
            gpu_text,
        )
    if "tafazzol" in tid:
        best = measured.get("final_mass_kg_best")
        if best is None:
            by_nodes = measured.get("final_mass_kg_by_nodes", {})
            statuses = measured.get("statuses", {})
            measured_text = "no converged run; " + "; ".join(
                f"{k}: {_fmt(v)} kg ({statuses.get(k, '?')})" for k, v in by_nodes.items()
            )
            gap_text = "not converged"
        else:
            measured_text = f"{_fmt(best)} kg"
            gap_text = f"{_fmt(gap.get('final_mass_minus_published_kg', '-'))} kg"
        return (f"{published.get('final_mass_text', '-')} kg final mass", measured_text, gap_text)
    if "tops" in tid:
        parts = [f"{k}: {v.get('status')}" for k, v in measured.items()]
        return ("no reference objectives", "; ".join(parts), "-")
    if "gtopx" in tid:
        return (
            "; ".join(
                f"{k}={v}"
                for k, v in published.items()
                if k not in {"evidence_label", "extraction"}
            ),
            "; ".join(f"{k}={_fmt(v)}" for k, v in measured.items()),
            "; ".join(f"{k}={v:.1e}" for k, v in gap.items()),
        )
    if "gtoc12" in tid:
        parts = [
            f"{k}: {v.get('mined_asteroids')} asteroids, {_fmt(v.get('total_resource_mass_kg'))} kg"
            for k, v in measured.items()
        ]
        return ("official verifier acceptance", "; ".join(parts), "-")
    if "gtoc9" in tid:
        parts = [
            f"{k}: valid={v.get('valid')} debris={v.get('debris_removed')}"
            for k, v in measured.items()
        ]
        return ("example1 removes debris 23, 3, 51", "; ".join(parts), "-")
    return (_fmt(published), _fmt(measured), _fmt(gap))


def render_markdown(document: dict[str, Any]) -> str:
    lines = [
        "# Reference reproduction report (Phase 0-1 of the comparative solver campaign)",
        "",
        f"Generated: {document['generated_utc']}  ",
        f"Repository commit: `{document['repository_commit']}`"
        + (" (working tree dirty)" if document.get("working_tree_dirty") else ""),
        f"  \nHost: {document['host']['platform']}, Python {document['host']['python']}",
        "",
        "Machine-readable twin: `benchmarks/literature/reference_reproduction.json`; "
        "per-target details in `results/literature/<target>.json`; provenance in "
        "`benchmarks/literature/provenance.json`; target registry in "
        "`benchmarks/literature/targets.json`.",
        "",
        "Status vocabulary: `reproduced` (within the declared envelope), `gap` (measured but "
        "outside the envelope or not converged), `descriptive-only` (published data "
        "unrecoverable), `unsupported` (dynamics/model not implemented), `blocked` (external "
        "dependency).",
        "",
        "## Summary",
        "",
        "| Target | Family | Status | Published | Measured | Gap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in document["targets"]:
        published, measured, gap = _headline(record)
        family = record.get("family") or "secondary"
        lines.append(
            f"| `{record['target_id']}` | {family} | **{record['status']}** | "
            f"{published} | {measured} | {gap} |"
        )
    lines += ["", "## Per-target detail", ""]
    for record in document["targets"]:
        lines.append(f"### `{record['target_id']}`")
        lines.append("")
        lines.append(f"- Family: {record.get('family') or 'secondary global mission-design track'}")
        lines.append(f"- Status: **{record['status']}** (support: {record.get('support', '?')})")
        if record.get("unsupported_reason"):
            lines.append(f"- Unsupported reason: {record['unsupported_reason']}")
        lines.append(f"- Wall time: {record.get('wall_seconds', 0.0):.1f} s")
        if record.get("labels"):
            lines.append("- Evidence labels:")
            for key, label in record["labels"].items():
                lines.append(f"  - `{key}`: `{label}`")
        if record.get("published"):
            lines.append("- Published:")
            for key, value in record["published"].items():
                lines.append(f"  - {key}: {_fmt(value)}")
        if record.get("measured"):
            lines.append("- Measured:")
            for key, value in record["measured"].items():
                lines.append(f"  - {key}: {_fmt(value)}")
        if record.get("gap"):
            lines.append("- Gap:")
            for key, value in record["gap"].items():
                lines.append(f"  - {key}: {_fmt(value)}")
        if record.get("envelope"):
            lines.append("- Discretisation envelope:")
            for key, value in record["envelope"].items():
                lines.append(f"  - {key}: {_fmt(value)}")
        if record.get("commands"):
            lines.append("- Commands: " + "; ".join(f"`{c}`" for c in record["commands"]))
        for note in record.get("notes", []):
            lines.append(f"- Note: {note}")
        lines.append("")
    return "\n".join(lines) + "\n"
