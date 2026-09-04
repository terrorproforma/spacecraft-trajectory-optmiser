"""``spacepdhcg literature`` sub-commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from spacepdhcg.literature import external_sources
from spacepdhcg.literature.registry import load_target_registry


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("literature", help="literature reproduction targets")
    commands = parser.add_subparsers(dest="literature_command", required=True)

    list_parser = commands.add_parser("list", help="list registered targets")
    list_parser.set_defaults(func=_list)

    fetch_parser = commands.add_parser(
        "fetch", help="download and verify pinned external artifacts"
    )
    fetch_parser.add_argument("artifacts", nargs="*", help="artifact ids (default: all)")
    fetch_parser.set_defaults(func=_fetch)

    status_parser = commands.add_parser("status", help="show cache status of pinned artifacts")
    status_parser.set_defaults(func=_status)

    run_parser = commands.add_parser("run", help="run one or more targets")
    run_parser.add_argument("targets", nargs="+", help="target ids or 'all'")
    run_parser.add_argument(
        "--option", action="append", default=[], help="key=json-value passed to the runner"
    )
    run_parser.add_argument(
        "--no-report", action="store_true", help="do not update the reproduction report"
    )
    run_parser.set_defaults(func=_run)

    report_parser = commands.add_parser("report", help="re-render the report from existing records")
    report_parser.set_defaults(func=_report)

    provenance_parser = commands.add_parser("provenance", help="validate the provenance store")
    provenance_parser.set_defaults(func=_provenance)

    preflight_parser = commands.add_parser(
        "gpu-preflight",
        help="report whether the literature GPU legs may run (refuses while G4 owns the device)",
    )
    preflight_parser.add_argument(
        "--allow-shared",
        action="store_true",
        help="tolerate non-G4 compute processes on the device",
    )
    preflight_parser.set_defaults(func=_gpu_preflight)

    gpu_run_parser = commands.add_parser(
        "gpu-run",
        help=(
            "deferred GPU legs (P1-C pure-QOCO SCvx, P1-D-MC pure-QOCO batch): run the preflight, "
            "refuse while the G4 session owns the RTX 5090, otherwise run the target with "
            "run_gpu=true"
        ),
    )
    gpu_run_parser.add_argument("targets", nargs="+", help="target ids")
    gpu_run_parser.add_argument(
        "--option", action="append", default=[], help="key=json-value passed to the runner"
    )
    gpu_run_parser.add_argument("--allow-shared", action="store_true")
    gpu_run_parser.add_argument("--no-report", action="store_true")
    gpu_run_parser.set_defaults(func=_gpu_run)


def _list(arguments: argparse.Namespace) -> int:
    registry = load_target_registry()
    for target in registry.targets:
        print(
            f"{target.id:36s} {target.family or 'secondary':18s} "
            f"{target.support:16s} {target.title}"
        )
    return 0


def _fetch(arguments: argparse.Namespace) -> int:
    manifest = external_sources.load_manifest()
    ids = arguments.artifacts or list(manifest)
    failures = 0
    for artifact_id in ids:
        try:
            path = external_sources.fetch(artifact_id, online=True, manifest=manifest)
            print(f"verified {artifact_id} -> {path}")
        except Exception as error:
            failures += 1
            print(f"FAILED {artifact_id}: {error}", file=sys.stderr)
    return 1 if failures else 0


def _status(arguments: argparse.Namespace) -> int:
    for row in external_sources.status():
        print(f"{row['state']:18s} {row['id']:40s} {row['path']}")
    return 0


def _parse_options(items: Sequence[str]) -> dict:
    options: dict = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            options[key] = json.loads(raw)
        except json.JSONDecodeError:
            options[key] = raw
    return {"*": options}


def _run(arguments: argparse.Namespace) -> int:
    from spacepdhcg.literature import report

    targets = None if arguments.targets == ["all"] else list(arguments.targets)
    records = report.run_targets(targets, options=_parse_options(arguments.option))
    for record in records:
        print(json.dumps(report._compact(record), indent=1, default=report._json_default))
    if not arguments.no_report:
        report.write_report(report.merge_records(_existing_records(), records))
        print(f"report updated: {report.report_markdown_path()}")
    return 0 if all(r["status"] in {"reproduced", "descriptive-only"} for r in records) else 2


def _existing_records() -> list:
    from spacepdhcg.literature import report

    path = report.report_json_path()
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("targets", [])


def _report(arguments: argparse.Namespace) -> int:
    from spacepdhcg.literature import report

    if not report.report_json_path().is_file():
        print("no reproduction records yet", file=sys.stderr)
        return 1
    report.write_report(_existing_records())
    print(f"report re-rendered: {report.report_markdown_path()}")
    return 0


#: Exit code of ``gpu-run`` / ``gpu-preflight`` when the device must not be used.
GPU_REFUSED_EXIT_CODE = 3


def _gpu_preflight(arguments: argparse.Namespace) -> int:
    from spacepdhcg.literature.gpu_preflight import preflight

    result = preflight(allow_shared=arguments.allow_shared)
    print(json.dumps(result.as_dict(), indent=1))
    return 0 if result.ok else GPU_REFUSED_EXIT_CODE


def _gpu_run(arguments: argparse.Namespace) -> int:
    from spacepdhcg.literature import report
    from spacepdhcg.literature.gpu_preflight import preflight

    gate = preflight(allow_shared=arguments.allow_shared)
    print(json.dumps(gate.as_dict(), indent=1))
    if not gate.ok:
        print(f"GPU leg refused: {gate.reason}", file=sys.stderr)
        return GPU_REFUSED_EXIT_CODE
    options = _parse_options(arguments.option)
    options["*"]["run_gpu"] = True
    records = report.run_targets(list(arguments.targets), options=options)
    for record in records:
        print(json.dumps(report._compact(record), indent=1, default=report._json_default))
    if not arguments.no_report:
        report.write_report(report.merge_records(_existing_records(), records))
        print(f"report updated: {report.report_markdown_path()}")
    return 0 if all(r["status"] in {"reproduced", "descriptive-only"} for r in records) else 2


def _provenance(arguments: argparse.Namespace) -> int:
    from spacepdhcg.literature.provenance import load_provenance_store

    store = load_provenance_store(known_profiles=load_target_registry().ids())
    print(f"{len(store.records)} records across {len(store.profiles())} profiles")
    for label, count in store.labels().items():
        print(f"  {label:22s} {count}")
    return 0
