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
        existing = []
        if report.REPORT_JSON.is_file():
            existing = json.loads(report.REPORT_JSON.read_text(encoding="utf-8")).get("targets", [])
        report.write_report(report.merge_records(existing, records))
        print(f"report updated: {report.REPORT_MD}")
    return 0 if all(r["status"] in {"reproduced", "descriptive-only"} for r in records) else 2


def _report(arguments: argparse.Namespace) -> int:
    from spacepdhcg.literature import report

    if not report.REPORT_JSON.is_file():
        print("no reproduction records yet", file=sys.stderr)
        return 1
    existing = json.loads(report.REPORT_JSON.read_text(encoding="utf-8")).get("targets", [])
    report.write_report(existing)
    print(f"report re-rendered: {report.REPORT_MD}")
    return 0


def _provenance(arguments: argparse.Namespace) -> int:
    from spacepdhcg.literature.provenance import load_provenance_store

    store = load_provenance_store(known_profiles=load_target_registry().ids())
    print(f"{len(store.records)} records across {len(store.profiles())} profiles")
    for label, count in store.labels().items():
        print(f"  {label:22s} {count}")
    return 0
