"""CPU construction/identity and prerequisite gates; native execution forbidden."""

from __future__ import annotations

import ctypes
import json
import os

import pytest
from common import KIT, construct, read, sha, signature
from launch import validate_prerequisite

from spacepdhcg.gtoc12.collectdp import CollectPairTable
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.search import RouteSearch


@pytest.fixture(autouse=True)
def no_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "preparation cannot load native code, generate geometry, or cost routes"
        )

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    monkeypatch.setattr(RouteSearch, "_finish_cpu", forbidden)
    monkeypatch.setattr(RouteSearch, "_finish_many", forbidden)
    monkeypatch.setattr(RouteSearch, "run", forbidden)
    monkeypatch.setattr(CollectPairTable, "pair_geometry", forbidden)


@pytest.fixture(scope="module")
def catalogue():
    os.environ["SPACEPDHCG_GTOC12_DATA"] = read(KIT / "plan.json")["data_path"]
    result = load_catalogue()
    assert result.source_sha256 == read(KIT / "plan.json")["catalogue_sha256"]
    return result


@pytest.mark.parametrize("index", range(8))
def test_each_arm_constructs_identical_archived_requests_with_real_catalogue(index, catalogue):
    group = read(KIT / "plan.json")["groups"][index]
    a, ra, ca, _ = construct(group, catalogue)
    b, rb, cb, _ = construct(group, catalogue)
    assert a.catalogue is b.catalogue is catalogue
    assert a.collect_table.catalogue is catalogue
    assert not a.collect_table._geometry and not b.collect_table._geometry
    assert len(ra) == len(rb) == group["size"]
    assert signature(ra) == signature(rb)
    assert ca == cb
    assert sum(x["expected"]["failure_name"] == "ok" for x in ca) == group["expected_accepted"]
    assert not getattr(a, "completion_capture", None)
    # Independent arm/row storage prevents one arm mutating the other's requests.
    previous = signature(rb)
    ra[0][1][next(iter(ra[0][1]))] += 1
    assert signature(rb) == previous
    if len(ra) > 10:
        assert ra[0][1] is not ra[10][1]


def test_finite_sample_budget_and_unique_controls_are_distinct():
    plan = read(KIT / "plan.json")
    unique = set()
    totals = {"ordinary": 0, "compact": 0}
    calls = 0
    for number, group in enumerate(plan["groups"]):
        sequence = ["ordinary", "compact"] if number % 2 == 0 else ["compact", "ordinary"]
        sequence += plan["warm_order"]
        assert sequence.count("ordinary") == sequence.count("compact") == 5
        assert sequence[2:6] == ["ordinary", "compact", "compact", "ordinary"]
        assert sequence[6:] == ["compact", "ordinary", "ordinary", "compact"]
        for backend in sequence:
            totals[backend] += group["size"]
            calls += 1
        unique.update(group["fixture_indices"])
    assert calls == 80 and totals == {"ordinary": 13520, "compact": 13520}
    assert len(unique) == 20


def passing_prerequisite(profile):
    return {
        "complete": True,
        "exit_code": 0,
        "expected_readbacks_present": True,
        "source_report_sha256": profile["source_report_sha256"],
        "library_sha256": profile["library"]["sha256"],
        "junit": {"tests": 7, "failures": 0, "errors": 0, "skipped": 0},
    }


@pytest.mark.parametrize(
    "field",
    [
        "complete",
        "exit_code",
        "expected_readbacks_present",
        "source_report_sha256",
        "library_sha256",
        "junit",
    ],
)
def test_missing_or_failed_correctness_evidence_cannot_authorize_benchmark(field):
    profile = read(KIT / "profile.json")
    report = passing_prerequisite(profile)
    validate_prerequisite(report, profile)
    report[field] = None
    with pytest.raises(ValueError, match="correctness"):
        validate_prerequisite(report, profile)


def test_actual_compact_source_and_control_inputs_are_pinned():
    for name, digest in read(KIT / "source-sha256.json").items():
        assert sha(KIT / "source" / name) == digest
    for name, digest in read(KIT / "input-sha256.json").items():
        assert sha(KIT / "inputs" / name) == digest
    profile = read(KIT / "profile.json")
    core = read(KIT / "inputs/compact-g-report.json")
    assert core["library"] == profile["library"]
    assert sha(KIT / "inputs/compact-g-report.json") == profile["source_report_sha256"]
    assert core["source_archive_sha256"] == sha(KIT / "inputs/compact-g-source.tar.gz")
    assert json.loads((KIT / "plan.json").read_text())["preparation_only"] is True
