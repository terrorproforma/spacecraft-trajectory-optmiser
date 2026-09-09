"""Portable stdlib saved-byte audit; no archived numerical execution."""

if not __debug__:
    raise RuntimeError("Assertions must remain enabled; Python -O is unsupported")

import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import stat
import zipfile

KIT = "build/performance/core-conditioning-v637/"
RESULT_INDEX = "ce20e4b9bd34baf221614a14a0d30317cd00dc8e5e6ecfabdacb8b656da976c2"
READY = "a36eca5e60a50b7ae5189df3027f23894f645d227a2e5f504d256367c01fee13"
RUN = "33f73a41db55a45d0c2632c495a5ba6996d065e1ec745534e719cbc404ff2a1c"
WORK = "94160c263477132572293e974f6c95dd34310bca316478d8c5a8e40a1577137f"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(name,payload):
    p = PurePosixPath(name)
    assert not p.is_absolute() and p.as_posix() == name and ":" not in name and "\\" not in name
    assert not any(x in (".","..",".git","__pycache__",".pytest_cache",".venv") for x in p.parts)
    assert p.suffix.lower() not in (".exe",".dll",".so",".a",".o",".pyc",".pyd",".pem")
    assert not payload.startswith((b"MZ",b"\x7fELF"))
    assert not any(x.startswith(b"-----BEGIN ") and x.endswith(b"PRIVATE KEY-----") for x in payload.splitlines())


def verify(package,index_digest):
    raw_index = (package/"index.json").read_bytes()
    if index_digest:
        assert sha(raw_index) == index_digest
    index = json.loads(raw_index)
    archive = package/"evidence.zip"
    assert archive.stat().st_size == index["archive"]["bytes"] and sha(archive.read_bytes()) == index["archive"]["sha256"]
    for name,pin in index["auxiliary_files"].items():
        value = (package/name).read_bytes()
        safe(name,value)
        assert len(value) == pin["bytes"] and sha(value) == pin["sha256"]
    assert (package/".gitattributes").read_bytes() == b"* -text -whitespace\n"
    data = {}
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        assert len(infos) == len({i.filename for i in infos}) == index["file_count"]
        assert {i.filename for i in infos} == set(index["files"])
        for item in infos:
            assert not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16)
            value = z.read(item)
            safe(item.filename,value)
            pin = index["files"][item.filename]
            assert len(value) == item.file_size == pin["bytes"] and sha(value) == pin["sha256"]
            data[item.filename] = value
    assert sum(map(len,data.values())) == index["expanded_bytes"]
    read = lambda name:json.loads(data[name])
    for name,digest in (("result-index.json",RESULT_INDEX),("ready.json",READY),("run-a/report.json",RUN),("run-a/worker/report.json",WORK)):
        assert sha(data[KIT+name]) == digest
    for manifest_name in ("result-index.json","ready.json"):
        for name,pin in read(KIT+manifest_name)["files"].items():
            value = data[KIT+name]
            assert len(value) == pin["bytes"] and sha(value) == pin["sha256"]
    coefficient = read(KIT+"coefficient-findings.json")
    for name,digest in coefficient["input_sha256"].items():
        assert sha(data[name]) == digest
    launch,worker = read(KIT+"run-a/report.json"),read(KIT+"run-a/worker/report.json")
    assert launch["complete"] and launch["passed"] and launch["worker_terminal"]
    assert launch["processes_started"] == 1 and launch["worker_returncode"] == launch["final_returncode"] == 0
    assert launch["worker_report_sha256"] == launch["terminal_worker_report_sha256"] == WORK
    assert worker["complete"] and len(worker["cases"]) == 2
    assert worker["counter"] == launch["counter"]
    expected = dict(factor_attempts=2,factor_successes=2,block_solve_attempts=2,block_solve_successes=2,
                    rhs_attempts=6,rhs_completed=6,implied_triangular_solves=12,C_products=6,
                    Gram_products=2,E_vector_products=24,ET_vector_products=12,H_vector_products=6,
                    original_residuals_formed=4,Decimal65_verification_probes=6)
    assert worker["counter"] == expected
    assert worker["optimizer_calls"] == worker["proximal_calls"] == worker["original_points_updated"] == worker["eigensolvers"] == worker["GPU_calls"] == 0
    for name,digest in worker["source_sha256"].items():
        assert sha(data[KIT+name]) == digest
    assert worker["input_sha256"] == coefficient["input_sha256"]
    assert sha(data[KIT+"runtime.json"]) == worker["runtime_sha256"] == launch["runtime_sha256"]
    tiny = read(KIT+"tiny-a/report.json")
    assert tiny["complete"] and tiny["passed"] and tiny["captured_factorizations"] == tiny["captured_solves"] == tiny["GPU_calls"] == 0
    for name,digest in tiny["source_sha256"].items():
        assert sha(data[KIT+name]) == digest
    tests = tiny["tests"]
    assert tests["passed"] and tests["exact_projector_covectors"] == 3 and tests["mass_epsilon_identity"]
    assert tests["rank_failure_retained"] and tests["duplicate_block_rejected"] and tests["invalid_operand_rejections"] == 3
    for case in worker["cases"]:
        assert case["status"] == "valid_diagnostics" and case["half_bandwidth"] == 19
        capture = case["capture"]
        path = KIT+"run-a/worker/"+capture+"/vectors.json"
        assert sha(data[path]) == case["vectors_sha256"]
        vectors = read(path)
        assert len(vectors["d"]) == len(vectors["covectors"]) == 836
        assert len(vectors["mu_original"]) == 542 and all(len(r) == 3 for r in vectors["d"])
        assert all(q["passed"] for q in vectors["quality"])
        assert len(case["probes"]) == 3 and all(p["valid"] for p in case["probes"])
        assert all(s["passed"] for s in case["cross_symmetry"])
        for probe in case["probes"]:
            assert len(probe["weighted_C_displacement"]) == 2274
            assert probe["cone_response"]["inventory"] == dict(scalar_negative=1930,scalar_positive=44,SOC_middle=37,SOC_inside=38)
        for version,digest in case["saved_points_sha256"].items():
            folder = "core-joint-reference-v635" if version == "v635" else "core-joint-halpern-v636"
            name = "build/performance/"+folder+"/run-a/worker/"+capture+"/committed_outer-10000.json"
            assert sha(data[name]) == digest
    scale = read(KIT+"covector-scale.json")
    assert scale["worker_report_sha256"] == WORK and sha(data[KIT+"check_covector_scale.py"]) == scale["source_sha256"]
    assert scale["factorizations"] == scale["solves"] == scale["projections"] == scale["optimizer_calls"] == scale["GPU_calls"] == 0
    assert len(scale["rows"]) == 2
    for dependency in index["published_predecessor_dependencies"].values():
        raw = data[dependency["package_path"]+"/index.json"]
        assert sha(raw) == dependency["index_sha256"] and json.loads(raw)["archive"]["sha256"] == dependency["archive_sha256"]
    return dict(passed=True,index_sha256=sha(raw_index),files=len(data),expanded_bytes=index["expanded_bytes"],
                captured_factors=2,captured_rhs=6,valid_saved_probes=6,original_points_updated=0,
                archived_numerical_code_executed=False,new_factors_or_solves=0,GPU_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--index-sha256")
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    result = verify(args.package,args.index_sha256)
    if args.out:
        assert not args.out.exists()
        args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
