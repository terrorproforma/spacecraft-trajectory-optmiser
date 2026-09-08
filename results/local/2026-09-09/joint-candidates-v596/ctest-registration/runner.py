"""Validate only the registered joint CTests without modifying the v596 build."""

import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path("/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser")
EVIDENCE = WORKSPACE / "build/performance/joint-ctest-registration-20260909/evidence"
ROOT = Path("/home/angus/build-spacepdhcg-joint-ctest-registration-20260909")
SOURCE, BUILD = ROOT / "source", ROOT / "build"
V596 = Path("/home/angus/build-spacepdhcg-joint-v596")
PINNED_CORE = V596 / "build/cuda/libspacepdhcg_cuda.so"
CMAKE = Path("/home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake")
CTEST = CMAKE.with_name("ctest")
UPSTREAM = Path("/home/angus/spacecraft-trajectory-optmiser/_upstream/pdhcg")
NAMES = ("gtoc12_joint_smoke", "gtoc12_joint_selection_test")
PATCH_HASH = "1fca2deaa9ea496773f82e017d9849da47778717ca6daa7ce63a02c13e365b5a"
REPORT = {"schema": "joint-ctest-registration-20260909", "complete": False,
          "success": False, "commands": [], "pid": os.getpid(),
          "scope": "Only two registered CUDA joint tests; no fleet or trajectory evaluation."}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(phase, **extra):
    REPORT.update(phase=phase, updated_utc=stamp(), **extra)
    (EVIDENCE / "report.json").write_text(json.dumps(REPORT, indent=2) + "\n")
    print(json.dumps({"phase": phase, "updated_utc": REPORT["updated_utc"], **extra}), flush=True)


def run(name, command, *, environment=None):
    command = [str(value) for value in command]
    started = time.perf_counter()
    record = {"name": name, "command": command, "started_utc": stamp()}
    REPORT["commands"].append(record)
    status(name)
    with (EVIDENCE / (name + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, seconds=time.perf_counter() - started)
    status(name + "-complete", returncode=result.returncode)
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; see its evidence log")


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    REPORT["started_utc"] = stamp()
    original_core_hash = digest(PINNED_CORE)
    original_cmake_hash = digest(V596 / "source/cpp/cuda/CMakeLists.txt")
    REPORT["preserved_v596"] = {"core": str(PINNED_CORE), "core_sha256": original_core_hash,
                               "cmake_sha256": original_cmake_hash}
    try:
        assert original_core_hash == "86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671"
        patch = WORKSPACE / "cpp/cuda/CMakeLists.txt"
        assert digest(patch) == PATCH_HASH, "test-registration source changed before freezing"
        ROOT.mkdir(exist_ok=False)
        status("copying-isolated-source")
        shutil.copytree(V596 / "source", SOURCE)
        shutil.copyfile(patch, SOURCE / "cpp/cuda/CMakeLists.txt")
        REPORT["isolated_source"] = str(SOURCE)
        REPORT["cmake_test_registration_sha256"] = PATCH_HASH
        REPORT["native_sources"] = {}
        for relative in ("cpp/cuda/src/gtoc12_joint.cu",
                         "cpp/cuda/include/spacepdhcg/cuda/gtoc12_joint_c_api.h",
                         *(f"cpp/cuda/tests/{name}.cu" for name in NAMES)):
            value = digest(SOURCE / relative)
            assert value == digest(V596 / "source" / relative)
            REPORT["native_sources"][relative] = value
        environment = dict(os.environ)
        environment.update(GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="safe.directory",
                           GIT_CONFIG_VALUE_0=str(UPSTREAM), GIT_CONFIG_KEY_1="safe.directory",
                           GIT_CONFIG_VALUE_1=str(SOURCE))
        run("configure", [CMAKE, "-S", SOURCE / "cpp", "-B", BUILD,
                          "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON",
                          "-DSPACEPDHCG_BUILD_NATIVE_TESTS=OFF", "-DSPACEPDHCG_BUILD_CUDA=ON",
                          "-DCMAKE_CUDA_ARCHITECTURES=120",
                          "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc",
                          "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                          "-DSPACEPDHCG_PDHCG_SOURCE_ROOT=" + str(UPSTREAM)], environment=environment)
        expressions = "^(gtoc12_joint_smoke|gtoc12_joint_selection_test)$"
        run("ctest-list", [CTEST, "--test-dir", BUILD, "-N", "-R", expressions])
        with Path("/home/angus/.spacepdhcg-gpu.lock").open("a") as lock:
            status("waiting-for-gpu-lock")
            fcntl.flock(lock, fcntl.LOCK_EX)
            REPORT["gpu_lock_acquired_utc"] = stamp()
            run("build-two-targets", [CMAKE, "--build", BUILD, "--target", *NAMES,
                                      "--parallel", "8", "--verbose"], environment=environment)
            commands = json.loads((BUILD / "compile_commands.json").read_text())
            assertions = []
            for name in NAMES:
                selected = [entry for entry in commands if Path(entry["file"]).name == name + ".cu"]
                assert len(selected) == 1
                flags = selected[0].get("arguments") or shlex.split(selected[0]["command"])
                assert "-DNDEBUG" in flags and "-UNDEBUG" in flags
                assert flags.index("-UNDEBUG") > flags.index("-DNDEBUG")
                executable = BUILD / "cuda-tests" / name
                symbols = subprocess.check_output(["nm", "-u", str(executable)], text=True)
                (EVIDENCE / (name + "-undefined-symbols.log")).write_text(symbols)
                assert "__assert_fail" in symbols, "assert checks were removed from Release test"
                assertions.append({"target": name, "compile": selected[0],
                                   "assert_fail_symbol_present": True,
                                   "executable_sha256": digest(executable)})
            REPORT["release_assertions"] = assertions
            REPORT["isolated_core"] = {
                "path": str(BUILD / "cuda/libspacepdhcg_cuda.so"),
                "sha256": digest(BUILD / "cuda/libspacepdhcg_cuda.so"),
            }
            run("ctest-two-targets", [CTEST, "--test-dir", BUILD, "--output-on-failure",
                                      "-j", "1", "-R", expressions,
                                      "--output-junit", EVIDENCE / "ctest.xml"])
            suite = ET.parse(EVIDENCE / "ctest.xml").getroot()
            totals = {key: int(suite.get(key, "0")) for key in ("tests", "failures", "errors", "skipped")}
            assert totals == {"tests": 2, "failures": 0, "errors": 0, "skipped": 0}, totals
            REPORT["ctest_totals"] = totals
            REPORT["gpu_lock_released_utc"] = stamp()
        assert digest(PINNED_CORE) == original_core_hash
        assert digest(V596 / "source/cpp/cuda/CMakeLists.txt") == original_cmake_hash
        REPORT.update(success=True, complete=True, v596_unchanged=True)
        status("complete")
        return 0
    except Exception as error:
        REPORT["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        status("failed")
        return 1
    finally:
        REPORT["finished_utc"] = stamp()
        (EVIDENCE / "report.json").write_text(json.dumps(REPORT, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
