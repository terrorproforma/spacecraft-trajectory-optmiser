"""Generate the H100 current-head reseal scripts from the sealed b6afb49 (RTX 5090) templates.

Usage: adapt_reseal.py TEMPLATE_DIR REPO_ROOT
Writes REPO_ROOT/results/gpu/current-head-<sha7>-h100/{preflight,g0,g1,g2,g3,seals}/...
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

template = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2]).resolve()
head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
tree = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], check=True, capture_output=True, text=True).stdout.strip()
assert re.fullmatch(r"[0-9a-f]{40}", head) and re.fullmatch(r"[0-9a-f]{40}", tree)
evidence_rel = f"results/gpu/current-head-{head[:7]}-h100"
out = root / evidence_rel
venv = f"{root}/.venv"

subs = [
    ("results/gpu/current-head-b0cd570", evidence_rel),
    ("b6afb49d7fc7da5ed1ac9003c3bcae5d35506026", head),
    ("a91553cd646393e343dabc332ce4921e753ca219", tree),
    ("/home/angus/spacecraft-trajectory-optmiser/.venv", venv),
    ("${root}/.venv-current-head", "${root}/.venv"),
    (".venv-current-head", ".venv"),
    ("/home/angus/.local/bin/uv", "/home/ubuntu/.local/bin/uv"),
    ("-DCMAKE_CUDA_ARCHITECTURES=120", "-DCMAKE_CUDA_ARCHITECTURES=90"),
    ("_upstream/qoco-current-head", "_upstream/qoco-g4"),
    ("_upstream/qoco-gpu", "_upstream/qoco-g4"),
    ("--parallel 4", "--parallel 8"),
    ("build-current-head-qoco-cudss-lib", "build-current-head-qoco-cudss-lib"),
]

files = {
    "preflight/capture.sh": [],
    "preflight/build_qoco.sh": [],
    "g0/run.sh": [],
    "g1/run.sh": [],
    "g2/run.sh": [],
    "g3/run.sh": [
        (
            "export CUDA_VISIBLE_DEVICES=0\n",
            "export CUDA_VISIBLE_DEVICES=0\n"
            "export SPACEPDHCG_HARDWARE_ID=lambda-h100-80gb-hbm3\n"
            "export SPACEPDHCG_CUDA_ARCHITECTURES=90\n",
        ),
    ],
    "g3/run_displaced_regressions.py": [],
    "seals/validate.sh": [],
    "seals/seal.sh": [],
    "seals/verify_seals.py": [],
    "seals/summarize.py": [
        # failed-attempt counts are measured, not inherited from the RTX 5090 reseal
        (
            'def text(path: Path) -> str:\n',
            'def failed_attempts(gate: Path) -> int:\n'
            '    failures = gate / "failures"\n'
            '    return len([p for p in failures.iterdir()]) if failures.is_dir() else 0\n\n\n'
            'def text(path: Path) -> str:\n',
        ),
        ('    "retained_failed_attempts": 3,\n', '    "retained_failed_attempts": failed_attempts(g0),\n'),
        ('g1_summary["retained_failed_attempts"] = 3\n', 'g1_summary["retained_failed_attempts"] = failed_attempts(g1)\n'),
        (
            '        "stream_lifetime racecheck with a 100-billion-iteration cancellation kernel was retained "\n'
            '        "as a hung instrumentation attempt; the complete persistent kernel racecheck passed on "\n'
            '        "persistent_cw_test and stream cancellation/destruction passed natively."\n',
            '        "racecheck runs on persistent_cw_test (the complete persistent kernel), the target "\n'
            '        "chosen for the b6afb49 RTX 5090 seal after a stream_lifetime racecheck hung there; "\n'
            '        "stream cancellation/destruction are covered natively by stream_lifetime_test."\n',
        ),
        (
            '    "nsight_gpu_kernel_records_available": False,\n'
            '    "nsight_gpu_memory_records_available": False,\n'
            '    "nsight_wsl_limitation": (\n'
            '        "CUDA API records are present, but Nsight Systems 2024.6.2 under WSL reports no CUDA "\n'
            '        "kernel or GPU-memory records; no timeline-residency claim is made."\n'
            '    ),\n',
            '    "nsight_gpu_kernel_records_available": nsight_kernel_records,\n'
            '    "nsight_gpu_memory_records_available": nsight_memory_records,\n'
            '    "nsight_note": nsight_note,\n',
        ),
        (
            'g3_sanitizers = sorted(g3.glob("sanitizer-*.log"))\n',
            'g3_sanitizers = sorted(g3.glob("sanitizer-*.log"))\n'
            'nsys_text = text(g3 / "nsys-stats.log")\n'
            'nsight_kernel_records = (\n'
            '    "cuda_gpu_kern_sum" in nsys_text\n'
            '    and re.search(r"cuda_gpu_kern_sum\\.py\\]\\.\\.\\.\\s*\\n\\s*SKIPPED", nsys_text) is None\n'
            ')\n'
            'nsight_memory_records = (\n'
            '    "cuda_gpu_mem_time_sum" in nsys_text\n'
            '    and re.search(r"cuda_gpu_mem_time_sum\\.py\\]\\.\\.\\.\\s*\\n\\s*SKIPPED", nsys_text) is None\n'
            ')\n'
            'nsight_note = (\n'
            '    "Native Linux H100 host: Nsight Systems CUDA API, GPU kernel and GPU memory summaries "\n'
            '    "are recorded when available (see nsys-stats.log); no timeline-residency claim is made."\n'
            ')\n',
        ),
        (
            '    "g4_claim_core_launch_ready": False,\n'
            '    "g4_launch_blocker": (\n'
            '        "A new official G4 capability must be generated for the final clean executable; the "\n'
            '        "b0cd570 capability was not used as G3 authority and no G4 campaign was launched."\n'
            '    ),\n',
            '    "g4_claim_core_launch_ready": False,\n'
            '    "g4_launch_blocker": (\n'
            '        "H100 reseal: a new official G4 capability (IPM probe plus a 20 s-deadline PDHCG "\n'
            '        "session probe) must be generated on this host from the final clean executable; the "\n'
            '        "executor deadline defect found on the RTX 5090 campaign (adaptive attempts running "\n'
            '        "to the 1,000,000-iteration cap) must be fixed and rebuilt first. No G4 campaign was "\n'
            '        "launched from this evidence."\n'
            '    ),\n'
            '    "hardware_id": "lambda-h100-80gb-hbm3",\n'
            '    "cuda_architecture": 90,\n'
            '    "reseal_of_rtx5090_seal": "b6afb49d7fc7da5ed1ac9003c3bcae5d35506026 (results/gpu/current-head-b0cd570, WSL)",\n',
        ),
    ],
}

for rel, extra in files.items():
    src = template / rel
    text = src.read_text(encoding="utf-8")
    for old, new in subs:
        text = text.replace(old, new)
    for old, new in extra:
        if old not in text:
            raise SystemExit(f"{rel}: expected snippet missing:\n{old}")
        text = text.replace(old, new, 1)
    if "angus" in text:
        raise SystemExit(f"{rel}: unresolved WSL path remains: " + [l for l in text.splitlines() if "angus" in l][0])
    dest = out / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    if rel.endswith(".sh"):
        dest.chmod(0o755)
print(f"head={head}\ntree={tree}\nevidence={out}")
