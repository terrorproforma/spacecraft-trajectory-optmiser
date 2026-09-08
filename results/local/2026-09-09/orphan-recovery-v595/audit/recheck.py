"""Fresh CPU-only independent and official verification; leaves run outputs untouched."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

repo = Path.cwd()
root = repo / "build/performance/orphan-recovery-v595"
out = root / "audit"
out.mkdir(exist_ok=True)
sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
sys.path.insert(0, str(root / "source/src"))
from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue
from spacepdhcg.gtoc12.official import run_official_verifier
from spacepdhcg.gtoc12.verifier import Gtoc12Verifier

path = root / "output-compatible/best/Result.txt"
before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
catalogue, bonus = load_catalogue(), load_bonus_table()
started = time.perf_counter()
independent = Gtoc12Verifier(catalogue, bonus=bonus).verify_file(path)
independent_seconds = time.perf_counter() - started
official = run_official_verifier(path)
after_hash = hashlib.sha256(path.read_bytes()).hexdigest()
assert before_hash == after_hash
summary = {
    "kind": "fresh_cpu_verification_no_gpu_execution",
    "pid": os.getpid(),
    "solution_sha256": before_hash,
    "independent": independent.summary(),
    "independent_seconds": independent_seconds,
    "official": official.summary(),
    "official_stdout": official.stdout,
    "official_stderr": official.stderr,
    "official_score_data": official.score_data,
    "scored_masses": independent.scored_masses,
    "ship_unloaded_mass": independent.ship_unloaded_mass,
    "catalogue_sha256": catalogue.source_sha256,
    "bonus_sha256": bonus.source_sha256,
    "tolerances": {"position_km": C.TOLERANCE_POSITION_KM,
                   "velocity_km_s": C.TOLERANCE_VELOCITY_KM_S,
                   "mass_kg": C.TOLERANCE_MASS_KG},
    "source_sha256": {name: hashlib.sha256((root / "source/src/spacepdhcg/gtoc12" / name).read_bytes()).hexdigest()
                      for name in ("verifier.py", "official.py", "constants.py", "data.py")},
}
(out / "fresh-verification.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "official-stdout.txt").write_text(official.stdout)
print(json.dumps({"independent": summary["independent"], "official": summary["official"],
                  "solution_sha256": before_hash, "independent_seconds": independent_seconds}, indent=2))
assert independent.ok and official.ok
