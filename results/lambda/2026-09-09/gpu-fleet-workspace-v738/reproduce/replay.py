"""Replay one frozen CUDA packing variant; run in a CUDA-capable Python environment."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive", type=Path)
parser.add_argument("--variant", choices=("v735",), default="v735")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--node-cap", type=int, default=200000)
parser.add_argument("--prefix-bits", type=int, default=8)
parser.add_argument("--exchange-rounds", type=int, default=16)
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
if args.repeats < 1 or args.output.exists():
    parser.error("repeats must be positive and output must not already exist")
with tempfile.TemporaryDirectory(prefix="gtoc12-fleet-") as directory:
    root = Path(directory)
    with tarfile.open(args.archive) as archive:
        manifest = json.load(archive.extractfile("FILES.json"))
        names = [m.name for m in archive.getmembers()]
        if len(names) != len(set(names)) or set(names) != set(manifest) | {"FILES.json"}:
            raise ValueError("archive member mismatch")
        for name, entry in manifest.items():
            if not (name.startswith(args.variant + "/repo/")
                    or name == args.variant + "/final/libspacepdhcg_cuda.so"
                    or name == "input/pool.json"):
                continue
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("invalid archive path")
            data = archive.extractfile(name).read()
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("archive hash mismatch: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    sys.meta_path = [f for f in sys.meta_path
                     if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
    sys.path.insert(0, str(root / args.variant / "repo/src"))
    os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"] = str(
        root / args.variant / "final/libspacepdhcg_cuda.so")
    from spacepdhcg.gtoc12.cooperative import FleetColumn
    from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace

    payload = json.loads((root / "input/pool.json").read_text())
    columns = [FleetColumn(r["identifier"], r["ship_id"], r["label"],
                           *({int(k): v for k, v in r[key].items()}
                             for key in ("deploys", "collects", "foreign", "mass")), True)
               for r in payload["rows"]]
    warm = tuple(c for c in columns if c.identifier in payload["warm"])
    weights = {int(k): v for k, v in payload["weights"].items()}
    report = dict(variant=args.variant, node_cap=args.node_cap,
                  prefix_bits=args.prefix_bits, rows=[], physics_reverified=False)
    with (Path.home() / ".spacepdhcg-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with CudaFleetWorkspace(columns, weights=weights, prefix_bits=args.prefix_bits) as workspace:
            report["setup_seconds"] = workspace.setup_seconds
            for repeat in range(args.repeats):
                started = time.perf_counter()
                result = workspace.solve(incumbent=warm,
                                         node_cap=args.node_cap, exchange_rounds=args.exchange_rounds)
                report["rows"].append(dict(repeat=repeat, seconds=time.perf_counter()-started,
                                           native_seconds=result.native_seconds,
                                           nodes=result.nodes, objective=result.objective,
                                           exchange_proposals=result.exchange_proposals,
                                           exchange_moves=result.exchange_moves,
                                           exhaustive=result.exhaustive,
                                           selected=[c.identifier for c in result.selected]))
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))
