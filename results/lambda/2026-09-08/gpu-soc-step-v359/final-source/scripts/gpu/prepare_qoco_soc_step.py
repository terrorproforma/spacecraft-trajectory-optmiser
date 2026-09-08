"""Use compensated coefficients for the GPU SOC boundary step calculation."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(root: Path) -> dict:
    path = root / "src/cone.cu"
    original = path.read_text()
    signature = "__device__ QOCOFloat soc_step_length_dev("
    if original.count(signature) != 1:
        raise RuntimeError("unexpected QOCO SOC step function")
    start = original.index(signature)
    end = original.index("/**", start)
    body = original[start:end]
    if "QOCOFloat d = b * b - four * a * c;" not in body:
        raise RuntimeError("unexpected QOCO SOC quadratic coefficients")
    header = Path(__file__).resolve().parents[2] / "cpp/cuda/patches/qoco_soc_step.cuh"
    target = path.with_name(header.name)
    content = header.read_bytes()
    updated = original[:start] + '#include "qoco_soc_step.cuh"\n\n' + original[end:]
    target.write_bytes(content)
    path.write_text(updated)
    return {
        "path": str(path),
        "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
        "header_sha256": hashlib.sha256(content).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
