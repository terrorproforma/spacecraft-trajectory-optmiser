"""Add opt-in Ruiz equilibration that preserves objective magnitude."""

# ruff: noqa: E501 -- CUDA replacement literals match the pinned vendor sources.

import argparse
import hashlib
import json
from pathlib import Path

FLAG = "SPACEPDHCG_TEST_QOCO_RUIZ_PRESERVE_OBJECTIVE"


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise RuntimeError(f"unexpected QOCO objective scaling site: {before}")
    return text.replace(before, after)


def prepare(root: Path) -> dict:
    """Patch frozen prepared sources; workspace policy is immutable in graphs."""
    host = root / "src/equilibration.c"
    device = root / "algebra/cuda/qoco_device_update.cuh"
    originals = {path: path.read_text() for path in [host, device]}
    cpu = replace_once(originals[host], '#include "equilibration.h"',
                       '#include "equilibration.h"\n#include <stdlib.h>')
    cpu = replace_once(cpu, "  QOCOFloat g = 1.0;", f'''  const char* objective_policy = getenv("{FLAG}");
  const int preserve_objective = objective_policy && objective_policy[0] == '1';
  QOCOFloat g = 1.0;''')
    cpu = replace_once(cpu, "    g = g > 1e-15 ? 1.0 / g : 1.0;",
                       "    g = preserve_objective ? 1.0 : (g > 1e-15 ? 1.0 / g : 1.0);")
    gpu = replace_once(originals[device], "    QOCOSolver* solver{};",
                       "    QOCOSolver* solver{};\n    bool preserve_objective{};")
    gpu = replace_once(gpu, "        auto w = std::make_unique<Context>(); w->solver = solver;", f'''        auto w = std::make_unique<Context>(); w->solver = solver;
        const char* objective_policy = getenv("{FLAG}");
        w->preserve_objective = objective_policy && objective_policy[0] == '1';''')
    gpu = replace_once(gpu, "__global__ void cost_finish(const Pair* partial, int count, int n, double* factors) {",
                       "__global__ void cost_finish(const Pair* partial, int count, int n, double* factors, bool preserve_objective) {")
    gpu = replace_once(gpu, "factors[0] = equilibration_reciprocal(fmax(shared[0].sum / n, shared[0].maximum));",
                       "factors[0] = preserve_objective ? 1.0 : equilibration_reciprocal(fmax(shared[0].sum / n, shared[0].maximum));")
    anchor = "(w->partial.data, w->blocks, n, w->factors.data);"
    if gpu.count(anchor) not in (1, 2, 3):
        raise RuntimeError("unexpected synchronous/queued Ruiz launch sites")
    gpu = gpu.replace(anchor, "(w->partial.data, w->blocks, n, w->factors.data, w->preserve_objective);")
    gpu += '''
// Optional capability: report the actual immutable numerical-update policy.
extern "C" int qoco_gpu_numeric_preserves_objective(const void* opaque) {
    const auto* context = static_cast<const qoco_device_update::Context*>(opaque);
    return context && context->preserve_objective ? 1 : 0;
}
'''
    outputs = {host: cpu, device: gpu}
    manifest = {}
    for path, updated in outputs.items():
        path.write_text(updated)
        manifest[str(path.relative_to(root))] = {
            "before_sha256": hashlib.sha256(originals[path].encode()).hexdigest(),
            "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"flag": FLAG, "default_enabled": False, "files": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
