"""Add device-selected conditioning to the existing cold retry, without extra solves."""

# ruff: noqa: E501 -- exact prepared CUDA source replacements
import argparse
import hashlib
import json
from pathlib import Path


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise RuntimeError(f"unexpected retry-conditioning site: {before}")
    return text.replace(before, after)


def prepare(root: Path):
    path = root / "algebra/cuda/qoco_device_update.cuh"
    original = path.read_text()
    text = replace(
        original,
        "    bool preserve_objective{};",
        "    bool preserve_objective{};\n    const int* conditioning_retry{};",
    )
    text = replace(
        text,
        "const double* previous_kinv = nullptr) {",
        "const double* previous_kinv = nullptr, const double* incoming_c = nullptr) {",
    )
    text = replace(
        text,
        "c[i] = __dmul_rn(__dmul_rn(kinv, c[i]), s.di[i]);",
        "c[i] = incoming_c ? incoming_c[i] : __dmul_rn(__dmul_rn(kinv, c[i]), s.di[i]);",
    )
    # Uniform device guards preserve barrier convergence. Only the first of the
    # two existing inaccurate retries receives five ordinary Ruiz passes.
    signatures = [
        "void norms(Matrix a, Matrix p, Matrix g, Scales scales, double* p_norm, int n, int eq, int cone)",
        "void cost_partial(const double* p_norm, const double* c, int n, Pair* out)",
        "void cost_finish(const Pair* partial, int count, int n, double* factors, bool preserve_objective)",
        "void cone_scales(double* f, const int* starts, int count)",
        "void scale_matrix(Matrix a, const double* row, const double* column, const double* cost)",
        "void accumulate_scales(Scales s, double* c, int n, int p, int m, const double* factors)",
    ]
    for signature in signatures:
        text = replace(
            text,
            signature + " {",
            signature[:-1]
            + ", const int* retry = nullptr) {\n    if (retry && *retry != 1) return;",
        )
    text = replace(
        text,
        "for (int iteration = 0; iteration < solver->settings->ruiz_iters; ++iteration)",
        "for (int iteration = 0; iteration < (w->conditioning_retry ? 5 : solver->settings->ruiz_iters); ++iteration)",
        3,
    )
    calls = [
        (
            "w->scale_valid ? w->result.data + 1 : nullptr);",
            "w->scale_valid ? w->result.data + 1 : nullptr, w->conditioning_retry ? vectors : nullptr);",
        ),
        ("w->p_norm.data, n, p, m);", "w->p_norm.data, n, p, m, w->conditioning_retry);"),
        (
            "data->c->d_data, n, w->partial.data);",
            "data->c->d_data, n, w->partial.data, w->conditioning_retry);",
        ),
        (
            "w->factors.data, w->preserve_objective);",
            "w->factors.data, w->conditioning_retry ? false : w->preserve_objective, w->conditioning_retry);",
        ),
        (
            "w->cone_starts.data, w->cone_count);",
            "w->cone_starts.data, w->cone_count, w->conditioning_retry);",
        ),
        (
            "w->p, scales.delta, scales.delta, w->factors.data);",
            "w->p, scales.delta, scales.delta, w->factors.data, w->conditioning_retry);",
        ),
        (
            "w->a, scales.delta + n, scales.delta, nullptr);",
            "w->a, scales.delta + n, scales.delta, nullptr, w->conditioning_retry);",
        ),
        (
            "w->g, scales.delta + n + p, scales.delta, nullptr);",
            "w->g, scales.delta + n + p, scales.delta, nullptr, w->conditioning_retry);",
        ),
        (
            "data->c->d_data, n, p, m, w->factors.data);",
            "data->c->d_data, n, p, m, w->factors.data, w->conditioning_retry);",
        ),
    ]
    for before, after in calls:
        text = replace(text, before, after, 3)
    text += """
// Borrowed device retry counter; caller drains work before rebinding or freeing.
// Ruiz > 0 has an explicitly selected policy and is not overridden.
extern "C" int qoco_gpu_numeric_retry_conditioning(void* opaque, const int* retry) {
    auto* w = static_cast<qoco_device_update::Context*>(opaque);
    int device = -1;
    if (!w || cudaGetDevice(&device) != cudaSuccess || device != w->device ||
        w->owner != std::this_thread::get_id() || w->pending ||
        (retry && w->solver->settings->ruiz_iters != 0)) return 1;
    w->conditioning_retry = retry;
    return 0;
}
"""
    # All layout validation precedes the only mutation.
    path.write_text(text)
    return {
        "files": {
            str(path.relative_to(root)): {
                "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
        "retry_policy": "five Ruiz passes only when borrowed device counter equals one",
        "extra_solver_attempts": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
