"""Make the GPU evidence scripts target the local GPU architecture (default unchanged: sm_120)."""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
changed = []


def edit(rel, transforms):
    path = root / rel
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new, count in transforms:
        if old not in text:
            raise SystemExit(f"{rel}: pattern not found: {old!r}")
        text = text.replace(old, new, count if count else -1)
    if text != original:
        path.write_text(text, encoding="utf-8")
        changed.append(rel)


arch_old = "-DCMAKE_CUDA_ARCHITECTURES=120"
arch_new = '"-DCMAKE_CUDA_ARCHITECTURES=${SPACEPDHCG_CUDA_ARCHITECTURES:-120}"'

for script in ("scripts/gpu/run_g2_evidence.sh", "scripts/gpu/run_g3_evidence.sh"):
    edit(
        script,
        [
            (arch_old, arch_new, 0),
            (
                "    printf 'source_commit=%s\\n' \"$(git rev-parse HEAD)\"\n",
                "    printf 'source_commit=%s\\n' \"$(git rev-parse HEAD)\"\n"
                "    printf 'cuda_architectures=%s\\n' \"${SPACEPDHCG_CUDA_ARCHITECTURES:-120}\"\n",
                1,
            ),
        ],
    )

# G3: nsys stats must never reuse a stale .sqlite export (see the b6afb49 reseal note).
edit(
    "scripts/gpu/run_g3_evidence.sh",
    [
        (
            "    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \\\n",
            "    nsys stats --force-export=true \\\n"
            "    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \\\n",
            1,
        )
    ],
)

edit(
    "scripts/gpu/checkout_build_qoco_gpu.sh",
    [
        ("  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \\\n  -DCMAKE_CUDA_ARCHITECTURES=120 \\\n",
         '  -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}" \\\n'
         '  "-DCMAKE_CUDA_ARCHITECTURES=${SPACEPDHCG_CUDA_ARCHITECTURES:-120}" \\\n', 1),
    ],
)

edit(
    "scripts/gpu/run_g3_h1.py",
    [
        ('            "hardware_id": "local-rtx-5090",\n',
         '            "hardware_id": os.environ.get("SPACEPDHCG_HARDWARE_ID", "local-rtx-5090"),\n', 1),
    ],
)

print("changed:", changed)
