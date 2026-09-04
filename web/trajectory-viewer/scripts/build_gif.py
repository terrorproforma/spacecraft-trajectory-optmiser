"""Assemble the browser-check frame sequence into an animated GIF preview.

Usage:
    python scripts/build_gif.py [--frames test-artifacts/gtoc12-3d-frame-*.png]
        [--output test-artifacts/gtoc12-3d-preview.gif] [--width 800] [--duration-ms 700]

The frames are written by ``scripts/browser-check.cjs`` (whole fleet, 30 degree oblique preset,
6x vertical exaggeration, ten epochs from 2035 to 2050). Pillow is required; the script exits
with a clear message when it is unavailable so the check never fails because of it.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--frames", default=str(ROOT / "test-artifacts" / "gtoc12-3d-frame-*.png"), help="glob"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "test-artifacts" / "gtoc12-3d-preview.gif"
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--duration-ms", type=int, default=700)
    args = parser.parse_args()
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - environment dependent
        print("Pillow is not installed; skipping GIF assembly", file=sys.stderr)
        return 0
    paths = sorted(glob.glob(args.frames))
    if not paths:
        print(f"no frames match {args.frames}", file=sys.stderr)
        return 1
    frames = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            height = round(rgb.height * args.width / rgb.width)
            frames.append(
                rgb.resize((args.width, height), Image.LANCZOS).quantize(
                    colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG
                )
            )
    durations = [args.duration_ms] * len(frames)
    durations[-1] = args.duration_ms * 2  # linger on the final frame before looping
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
