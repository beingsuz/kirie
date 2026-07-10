#!/usr/bin/env python3
"""Image metrics for the P4 corpus render gate (PIL-only, no numpy).

Subcommands:
  nonblack <img>        Print the percentage of non-black pixels (max channel
                        > 8), the same threshold the Rust screenshot path uses.
  ssim <a> <b>          Print the mean structural similarity of two images,
                        resized to a common grayscale canvas and averaged over
                        non-overlapping 8x8 windows (MSSIM). Range ~[-1, 1].

Both metrics are intentionally simple and dependency-free so the gate runs on
any box with a stock Python + Pillow. SSIM here is a *structural signal* for the
review phase, not a gate — the gate is render-without-crash + non-black.
"""

import sys
import warnings

from PIL import Image

warnings.filterwarnings("ignore")

# 8-bit SSIM stabilisation constants (Wang et al. 2004): C1=(0.01*L)^2,
# C2=(0.03*L)^2 with L=255.
C1 = (0.01 * 255) ** 2
C2 = (0.03 * 255) ** 2

# Common canvas for SSIM. Small enough for pure-Python speed, large enough to
# retain layout structure.
SSIM_W, SSIM_H = 160, 90
WIN = 8


def nonblack(path):
    im = Image.open(path).convert("RGB")
    px = im.getdata()
    total = len(px)
    if total == 0:
        return 0.0
    lit = 0
    for r, g, b in px:
        if r > 8 or g > 8 or b > 8:
            lit += 1
    return 100.0 * lit / total


def _gray_pixels(path, w, h):
    im = Image.open(path).convert("L").resize((w, h), Image.BILINEAR)
    return list(im.getdata())


def ssim(a, b):
    xa = _gray_pixels(a, SSIM_W, SSIM_H)
    xb = _gray_pixels(b, SSIM_W, SSIM_H)
    scores = []
    for by in range(0, SSIM_H - WIN + 1, WIN):
        for bx in range(0, SSIM_W - WIN + 1, WIN):
            sx = sy = sxx = syy = sxy = 0.0
            n = WIN * WIN
            for j in range(WIN):
                row = (by + j) * SSIM_W + bx
                for i in range(WIN):
                    va = xa[row + i]
                    vb = xb[row + i]
                    sx += va
                    sy += vb
                    sxx += va * va
                    syy += vb * vb
                    sxy += va * vb
            mx = sx / n
            my = sy / n
            vx = sxx / n - mx * mx
            vy = syy / n - my * my
            cxy = sxy / n - mx * my
            num = (2 * mx * my + C1) * (2 * cxy + C2)
            den = (mx * mx + my * my + C1) * (vx + vy + C2)
            scores.append(num / den if den else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def main(argv):
    if len(argv) < 2:
        print("usage: img_metrics.py {nonblack <img> | ssim <a> <b>}", file=sys.stderr)
        return 2
    cmd = argv[1]
    try:
        if cmd == "nonblack" and len(argv) == 3:
            print(f"{nonblack(argv[2]):.2f}")
            return 0
        if cmd == "ssim" and len(argv) == 4:
            print(f"{ssim(argv[2], argv[3]):.4f}")
            return 0
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("usage: img_metrics.py {nonblack <img> | ssim <a> <b>}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
