"""Create the uncertainty-only figure (with title and colorbar) from 5-panel result images.

Usage:
    python -m src.uncertainty_figure INPUT_DIR OUTPUT_DIR
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .split_panels import EXTS, clean_name, find_panels, tight_rows


def uncertainty_panel(img):
    arr = np.asarray(img.convert("RGB"))
    cols = find_panels(arr)
    if len(cols) < 5:
        return None
    x0, x1 = cols[4]
    y0, y1 = tight_rows(arr[:, x0:x1])
    return arr[y0:y1, x0:x1]


def save_with_colorbar(panel, out_path, title="OrthoGuideNet predictive uncertainty"):
    h, w, _ = panel.shape
    fig = plt.figure(figsize=(w / 100 * 1.25, h / 100 * 1.3), dpi=200)
    ax = fig.add_axes([0.02, 0.04, 0.80, 0.84])
    ax.imshow(panel)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    cax = fig.add_axes([0.86, 0.04, 0.035, 0.84])
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, cax=cax).set_label("Predicted uncertainty", fontsize=12)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    files = []
    for root, _, names in os.walk(a.input_dir):
        if os.path.abspath(root).startswith(os.path.abspath(a.output_dir)):
            continue
        files += [os.path.join(root, n) for n in names if n.lower().endswith(EXTS)]
    for f in sorted(files):
        panel = uncertainty_panel(Image.open(f))
        if panel is None:
            print("skipped (not a 5-panel figure):", f)
            continue
        base = clean_name(f)
        save_with_colorbar(panel, os.path.join(a.output_dir, f"{base}_uncertainty.png"))
        print("saved", base)


if __name__ == "__main__":
    main()
