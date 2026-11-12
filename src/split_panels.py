"""Split the 5-panel result figures into separate images.

Panels (left to right): composite, ground_truth, prediction, boundaries, uncertainty.

Usage:
    python -m src.split_panels INPUT_DIR OUTPUT_DIR
"""
import argparse
import os

import numpy as np
from PIL import Image

PANEL_NAMES = ["composite", "ground_truth", "prediction", "boundaries", "uncertainty"]
BG = 250  # pixels brighter than this (all channels) count as white background
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def find_panels(arr, n=5, min_width=300):
    """Return (x0, x1) column ranges of the n image panels."""
    h, w, _ = arr.shape
    band = arr[int(h * 0.40):int(h * 0.80)]  # middle band: avoids the overlapping titles
    has = ((band < BG).any(axis=2).sum(axis=0)) > 5
    runs, s = [], None
    for x, v in enumerate(has):
        if v and s is None:
            s = x
        elif not v and s is not None:
            if x - s >= min_width:
                runs.append((s, x))
            s = None
    if s is not None and w - s >= min_width:
        runs.append((s, w))
    return runs[:n]


def tight_rows(sub):
    """Rows occupied by the (rectangular) image, excluding title and white margins."""
    dense = (sub < BG).any(axis=2).mean(axis=1) > 0.9
    rows = np.where(dense)[0]
    return rows[0], rows[-1] + 1


def clean_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace(" ", "_").replace("(", "").replace(")", "")


def split_one(path, out_root):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    base = clean_name(path)
    cols = find_panels(arr)
    if len(cols) != 5:
        print(f"skipped {base}: found {len(cols)} panels instead of 5")
        return False
    out_dir = os.path.join(out_root, base)
    os.makedirs(out_dir, exist_ok=True)
    for (x0, x1), name in zip(cols, PANEL_NAMES):
        y0, y1 = tight_rows(arr[:, x0:x1])
        img.crop((x0, y0, x1, y1)).save(os.path.join(out_dir, f"{base}_{name}.png"))
    print("done", base)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    a = ap.parse_args()
    files = []
    for root, _, names in os.walk(a.input_dir):
        if os.path.abspath(root).startswith(os.path.abspath(a.output_dir)):
            continue
        files += [os.path.join(root, n) for n in names if n.lower().endswith(EXTS)]
    print(f"Found {len(files)} images in {a.input_dir}")
    ok = sum(split_one(f, a.output_dir) for f in sorted(files))
    print(f"Split {ok} of {len(files)} images -> {a.output_dir}")


if __name__ == "__main__":
    main()
