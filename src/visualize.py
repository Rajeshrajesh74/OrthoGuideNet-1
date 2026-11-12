"""Result figures."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .config import IGNORE


def class_cmap(cfg):
    palette = [tuple(np.array(c) / 255) for c in cfg.palette]
    legend = [Patch(color=palette[i], label=f"{i}: {cfg.class_names[i]}") for i in range(cfg.n_classes)]
    return ListedColormap(palette), legend


def selected_band_composite(cube, sel):
    """False-colour composite from three of the bands chosen by ASBS."""
    idx3 = [sel[-1], sel[len(sel) // 2], sel[0]]
    rgb = cube[..., idx3]
    return np.clip((rgb - rgb.min((0, 1))) / (np.ptp(rgb, (0, 1)) + 1e-6), 0, 1)


def plot_sample(name, cube, lab, pred, pred_s, bnd, ent, sel, cfg, out_path):
    cmap, legend = class_cmap(cfg)
    kw = dict(cmap=cmap, vmin=0, vmax=cfg.n_classes - 1)
    fig, ax = plt.subplots(1, 6, figsize=(26, 4.8))
    ax[0].imshow(selected_band_composite(cube, sel)); ax[0].set_title("Composite")
    ax[1].imshow(np.where(lab == IGNORE, np.nan, lab), **kw); ax[1].set_title("Ground truth")
    ax[2].imshow(pred, **kw); ax[2].set_title("Prediction (raw)")
    ax[3].imshow(pred_s, **kw); ax[3].set_title("Prediction (smoothed)")
    ax[4].imshow(bnd, cmap="gray"); ax[4].set_title("Boundaries")
    im = ax[5].imshow(ent, cmap="inferno"); ax[5].set_title("Uncertainty")
    plt.colorbar(im, ax=ax[5], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.legend(handles=legend, loc="lower center", ncol=cfg.n_classes)
    plt.suptitle(name)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
