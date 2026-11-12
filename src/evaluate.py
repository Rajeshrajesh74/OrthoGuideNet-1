"""Evaluate a trained checkpoint on the held-out test samples and save result figures.

Usage:
    python -m src.evaluate --config configs/default.yaml --data-root /path/to/data --out-dir outputs
    python -m src.evaluate ... --ablation        # also compare inference variants
"""
import argparse
import os

import numpy as np
import torch

from .config import IGNORE, add_common_args, config_from_args, set_seed
from .inference import predict_cube, smooth_pred
from .metrics import confusion, scores
from .model import build_model
from .preprocessing import prepare_splits
from .visualize import plot_sample


def load_normalised(name, ck, cfg):
    cube = np.load(f"{cfg.paths.cache_dir}/{name}_cube.npy", mmap_mode="r")
    cube = (np.asarray(cube, np.float32) - ck["mu"]) / ck["sd"]
    return cube.astype(np.float32), np.load(f"{cfg.paths.cache_dir}/{name}_label.npy")


def eval_variant(model, ck, test_names, cfg, **kw):
    cm = np.zeros((cfg.n_classes, cfg.n_classes), np.int64)
    for n in test_names:
        cube, lab = load_normalised(n, ck, cfg)
        pred, _, _, _ = predict_cube(model, cube, cfg, **kw)
        cm += confusion(pred, lab, cfg.n_classes)
    _, _, miou, mdice = scores(cm)
    return miou, mdice, np.diag(cm).sum() / cm.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--ablation", action="store_true", help="compare inference variants (tiling / TTA / smoothing)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.training.seed)
    n_cls = cfg.n_classes

    _, _, test_names = prepare_splits(cfg)
    ck = torch.load(cfg.paths.ckpt_path, map_location=cfg.device, weights_only=False)
    model = build_model(len(ck["mu"]), cfg).to(cfg.device)
    model.load_state_dict(ck["model"]); model.eval()

    # ---- test-set metrics (deterministic: T=1) ----
    cm = np.zeros((n_cls, n_cls), np.int64); store = {}
    for n in test_names:
        cube, lab = load_normalised(n, ck, cfg)
        pred, bnd, ent, _ = predict_cube(model, cube, cfg, T=1)
        cm += confusion(pred, lab, n_cls); store[n] = (cube, lab, pred, bnd, ent)
    iou, dice, miou, mdice = scores(cm)
    print(f"TEST  mIoU {miou:.4f}  mDice {mdice:.4f}  OA {np.diag(cm).sum() / cm.sum():.4f}")
    for i, nm in enumerate(cfg.class_names):
        print(f"  {nm:18s} IoU {iou[i]:.3f}  Dice {dice[i]:.3f}")

    # ---- raw vs smoothed predictions ----
    size = cfg.inference.post_smooth_size
    cm_s = np.zeros_like(cm)
    for n, (_, lab, pred, _, _) in store.items():
        cm_s += confusion(smooth_pred(pred, n_cls, size), lab, n_cls)
    _, _, miou_s, mdice_s = scores(cm_s)
    print(f"smoothed  mIoU {miou_s:.4f}  mDice {mdice_s:.4f}  OA {np.diag(cm_s).sum() / cm_s.sum():.4f}")

    # ---- optional ablation of inference options ----
    if args.ablation:
        variants = {
            "plain tiles":               dict(stride=cfg.data.patch, tta=False, smooth_sigma=0, min_area=0),
            "overlap":                   dict(tta=False, smooth_sigma=0, min_area=0),
            "overlap + flip TTA":        dict(tta=True, smooth_sigma=0, min_area=0),
            "overlap + TTA + smoothing": dict(tta=True, smooth_sigma=cfg.inference.smooth_sigma,
                                              min_area=cfg.inference.min_area),
        }
        for tag, kw in variants.items():
            m_iou, m_dice, oa = eval_variant(model, ck, test_names, cfg, **kw)
            print(f"{tag:28s} mIoU {m_iou:.4f}  mDice {m_dice:.4f}  OA {oa:.4f}")

    # ---- figures ----
    fig_dir = os.path.join(cfg.paths.out_dir, "figures")
    sel = model.asbs.selected()
    for n, (cube, lab, pred, bnd, ent) in store.items():
        plot_sample(n, cube, lab, pred, smooth_pred(pred, n_cls, size), bnd, ent, sel, cfg,
                    os.path.join(fig_dir, f"{n}.png"))
    print("figures saved to", fig_dir)


if __name__ == "__main__":
    main()
