"""Train OrthoGuideNet.

Usage:
    python -m src.train --config configs/default.yaml --data-root /path/to/data --out-dir outputs
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import add_common_args, config_from_args, set_seed
from .dataset import HSIDataset
from .losses import class_weights, ortho_loss
from .metrics import confusion, scores
from .model import build_model
from .preprocessing import (compute_norm_stats, prepare_splits,
                            print_class_coverage)


@torch.no_grad()
def validate(model, val_cache, cfg):
    """Score every pixel of every validation image with tiled inference (deterministic)."""
    P, device, n_cls = cfg.data.patch, cfg.device, cfg.n_classes
    model.eval()
    cm = np.zeros((n_cls, n_cls), np.int64)
    for cube, lab in val_cache:
        H, W, _ = cube.shape
        ph, pw = (-H) % P, (-W) % P
        x = torch.from_numpy(cube.transpose(2, 0, 1).astype(np.float32))[None]
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        Hp, Wp = x.shape[-2:]
        pred = torch.zeros(Hp, Wp, dtype=torch.long)
        for r in range(0, Hp, P):
            for c in range(0, Wp, P):
                seg, _ = model(x[..., r:r + P, c:c + P].to(device))
                pred[r:r + P, c:c + P] = seg.argmax(1)[0].cpu()
        cm += confusion(pred[:H, :W].numpy(), lab, n_cls)
    iou, dice, miou, mdice = scores(cm)
    return iou, dice, (0.0 if np.isnan(miou) else miou), (0.0 if np.isnan(mdice) else mdice)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(ap)
    cfg = config_from_args(ap.parse_args())
    tr = cfg.training
    set_seed(tr.seed)
    print("device:", cfg.device)

    # ---- data ----
    train_names, val_names, test_names = prepare_splits(cfg)
    mu, sd = compute_norm_stats(train_names, cfg)
    print_class_coverage(train_names + val_names + test_names, cfg)
    train_dl = DataLoader(HSIDataset(train_names, cfg, mu, sd, True, tr.iters_per_epoch),
                          tr.batch_size, num_workers=0, drop_last=True)

    # cache normalised validation cubes once (float16 to save RAM)
    val_cache = []
    for n in val_names:
        cube = np.load(f"{cfg.paths.cache_dir}/{n}_cube.npy", mmap_mode="r")
        cube = ((np.asarray(cube, np.float32) - mu) / sd).astype(np.float16)
        val_cache.append((cube, np.load(f"{cfg.paths.cache_dir}/{n}_label.npy")))
    print("validation images:", len(val_cache))

    # ---- model, loss, optimiser ----
    model = build_model(len(mu), cfg).to(cfg.device)
    print("params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
    cls_w = class_weights(train_names, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tr.lr, weight_decay=tr.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr.lr, total_steps=tr.epochs * len(train_dl), pct_start=0.1)

    # ---- training loop ----
    ckpt = cfg.paths.ckpt_path
    best, hist = -1, []
    for ep in range(tr.epochs):
        model.train()
        model.asbs.tau = max(0.1, 5.0 * (0.1 / 5.0) ** (ep / max(1, tr.epochs - 1)))   # anneal 5.0 -> 0.1
        run = [0, 0, 0]; t0 = time.time()
        for x, y, bd in tqdm(train_dl, leave=False):
            x, y, bd = x.to(cfg.device).float(), y.to(cfg.device), bd.to(cfg.device).float()
            seg, bnd = model(x)
            loss, ls, lb = ortho_loss(model, seg, bnd, y, bd, cls_w, cfg.n_classes, tr.boundary_lambda)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            run[0] += loss.item(); run[1] += ls; run[2] += lb
        iou, dice, miou, mdice = validate(model, val_cache, cfg)
        n = len(train_dl); hist.append((run[0] / n, miou))
        print(f"ep {ep + 1:03d} loss {run[0] / n:.3f} seg {run[1] / n:.3f} bnd {run[2] / n:.3f} | "
              f"val mIoU {miou:.3f} mDice {mdice:.3f} | tau {model.asbs.tau:.2f} | {time.time() - t0:.0f}s")
        state = {"model": model.state_dict(), "mu": mu, "sd": sd}
        torch.save(state, ckpt.replace(".pt", "_last.pt"))            # always keep the latest
        if miou > best:                                              # best-so-far (saved from epoch 1)
            best = miou
            torch.save(state, ckpt)

    print("best val mIoU:", best)
    print("selected bands:", model.asbs.selected())
    print("per-class val IoU:", {cfg.class_names[i]: round(float(iou[i]), 3) for i in range(cfg.n_classes)})
    print("best checkpoint:", ckpt)


if __name__ == "__main__":
    main()
