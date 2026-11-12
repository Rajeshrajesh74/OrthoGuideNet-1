"""Tiled inference with test-time augmentation, optional MC-dropout and predictive-entropy uncertainty."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, gaussian_filter, uniform_filter
from scipy.ndimage import label as cc_label


def enable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d)):
            mod.train()


def remove_small(pred, n_classes, min_area=100):
    """Relabel tiny isolated patches with the majority label of their surroundings."""
    out = pred.copy()
    for k in range(n_classes):
        lab_, n = cc_label(pred == k)
        if n == 0:
            continue
        areas = np.bincount(lab_.ravel())
        for i in np.where(areas[1:] < min_area)[0] + 1:
            m = lab_ == i
            ring = binary_dilation(m, iterations=2) & ~m
            vals = out[ring]; vals = vals[vals != k]
            if len(vals):
                out[m] = np.bincount(vals, minlength=n_classes).argmax()
    return out


def smooth_pred(pred, n_classes, size=15):
    oh = np.stack([uniform_filter((pred == k).astype(np.float32), size=size) for k in range(n_classes)])
    return oh.argmax(0)


@torch.no_grad()
def predict_cube(model, cube, cfg, T=1, stride=None, tta=None, smooth_sigma=None, min_area=None):
    """cube: (H, W, B) normalised float32.  T>1 adds MC-dropout passes.

    Arguments left as None take their defaults from cfg.inference.
    Returns (prediction, boundary map, predictive entropy, mutual information).
    """
    P, n_cls, device = cfg.data.patch, cfg.n_classes, cfg.device
    inf = cfg.inference
    stride = stride if stride is not None else (inf.stride or P // 2)
    tta = inf.tta if tta is None else tta
    smooth_sigma = inf.smooth_sigma if smooth_sigma is None else smooth_sigma
    min_area = inf.min_area if min_area is None else min_area

    H, W, _ = cube.shape
    Hp = max(P, math.ceil((H - P) / stride) * stride + P)
    Wp = max(P, math.ceil((W - P) / stride) * stride + P)
    x = F.pad(torch.from_numpy(cube.transpose(2, 0, 1))[None], (0, Wp - W, 0, Hp - H), mode="reflect")
    w1 = torch.hann_window(P, periodic=False).clamp(min=1e-2)
    win = (w1[:, None] * w1[None, :]) if stride < P else torch.ones(P, P)
    Pr = torch.zeros(n_cls, Hp, Wp); Bd = torch.zeros(Hp, Wp); E = torch.zeros(Hp, Wp); Wt = torch.zeros(Hp, Wp)
    flips = [(), (3,), (2,)] if tta else [()]
    model.eval()
    if T > 1:
        enable_dropout(model)
    for r in range(0, Hp - P + 1, stride):
        for c in range(0, Wp - P + 1, stride):
            tile = x[..., r:r + P, c:c + P].to(device)
            ps, bs = [], []
            for _ in range(T):
                for fl in flips:
                    s, b = model(tile.flip(fl) if fl else tile)
                    s, b = F.softmax(s, 1), torch.sigmoid(b)
                    if fl:
                        s, b = s.flip(fl), b.flip(fl)
                    ps.append(s[0]); bs.append(b[0, 0])
            ps = torch.stack(ps)
            ent_each = (-(ps * torch.log(ps + 1e-9)).sum(1) / math.log(n_cls)).mean(0)
            sl = (slice(r, r + P), slice(c, c + P))
            Pr[(slice(None),) + sl] += ps.mean(0).cpu() * win
            Bd[sl] += torch.stack(bs).mean(0).cpu() * win
            E[sl] += ent_each.cpu() * win; Wt[sl] += win
    Wt = Wt.clamp(min=1e-6)
    Pm = (Pr / Wt)[:, :H, :W].numpy(); Bd = (Bd / Wt)[:H, :W].numpy(); Eavg = (E / Wt)[:H, :W].numpy()
    if smooth_sigma > 0:
        Pm = np.stack([gaussian_filter(p, smooth_sigma) for p in Pm]); Pm /= Pm.sum(0, keepdims=True)
    pred = Pm.argmax(0)
    if min_area > 0:
        pred = remove_small(pred, n_cls, min_area)
    ent = -(Pm * np.log(Pm + 1e-9)).sum(0) / math.log(n_cls)
    mi = np.clip(ent - Eavg, 0, None)
    return pred, Bd, ent, mi
