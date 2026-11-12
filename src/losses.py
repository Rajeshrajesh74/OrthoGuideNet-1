"""Boundary-aware adaptive loss (class-weighted, Kendall-style task weighting)."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import IGNORE


def class_weights(train_names, cfg):
    """Gentle inverse-frequency class weights from the training labels."""
    n_cls = cfg.n_classes
    cnt = np.zeros(n_cls)
    for n in train_names:
        lab = np.load(f"{cfg.paths.cache_dir}/{n}_label.npy")
        cnt += np.bincount(lab[lab != IGNORE].ravel(), minlength=n_cls)[:n_cls]
    w = 1 / np.sqrt(np.maximum(cnt, 1) / cnt.sum())
    w = np.clip(w / w.mean(), 0.2, 5.0)
    print("class weights:", dict(zip(cfg.class_names, w.round(2).tolist())))
    return torch.tensor(w, dtype=torch.float32, device=cfg.device)


def soft_dice(prob, target_onehot, valid, eps=1e-6):
    prob, target_onehot = prob * valid, target_onehot * valid
    inter = (prob * target_onehot).sum((0, 2, 3))
    den = prob.sum((0, 2, 3)) + target_onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def ortho_loss(model, seg, bnd, y, bd, cls_w, n_classes, lam_b=4.0):
    valid = (y != IGNORE).unsqueeze(1).float()
    y_safe = y.clone(); y_safe[y == IGNORE] = 0
    pix_w = 1 + lam_b * bd                                            # up-weight tissue interfaces
    ce = (F.cross_entropy(seg, y_safe, weight=cls_w, reduction="none") * pix_w * valid[:, 0]).sum() \
        / (pix_w * valid[:, 0]).sum().clamp(min=1)
    onehot = F.one_hot(y_safe, n_classes).permute(0, 3, 1, 2).float()
    l_seg = ce + soft_dice(F.softmax(seg, 1), onehot, valid)
    pos_w = ((1 - bd).sum() / bd.sum().clamp(min=1)).clamp(1, 20)
    bce = F.binary_cross_entropy_with_logits(bnd[:, 0], bd, pos_weight=pos_w, reduction="none")
    l_bnd = (bce * valid[:, 0]).sum() / valid.sum().clamp(min=1) \
        + soft_dice(torch.sigmoid(bnd), bd.unsqueeze(1), valid)
    s = model.log_vars                                                # Kendall-style adaptive task weights
    total = torch.exp(-s[0]) * l_seg + s[0] + torch.exp(-s[1]) * l_bnd + s[1]
    return total + model.asbs.regulariser(), l_seg.item(), l_bnd.item()
