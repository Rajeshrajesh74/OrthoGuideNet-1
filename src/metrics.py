"""Confusion matrix and segmentation scores."""
import numpy as np

from .config import IGNORE


def confusion(pred, y, n):
    m = y != IGNORE
    return np.bincount(n * y[m].astype(int) + pred[m].astype(int), minlength=n * n).reshape(n, n)


def scores(cm):
    tp = np.diag(cm)
    iou = tp / (cm.sum(0) + cm.sum(1) - tp + 1e-9)
    dice = 2 * tp / (cm.sum(0) + cm.sum(1) + 1e-9)
    return iou, dice, np.nanmean(iou[cm.sum(1) > 0]), np.nanmean(dice[cm.sum(1) > 0])
