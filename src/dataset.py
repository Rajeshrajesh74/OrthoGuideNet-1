"""Patch dataset with boundary targets."""
import random

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from .config import IGNORE


def make_boundary(y):
    v = y != IGNORE
    b = np.zeros(y.shape, bool)
    d = (y[1:] != y[:-1]) & v[1:] & v[:-1]; b[1:] |= d; b[:-1] |= d
    d = (y[:, 1:] != y[:, :-1]) & v[:, 1:] & v[:, :-1]; b[:, 1:] |= d; b[:, :-1] |= d
    return binary_dilation(b, iterations=1).astype(np.float32)


class HSIDataset(Dataset):
    def __init__(self, names, cfg, mu, sd, train=True, n_iter=200):
        self.names, self.train, self.n_iter = names, train, n_iter
        self.patch, self.mu, self.sd = cfg.data.patch, mu, sd
        cache = cfg.paths.cache_dir
        self.cubes = {n: np.load(f"{cache}/{n}_cube.npy", mmap_mode="r") for n in names}
        self.labels = {n: np.load(f"{cache}/{n}_label.npy") for n in names}

    def __len__(self):
        return self.n_iter if self.train else len(self.names)

    def __getitem__(self, i):
        P = self.patch
        n = random.choice(self.names) if self.train else self.names[i]
        cube, lab = self.cubes[n], self.labels[n]
        H, W = lab.shape
        for _ in range(10):                                    # prefer crops containing labels
            r, c = random.randint(0, H - P), random.randint(0, W - P)
            y = lab[r:r + P, c:c + P]
            if (y != IGNORE).mean() > 0.2:
                break
        x = (np.asarray(cube[r:r + P, c:c + P], np.float32) - self.mu) / self.sd
        if self.train:
            k = random.randint(0, 3); x = np.rot90(x, k); y = np.rot90(y, k)
            if random.random() < .5:
                x, y = x[:, ::-1], y[:, ::-1]
            x = x * (1 + 0.02 * np.random.randn(1, 1, x.shape[-1])) + 0.02 * np.random.randn(*x.shape)  # spectral jitter
        y = np.ascontiguousarray(y)
        bd = make_boundary(y)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(y.astype(np.int64)), torch.from_numpy(bd))
