"""Preprocessing: ENVI cube -> spatial binning -> spectral smoothing -> cached .npy files."""
import glob
import os

import numpy as np
import spectral.io.envi as envi
from PIL import Image
from scipy.signal import savgol_filter
from tqdm.auto import tqdm

from .config import IGNORE


def radiometric_calibration(raw, dark, white, eps=1e-6):
    # only needed for raw cubes; the *_ref files are already reflectance
    return np.clip((raw - dark) / (white - dark + eps), 0, None)


def png_to_label(png_path, out_hw, palette):
    """Fixed colour -> class ID, so the same tissue gets the same ID in every image."""
    a = np.array(Image.open(png_path).convert("RGB")).astype(np.int32)
    d = ((a[:, :, None, :] - palette[None, None]) ** 2).sum(-1)
    ids = d.argmin(-1).astype(np.uint8)
    ids[d.min(-1) > 3 * 25 ** 2] = IGNORE                      # unknown colour -> ignored
    return np.array(Image.fromarray(ids).resize((out_hw[1], out_hw[0]), Image.NEAREST))


def preprocess_sample(folder, name, cfg):
    ds = cfg.data.downsample
    cache = cfg.paths.cache_dir
    os.makedirs(cache, exist_ok=True)
    out_c, out_l = f"{cache}/{name}_cube.npy", f"{cache}/{name}_label.npy"
    if os.path.exists(out_c) and os.path.exists(out_l):        # label is written last = sample complete
        return
    hdrs, dats = glob.glob(f"{folder}/*.hdr"), glob.glob(f"{folder}/*.dat")
    pngs = glob.glob(f"{folder}/Annotations/*.png")
    if not hdrs or not dats:
        raise FileNotFoundError("missing .hdr/.dat")
    if not pngs:
        raise FileNotFoundError("missing Annotations png")
    img = envi.open(hdrs[0], dats[0])
    H, W, B = img.nrows, img.ncols, img.nbands
    expected = H * W * B * np.dtype(img.dtype).itemsize
    if os.path.getsize(dats[0]) < expected:                    # truncated / partly uploaded file
        raise IOError(f".dat is incomplete ({os.path.getsize(dats[0])} < {expected} bytes)")
    mm = img.open_memmap(writable=False)                       # (H, W, B) lazily, no full load
    Hd, Wd = H // ds, W // ds
    cube = np.zeros((Hd, Wd, B), np.float16)
    for r0 in range(0, Hd, 32):                                # stream in row strips
        r1 = min(r0 + 32, Hd)
        blk = np.asarray(mm[r0 * ds:r1 * ds, :Wd * ds, :], np.float32)
        blk = blk.reshape(r1 - r0, ds, Wd, ds, B).mean((1, 3))  # spatial binning
        blk = np.clip(blk, 0, None)                            # (calibration already applied in *_ref)
        blk = savgol_filter(blk, cfg.data.savgol_window, cfg.data.savgol_poly, axis=-1)
        cube[r0:r1] = blk
    lab = png_to_label(pngs[0], (Hd, Wd), cfg.palette)
    if (lab != IGNORE).sum() == 0:
        raise ValueError("no valid annotations")
    if os.path.exists(out_l):
        os.remove(out_l)                                       # so a half-finished sample is redone
    np.save(out_c, cube)                                       # cube first
    np.save(out_l, lab)                                        # label last


def build_cache(cfg):
    """Preprocess every sample; return the names of the usable ones."""
    folders = sorted(d for d in glob.glob(f"{cfg.paths.data_root}/*_ref") if os.path.isdir(d))
    assert folders, f"No *_ref folders in {cfg.paths.data_root}. Check paths.data_root."
    print(len(folders), "samples found in", cfg.paths.data_root)
    names, skipped = [], []
    for d in tqdm(folders):
        n = os.path.basename(d)
        try:
            preprocess_sample(d, n, cfg)
            names.append(n)
        except Exception as e:                                 # failed upload / unusable annotation
            skipped.append(n)
            print(f"skipping {n}: {e}")
    print(len(names), "usable samples | skipped:", skipped)
    return names


def split_names(names, cfg):
    """Sample-level split (no leakage between train / val / test)."""
    rng = np.random.RandomState(cfg.training.seed)
    perm = rng.permutation(len(names))
    n_tr = int(cfg.data.train_frac * len(names))
    n_va = int(cfg.data.val_frac * len(names))
    train = [names[i] for i in perm[:n_tr]]
    val = [names[i] for i in perm[n_tr:n_tr + n_va]]
    test = [names[i] for i in perm[n_tr + n_va:]]
    print("train/val/test:", len(train), len(val), len(test))
    return train, val, test


def compute_norm_stats(train_names, cfg):
    """Global band-wise mean / std (training samples only)."""
    acc = []
    for n in train_names:
        c = np.load(f"{cfg.paths.cache_dir}/{n}_cube.npy", mmap_mode="r")
        acc.append(np.asarray(c[::4, ::4], np.float32).reshape(-1, c.shape[-1]))
    acc = np.concatenate(acc)
    mu, sd = acc.mean(0), acc.std(0) + 1e-6
    print("bands:", len(mu))
    return mu, sd


def print_class_coverage(names, cfg):
    cov = np.zeros(cfg.n_classes, int)
    for n in names:
        lab = np.load(f"{cfg.paths.cache_dir}/{n}_label.npy")
        cov += np.bincount(lab[lab != IGNORE].ravel(), minlength=cfg.n_classes)[:cfg.n_classes] > 0
    print("images containing each class:", dict(zip(cfg.class_names, cov.tolist())))


def prepare_splits(cfg):
    names = build_cache(cfg)
    return split_names(names, cfg)
