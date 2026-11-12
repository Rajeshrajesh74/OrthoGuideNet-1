"""Configuration loading and small shared helpers."""
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml

IGNORE = 255  # label value for unannotated / unknown-colour pixels


def _to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    return x


def add_common_args(ap):
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root", help="override paths.data_root")
    ap.add_argument("--out-dir", help="override paths.out_dir")
    ap.add_argument("--cache-dir", help="override paths.cache_dir")
    ap.add_argument("--checkpoint", help="override paths.checkpoint (file name)")
    ap.add_argument("--epochs", type=int, help="override training.epochs")
    return ap


def load_config(path="configs/default.yaml", data_root=None, out_dir=None,
                cache_dir=None, checkpoint=None, epochs=None):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if data_root:
        raw["paths"]["data_root"] = data_root
    if out_dir:
        raw["paths"]["out_dir"] = out_dir
    if cache_dir:
        raw["paths"]["cache_dir"] = cache_dir
    if checkpoint:
        raw["paths"]["checkpoint"] = checkpoint
    if epochs:
        raw["training"]["epochs"] = epochs

    cfg = _to_ns(raw)
    cfg.paths.cache_dir = cfg.paths.cache_dir or os.path.join(cfg.paths.out_dir, "cache")
    cfg.paths.ckpt_path = os.path.join(cfg.paths.out_dir, cfg.paths.checkpoint)
    os.makedirs(cfg.paths.out_dir, exist_ok=True)
    os.makedirs(cfg.paths.cache_dir, exist_ok=True)

    cfg.class_names = [c.name for c in cfg.classes]
    cfg.n_classes = len(cfg.classes)
    cfg.palette = np.array([c.color for c in cfg.classes], np.int32)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def config_from_args(args):
    return load_config(args.config, args.data_root, args.out_dir,
                       args.cache_dir, args.checkpoint, args.epochs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
