# OrthoGuideNet

Hyperspectral image segmentation with boundary-aware multi-task learning and predictive-entropy uncertainty.

**Pipeline:** ASBS (adaptive band selection) → Spectral/Spatial DualMamba → ASSF → BAR → multi-task decoder
(7-class segmentation + boundary map), trained with a boundary-aware adaptive loss and AdamW.
Uncertainty is estimated as predictive entropy (MC-dropout is available through `predict_cube(..., T>1)`).

> TODO: add the paper title, a short summary and the main figure from the manuscript.

## Repository structure

```
OrthoGuideNet/
├── configs/
│   └── default.yaml          # paths, classes, model, training and inference settings
├── src/
│   ├── config.py             # config loading, seeding
│   ├── preprocessing.py      # ENVI cube → binning → smoothing → cached .npy, train/val/test split
│   ├── dataset.py            # patch dataset with boundary targets
│   ├── model.py              # OrthoGuideNet (ASBS, DualMamba, ASSF, BAR)
│   ├── losses.py             # boundary-aware adaptive loss
│   ├── metrics.py            # confusion matrix, IoU, Dice
│   ├── inference.py          # tiled inference, TTA, uncertainty
│   ├── train.py              # training entry point
│   ├── evaluate.py           # test-set evaluation and figures
│   ├── visualize.py          # result figures
│   ├── split_panels.py       # split 5-panel figures into separate images
│   └── uncertainty_figure.py # uncertainty-only figure with title and colorbar
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Python 3.9+ and a CUDA GPU (recommended for training).

```bash
git clone https://github.com/<your-username>/OrthoGuideNet.git
cd OrthoGuideNet
pip install -r requirements.txt
```

The selective state-space (Mamba-style) block is written in pure PyTorch, so `mamba_ssm` is **not** required.

## Data

> TODO: describe the dataset (source, acquisition, number of samples, annotation protocol) and how to obtain it.

Expected layout under `paths.data_root`:

```
data_root/
├── <sample>_ref/
│   ├── *.hdr, *.dat            # ENVI hyperspectral cube (reflectance)
│   └── Annotations/*.png       # colour-coded label image
└── ...
```

Annotation colours are mapped to class IDs through the `classes` list in `configs/default.yaml`
(class ID = position in the list). Pixels with an unknown colour are ignored.

## Usage

```bash
# 1) train (preprocesses and caches the cubes on the first run)
python -m src.train --config configs/default.yaml --data-root /path/to/data --out-dir outputs

# 2) evaluate on the held-out test samples and save figures to outputs/figures
python -m src.evaluate --config configs/default.yaml --data-root /path/to/data --out-dir outputs

#    add --ablation to also compare inference variants (tiling / TTA / smoothing)
```

Settings can be edited in `configs/default.yaml` or overridden with `--data-root`, `--out-dir`,
`--cache-dir`, `--checkpoint` and `--epochs`.

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<your-username>/OrthoGuideNet.git
%cd OrthoGuideNet
!pip install -q -r requirements.txt
!python -m src.train --data-root /content/drive/MyDrive/<data-folder> --out-dir /content/drive/MyDrive/<output-folder>
```

### Figure utilities

```bash
# split the 5-panel figures (composite, ground truth, prediction, boundaries, uncertainty)
python -m src.split_panels path/to/figures path/to/split_panels

# uncertainty-only figure with title and colorbar
python -m src.uncertainty_figure path/to/figures path/to/uncertainty_only
```

## Reproducibility

| Setting | Value |
|---|---|
| Random seed | 42 |
| Train / val / test split | 60 / 20 / 20 %, at sample level (no leakage) |
| Spatial down-sampling | 2 |
| Training patch | 128 × 128 |
| Bands kept by ASBS | 32 |
| Feature width | 64 |
| Batch size | 8 |
| Epochs | 60 |
| Optimiser | AdamW, lr 3e-4, weight decay 1e-2, one-cycle schedule |
| Spectral smoothing | Savitzky–Golay, window 7, order 2 |

All values are set in `configs/default.yaml`. Model weights and the dataset are not stored in this repository.

> TODO: add the link / DOI to the released weights (e.g. Zenodo).

## Citation

> TODO: add the BibTeX entry once the paper is published.

## License

Released under the MIT License (see `LICENSE`). Replace `[YOUR NAME]` with your name.
