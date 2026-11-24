"""Generates the two Kaggle training notebooks for the 1000-class comparison.

    python training_notebooks/_build_notebooks.py

Outputs (next to this file):
    train_1000_swin3d_t.ipynb          -- Video Swin-T on the cached RGB crops (GPU T4 x2)
    train_1000_landmark_models.ipynb   -- v10 Geometry Transformer + ST-GCN on cached landmarks

Shared cells (setup, input discovery, manifest) are defined once here so both notebooks
build the *identical* 1000-class subset with the same class-index mapping as the team's
existing v10 notebooks (manifest code copied verbatim from
asl-1200-class-single-stream-model-pilot-ru.ipynb, Section 3).
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


# --------------------------------------------------------------------------------------
# Shared cells
# --------------------------------------------------------------------------------------

SETUP = r'''
import csv
import glob
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from IPython.display import display
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

NOTEBOOK_START = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count()


def _env(name, default):
    # Environment overrides exist ONLY for the local CPU smoke test of this notebook.
    # On Kaggle none of these are set, so the defaults in CONFIG below are what run.
    return os.environ.get(name, default)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


print(f"Torch {torch.__version__} | device={DEVICE} | GPUs={N_GPUS}")
for i in range(N_GPUS):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
'''

RESOLVE_INPUTS = r'''
def resolve_inputs():
    """Use CONFIG paths if they exist; otherwise auto-discover the cache shards / splits
    under /kaggle/input (Kaggle mount paths change depending on how inputs are attached)."""
    env_cache = os.environ.get("ASL_CACHE_DIRS")
    if env_cache:
        CONFIG.CACHE_DIRS = [Path(p) for p in env_cache.split(os.pathsep)]
    env_splits = os.environ.get("ASL_SPLITS_DIR")
    if env_splits:
        CONFIG.SPLITS_DIR = Path(env_splits)

    if not all(p.exists() for p in CONFIG.CACHE_DIRS):
        found = sorted({Path(p) for p in glob.glob("/kaggle/input/**/feature_cache_shard_*", recursive=True)
                        if Path(p).is_dir()})
        print(f"[inputs] configured CACHE_DIRS not all present -- auto-discovered {len(found)} shard dirs:")
        for p in found:
            print("   ", p)
        CONFIG.CACHE_DIRS = found
    assert CONFIG.CACHE_DIRS, ("No feature_cache_shard_* directories found. Attach the three "
                               "asl-shared-cache-building notebooks' outputs as inputs (see top of notebook).")
    if len(CONFIG.CACHE_DIRS) != 3:
        print(f"[inputs] WARNING: expected 3 cache shards (as in the v10 notebooks), found "
              f"{len(CONFIG.CACHE_DIRS)}. The class subset may differ from the v10 runs.")

    if not (CONFIG.SPLITS_DIR / "train.csv").exists():
        hits = sorted(glob.glob("/kaggle/input/**/splits/train.csv", recursive=True))
        assert hits, "ASL Citizen splits/train.csv not found -- attach the asl-citizen dataset."
        CONFIG.SPLITS_DIR = Path(hits[0]).parent
        print(f"[inputs] auto-discovered SPLITS_DIR: {CONFIG.SPLITS_DIR}")
    print("Cache shards:", *CONFIG.CACHE_DIRS, sep="\n   ")
    print("Splits dir:  ", CONFIG.SPLITS_DIR)


resolve_inputs()
seed_everything(CONFIG.SEED)
CONFIG.OUT_DIR.mkdir(parents=True, exist_ok=True)
'''

MANIFEST_MD = r'''
## Build the manifest & apply the Top-1000 class filter

**Copied verbatim** from the team's v10 notebook (Section 3): rebuild the label space from the
official ASL Citizen `train.csv` / `val.csv` / `test.csv`, restricted to videos present in the
cache, then keep the `TOP_K_CLASSES` most-represented classes (remapped to `0..K-1`, most
represented = index 0). Same code + same cache = same 1000 classes and same index mapping as the
team's other 1000-class runs, so every model in the comparison is evaluated on the identical
signer-independent test split.

The class list is saved to `class_list.json` with a fingerprint -- both notebooks print it, and
**the fingerprints must match** for the comparison to be valid.
'''

MANIFEST = r'''
cache_index = {}
for shard_dir in CONFIG.CACHE_DIRS:
    assert shard_dir.exists(), f"Cache shard not found: {shard_dir} -- update CONFIG.CACHE_DIRS"
    for f in shard_dir.glob("*.npz"):
        cache_index[f.stem] = f
print(f"Total cached .npz files across {len(CONFIG.CACHE_DIRS)} shards: {len(cache_index)}")

split_files = {
    "train": CONFIG.SPLITS_DIR / "train.csv",
    "val": CONFIG.SPLITS_DIR / "val.csv",
    "test": CONFIG.SPLITS_DIR / "test.csv",
}
rows, all_glosses = [], set()
for subset, csv_path in split_files.items():
    assert csv_path.exists(), f"Missing split file: {csv_path}"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = Path(row["Video file"]).stem
            gloss = row["Gloss"]
            all_glosses.add(gloss)
            if video_id in cache_index:
                rows.append((video_id, gloss, subset))

class_list_full = sorted(all_glosses)
class_to_idx_full = {g: i for i, g in enumerate(class_list_full)}

nslt_data_full = {}
for video_id, gloss, subset in rows:
    nslt_data_full[video_id] = {"action": [class_to_idx_full[gloss], 0, 0], "subset": subset}

print(f"Cached rows matched against splits: {len(rows)}")
print(f"Full ASL Citizen vocabulary present in cache: {len(class_list_full)} classes")

# --- top-K filter, identical remap logic to the v10 notebooks ---
class_video_counts = Counter(v["action"][0] for v in nslt_data_full.values())
top_k_old_indices = [c for c, _ in class_video_counts.most_common(CONFIG.TOP_K_CLASSES)]
top_k_old_indices_set = set(top_k_old_indices)
old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(top_k_old_indices)}

nslt_data = {
    vid: {"action": [old_to_new_idx[meta["action"][0]], 0, 0], "subset": meta["subset"]}
    for vid, meta in nslt_data_full.items()
    if meta["action"][0] in top_k_old_indices_set
}
class_list = {old_to_new_idx[old]: class_list_full[old] for old in top_k_old_indices}
NUM_CLASSES = len(class_list)

subset_counts = Counter(v["subset"] for v in nslt_data.values())
print(f"\nTop-{CONFIG.TOP_K_CLASSES} filter applied -- NUM_CLASSES: {NUM_CLASSES}")
print(f"Videos remaining: {len(nslt_data)} across splits {dict(subset_counts)}")

per_class_counts = Counter(v["action"][0] for v in nslt_data.values())
print(f"Videos/class (post-filter, all splits combined): "
      f"min={min(per_class_counts.values())} median={int(np.median(list(per_class_counts.values())))} "
      f"max={max(per_class_counts.values())}")

# --- added: persist the exact class mapping for the web app + cross-notebook check ---
CLASS_NAMES = [class_list[i] for i in range(NUM_CLASSES)]
CLASS_FINGERPRINT = hashlib.sha1("\n".join(CLASS_NAMES).encode("utf-8")).hexdigest()[:12]
with open(CONFIG.OUT_DIR / "class_list.json", "w", encoding="utf-8") as f:
    json.dump({"num_classes": NUM_CLASSES, "fingerprint": CLASS_FINGERPRINT,
               "rule": f"ASL Citizen top-{CONFIG.TOP_K_CLASSES} most-represented glosses in cache "
                       f"(Counter.most_common), index 0 = most represented",
               "classes": CLASS_NAMES}, f, indent=1)
print(f"\nClass-list fingerprint: {CLASS_FINGERPRINT}   (must match across all 1000-class notebooks)")


def split_samples(subset):
    # Same sample construction (and order) as the v10 WLASLCachedDataset.
    s = [(vid, meta["action"][0]) for vid, meta in nslt_data.items()
         if meta["subset"] == subset and vid in cache_index]
    if CONFIG.SMOKE_TEST:
        rng = random.Random(0)
        s = rng.sample(s, min(len(s), CONFIG.SMOKE_SAMPLES[subset]))
    return s


train_samples, val_samples, test_samples = (split_samples(s) for s in ("train", "val", "test"))
print(f"Train / val / test samples: {len(train_samples)} / {len(val_samples)} / {len(test_samples)}"
      + ("   [SMOKE TEST SUBSET]" if CONFIG.SMOKE_TEST else ""))
'''

EVAL_HELPERS = r'''
def summarize_predictions(y_true, y_pred, top5, num_classes):
    labels = list(range(num_classes))
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "top1": float((y_true == y_pred).mean()),
        "top5": float(np.mean([t in row for t, row in zip(y_true, top5)])),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "weighted_precision": float(wp), "weighted_recall": float(wr), "weighted_f1": float(wf),
        "n_samples": int(len(y_true)),
    }


def save_evaluation(out_dir, tag, preds, video_ids, class_names):
    """Writes metrics-ready artifacts the web app reads directly."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = preds["y_true"], preds["y_pred"]
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    np.savez_compressed(out_dir / "confusion_matrix.npz", cm=cm.astype(np.uint16))

    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n)), zero_division=0)
    pd.DataFrame({"class_id": range(n), "class_name": class_names, "support": sup,
                  "correct": np.diag(cm), "precision": p, "recall": r, "f1": f1}
                 ).to_csv(out_dir / "per_class_metrics.csv", index=False)

    pd.DataFrame({"video_id": video_ids, "y_true": y_true, "y_pred": y_pred,
                  "confidence": preds["confidence"],
                  "top5": [" ".join(map(str, t)) for t in preds["top5"]]}
                 ).to_csv(out_dir / "test_predictions.csv", index=False)

    pairs = [(class_names[t], class_names[q], int(cm[t, q])) for t, q in zip(*np.nonzero(cm)) if t != q]
    pairs_df = pd.DataFrame(pairs, columns=["true", "pred", "errors"]).sort_values("errors", ascending=False)
    pairs_df.head(50).to_csv(out_dir / "top_confusion_pairs.csv", index=False)
    print(f"[{tag}] top confused pairs:")
    print(pairs_df.head(10).to_string(index=False))
    return cm


def plot_history(history, title, path):
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title(f"{title} -- loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, [h["train_top1"] for h in history], label="train top-1")
    ax[1].plot(ep, [h["val_top1"] for h in history], label="val top-1")
    ax[1].plot(ep, [h["val_top5"] for h in history], label="val top-5", linestyle="--")
    ax[1].set_title(f"{title} -- accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.show()


def measure_cpu_latency(model_cpu, example, n_runs=3, threads=2):
    """Single-clip CPU latency, roughly what a free 2-vCPU host sees (indicative only)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    model_cpu.eval()
    with torch.inference_mode():
        model_cpu(example)  # warm-up
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter(); model_cpu(example); times.append(time.perf_counter() - t0)
    torch.set_num_threads(prev)
    return {"cpu_threads": threads, "mean_s": float(np.mean(times)), "runs": n_runs}
'''


# --------------------------------------------------------------------------------------
# Notebook A -- Video Swin-T (RGB)
# --------------------------------------------------------------------------------------

def build_swin_notebook():
    cells = []
    cells.append(md(r'''
# ASL Citizen -- 1000 classes -- Video Swin Transformer (Swin3D-T), RGB stream
### Capstone C -- model comparison track (new architecture #1 of 2)

**What this trains:** `torchvision.models.video.swin3d_t`, Kinetics-400 pretrained, fine-tuned
end-to-end on the **cached RGB `appearance` crops** of the ASL Citizen top-1000 classes.

**Why it is in the comparison:** Capstone B's RGB model was `r3d_18` (3D CNN), which overfit badly
(1000-class dual-stream test top-1 = 0.282). Video Swin is the transformer counterpart: shifted-window
self-attention over space *and* time instead of 3x3x3 convolutions. This run answers
*"is the RGB-stream failure specific to the 3D-CNN, or does a modern video transformer hit the
same data ceiling (~15 training clips per class)?"* -- a valid result either way.

**Fair comparison:** same cache, same manifest code, same top-1000 class mapping, same
signer-independent train/val/test split, same `WeightedRandomSampler`, early stopping on `val_top1`,
test set touched once at the end -- identical protocol to the v10 notebooks.

---
### How to run on Kaggle (one-time setup, ~3 min)
1. **Create** -> New Notebook -> *File -> Import Notebook* -> upload this `.ipynb`.
2. **Add Input** (right panel) -- the same inputs as the v10 notebooks:
   - notebook outputs `asl-shared-cache-building-id00`, `-id01`, `-id02` (the 3 cache shards)
   - dataset `asl-citizen` (for `splits/`)
   Paths are auto-discovered if they differ from `CONFIG` below.
3. **Settings:** Accelerator = **GPU T4 x2**, Internet = **On** (downloads Kinetics-400 weights, ~120 MB).
4. *(Recommended)* set `SMOKE_TEST = True`, click **Run All** interactively, confirm it finishes
   (~5-10 min), then set it back to `False`.
5. **Save Version -> Save & Run All (Commit)** -- runs in the background, you can close the tab.
6. When done: **Output** tab -> download `swin3d_t_1000cls/app_export/` (+ the metrics files).

**Time budget:** estimated **3-5 h** on T4 x2 (printed live after the first ~100 steps).
A guard stops training before Kaggle's 12 h limit and still runs the test evaluation.
If a run is ever cut off: attach this notebook's previous version output as an input and set
`RESUME_FROM` to its `last.pt`.
'''))
    cells.append(md("## 1. Setup"))
    cells.append(code(SETUP + r'''
import torchvision
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.ops import StochasticDepth
print(f"torchvision {torchvision.__version__}")
'''))
    cells.append(md(r'''
## 2. Configuration
Hyperparameters were chosen for a 28M-parameter pretrained transformer on ~15 clips/class:
low backbone LR, higher head LR, warm-up + cosine, strong regularisation (stochastic depth 0.2,
head dropout 0.5, weight decay 0.05, label smoothing 0.1). Mixed precision (fp16) is ON here for
speed; the NaN guard + `GradScaler` handle the instability that made v9.1 turn AMP off.
'''))
    cells.append(code(r'''
class CONFIG:
    WORK_DIR = Path(_env("ASL_WORK_DIR", "/kaggle/working"))

    # ---- inputs (same as the v10 notebooks; auto-discovered if these paths don't exist) ----
    CACHE_DIRS = [
        Path("/kaggle/input/notebooks/aniruddha2022/asl-shared-cache-building-id00/feature_cache_shard_0"),
        Path("/kaggle/input/notebooks/redoanmozumder/asl-shared-cache-building-id02/feature_cache_shard_2"),
        Path("/kaggle/input/notebooks/pranaysahaps/asl-shared-cache-building-id01/feature_cache_shard_1"),
    ]
    SPLITS_DIR = Path("/kaggle/input/datasets/pranaysahaps/asl-citizen/splits")
    OUT_DIR = WORK_DIR / "swin3d_t_1000cls"

    TOP_K_CLASSES = int(_env("ASL_TOP_K", 1000))
    FRAME_COUNT = 16                     # cache extraction config (v10 CONFIG.FRAME_COUNT)
    APPEARANCE_STREAM = 0                # appearance[t, 0] = signer crop (as in v10 Section 11)

    # ---- model input (torchvision Kinetics-400 video preset) ----
    IMG_SIZE = int(_env("ASL_IMG_SIZE", 224))   # env override = local CPU smoke test only
    MEAN = (0.43216, 0.394666, 0.37645)
    STD = (0.22803, 0.22145, 0.216989)
    PRETRAINED = _env("ASL_PRETRAINED", "1") == "1"

    # ---- regularisation / augmentation ----
    STOCHASTIC_DEPTH = 0.2
    HEAD_DROPOUT = 0.5
    AUG_SCALE_MIN = 0.80                 # random-resized-crop area range [0.8, 1.0], same crop for all frames
    AUG_HFLIP = False                    # mirroring swaps the dominant hand -- off, matches prior runs

    # ---- optimisation ----
    BATCH_SIZE = int(_env("ASL_BATCH", 16))   # total across GPUs (8 per T4)
    NUM_WORKERS = int(_env("ASL_NUM_WORKERS", 4))
    EPOCHS = int(_env("ASL_EPOCHS", 25))
    WARMUP_EPOCHS = 1
    LR_BACKBONE = 5e-5
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_NORM = 1.0
    USE_AMP = True

    # ---- early stopping on val_top1 (same rule as v10) ----
    MIN_EPOCHS_BEFORE_STOPPING = 10
    EARLY_STOPPING_PATIENCE = 5

    # ---- Kaggle session safety ----
    TIME_BUDGET_HOURS = float(_env("ASL_TIME_BUDGET_H", 11.0))   # Kaggle hard limit is 12 h
    RESUME_FROM = None                   # e.g. Path("/kaggle/input/<prev-version>/swin3d_t_1000cls/last.pt")

    # ---- smoke test: tiny subset, 1 epoch -- verifies the whole pipeline in minutes ----
    SMOKE_TEST = _env("ASL_SMOKE_TEST", "0") == "1"
    SMOKE_SAMPLES = {"train": 256, "val": 128, "test": 128}

    SEED = 42


if CONFIG.SMOKE_TEST:
    CONFIG.EPOCHS = 1
    CONFIG.MIN_EPOCHS_BEFORE_STOPPING = 1
''' + "\n" + RESOLVE_INPUTS))
    cells.append(md(MANIFEST_MD))
    cells.append(code(MANIFEST))
    cells.append(md(r'''
## 3. Inspect the cached `appearance` tensor
The v10 notebook only shows that `appearance[t, 0]` is a displayable frame. Rather than assume the
crop size / dtype / value range, this cell **measures** them from the cache and fails loudly if the
data is not raw pixel values. A sample strip is saved to `appearance_probe.png` -- glance at it to
confirm colours look natural (if faces look blue, the cache stored BGR: set the flag below and re-run).
'''))
    cells.append(code(r'''
APPEARANCE_IS_BGR = False   # flip to True only if appearance_probe.png shows swapped colours

probe_ids = [vid for vid, _ in train_samples[:12]]
with np.load(cache_index[probe_ids[0]]) as d:
    print("npz keys:", {k: (d[k].shape, str(d[k].dtype)) for k in d.files})

probe = []
for vid in probe_ids:
    with np.load(cache_index[vid]) as d:
        probe.append(np.asarray(d["appearance"]))
shapes = {p.shape for p in probe}
vmin = float(min(p.min() for p in probe)); vmax = float(max(p.max() for p in probe))
print(f"appearance shapes: {shapes} | dtype: {probe[0].dtype} | value range: [{vmin:.3f}, {vmax:.3f}]")

assert vmin >= -1e-3, ("appearance contains negative values -> it was stored already normalised with "
                       "unknown statistics. Stop and check the cache-building notebook before training.")
VALUE_SCALE = 255.0 if vmax <= 1.0 + 1e-3 else 1.0
print(f"-> pixel scale factor to uint8: {VALUE_SCALE}")


def appearance_to_thwc(a):
    """Cached appearance array -> uint8 (T, H, W, 3), temporal order preserved."""
    a = np.asarray(a)
    if a.ndim == 5:                       # (T, S, H, W, C)
        a = a[:, CONFIG.APPEARANCE_STREAM]
    if a.ndim != 4:
        raise ValueError(f"unexpected appearance ndim {a.ndim}: {a.shape}")
    if a.shape[-1] != 3 and a.shape[1] == 3:   # (T, C, H, W) -> (T, H, W, C)
        a = a.transpose(0, 2, 3, 1)
    if a.shape[-1] != 3:
        raise ValueError(f"cannot find RGB channel axis in {a.shape}")
    if a.shape[0] != CONFIG.FRAME_COUNT:  # uniform resample (only if cache differs from config)
        a = a[np.linspace(0, a.shape[0] - 1, CONFIG.FRAME_COUNT).round().astype(int)]
    if a.dtype != np.uint8:
        a = np.clip(a.astype(np.float32) * VALUE_SCALE, 0, 255).astype(np.uint8)
    if APPEARANCE_IS_BGR:
        a = a[..., ::-1]
    return np.ascontiguousarray(a)


ref = appearance_to_thwc(probe[0])
REF_HW = ref.shape[1:3]
print(f"normalised clip: {ref.shape} uint8 (T, H, W, C); native crop size {REF_HW} -> resized to "
      f"{CONFIG.IMG_SIZE}x{CONFIG.IMG_SIZE} on GPU")

fig, axes = plt.subplots(1, 8, figsize=(16, 2.4))
for ax, t in zip(axes, np.linspace(0, CONFIG.FRAME_COUNT - 1, 8).astype(int)):
    ax.imshow(ref[t]); ax.set_title(f"t={t}", fontsize=8); ax.axis("off")
plt.suptitle(f"appearance probe -- {probe_ids[0]} ({CLASS_NAMES[train_samples[0][1]]})")
plt.savefig(CONFIG.OUT_DIR / "appearance_probe.png", dpi=100, bbox_inches="tight"); plt.show()
del probe
'''))
    cells.append(md(r'''
## 4. Dataset & loaders
Workers only decode the `.npz` and return raw `uint8` clips; resizing, augmentation and
normalisation run batched **on the GPU** so the 4 Kaggle CPU cores don't bottleneck two T4s.
'''))
    cells.append(code(r'''
class AppearanceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vid, label = self.samples[i]
        with np.load(cache_index[vid]) as d:
            clip = appearance_to_thwc(d["appearance"])
        clip = torch.from_numpy(clip)
        if tuple(clip.shape[1:3]) != tuple(REF_HW):   # defensive: keep batch shapes equal
            clip = F.interpolate(clip.permute(0, 3, 1, 2).float(), size=REF_HW, mode="bilinear",
                                 align_corners=False).round().clamp(0, 255).byte().permute(0, 2, 3, 1)
        return clip, label


def make_loader(samples, train):
    ds = AppearanceDataset(samples)
    kw = dict(batch_size=CONFIG.BATCH_SIZE, num_workers=CONFIG.NUM_WORKERS,
              pin_memory=DEVICE == "cuda", persistent_workers=CONFIG.NUM_WORKERS > 0)
    if CONFIG.NUM_WORKERS > 0:
        kw["prefetch_factor"] = 4
    if train:
        counts = Counter(c for _, c in samples)
        weights = [1.0 / counts[c] for _, c in samples]   # same class balancing as v10
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, sampler=sampler, drop_last=True, **kw)
    return DataLoader(ds, shuffle=False, **kw)


train_loader = make_loader(train_samples, train=True)
val_loader = make_loader(val_samples, train=False)
test_loader = make_loader(test_samples, train=False)

_MEAN = torch.tensor(CONFIG.MEAN, device=DEVICE).view(1, 3, 1, 1, 1)
_STD = torch.tensor(CONFIG.STD, device=DEVICE).view(1, 3, 1, 1, 1)


def gpu_preprocess(x, train):
    """uint8 (B, T, H, W, 3) -> normalised float (B, 3, T, S, S).
    Eval / app inference: full-frame resize to S x S, no crop."""
    x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)          # (B, 3, T, H, W)
    B, _, T, H, W = x.shape
    S = CONFIG.IMG_SIZE
    if train:
        out = []
        for i in range(B):
            area = random.uniform(CONFIG.AUG_SCALE_MIN, 1.0)
            ch, cw = max(8, int(H * math.sqrt(area))), max(8, int(W * math.sqrt(area)))
            y0, x0 = random.randint(0, H - ch), random.randint(0, W - cw)
            c = x[i:i + 1, :, :, y0:y0 + ch, x0:x0 + cw]
            if CONFIG.AUG_HFLIP and random.random() < 0.5:
                c = c.flip(-1)
            out.append(F.interpolate(c, size=(T, S, S), mode="trilinear", align_corners=False))
        x = torch.cat(out)
    else:
        x = F.interpolate(x, size=(T, S, S), mode="trilinear", align_corners=False)
    return (x - _MEAN) / _STD


xb, yb = next(iter(val_loader))
print("raw batch:", tuple(xb.shape), xb.dtype, "| model input:", tuple(gpu_preprocess(xb.to(DEVICE), False).shape))
'''))
    cells.append(md(r'''
## 5. Model -- Swin3D-T with a new 1000-way head
`head` (768 -> 400 Kinetics classes) is replaced by `Dropout(0.5) -> Linear(768, 1000)`. Every
other weight starts from Kinetics-400 and is fine-tuned. Stochastic depth is raised from
torchvision's default 0.1 to 0.2 (scaled per block, keeping torchvision's linear schedule).
'''))
    cells.append(code(r'''
def build_model(num_classes, pretrained):
    weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
    m = swin3d_t(weights=weights)
    for mod in m.modules():                 # 0.1 -> STOCHASTIC_DEPTH, same per-block schedule
        if isinstance(mod, StochasticDepth):
            mod.p = mod.p * (CONFIG.STOCHASTIC_DEPTH / 0.1)
    m.head = nn.Sequential(nn.Dropout(CONFIG.HEAD_DROPOUT), nn.Linear(m.head.in_features, num_classes))
    return m
'''))
    cells.append(code(r'''
seed_everything(CONFIG.SEED)
base_model = build_model(NUM_CLASSES, CONFIG.PRETRAINED).to(DEVICE)
model = nn.DataParallel(base_model) if N_GPUS > 1 else base_model
n_params = sum(p.numel() for p in base_model.parameters())
print(f"Swin3D-T parameters: {n_params:,}  | pretrained={CONFIG.PRETRAINED} | DataParallel={N_GPUS > 1}")

head_params = list(base_model.head.parameters())
head_ids = {id(p) for p in head_params}
decay, no_decay = [], []
for n_, p in base_model.named_parameters():
    if id(p) in head_ids:
        continue
    (no_decay if p.ndim <= 1 or "relative_position_bias_table" in n_ else decay).append(p)
optimizer = torch.optim.AdamW([
    {"params": decay, "lr": CONFIG.LR_BACKBONE, "weight_decay": CONFIG.WEIGHT_DECAY},
    {"params": no_decay, "lr": CONFIG.LR_BACKBONE, "weight_decay": 0.0},
    {"params": head_params, "lr": CONFIG.LR_HEAD, "weight_decay": CONFIG.WEIGHT_DECAY},
])
steps_per_epoch = len(train_loader)
total_steps = max(1, CONFIG.EPOCHS * steps_per_epoch)
warmup_steps = min(CONFIG.WARMUP_EPOCHS * steps_per_epoch, total_steps // 2)


def lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG.LABEL_SMOOTHING)
amp_on = CONFIG.USE_AMP and DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
print(f"steps/epoch={steps_per_epoch} | total={total_steps} | warmup={warmup_steps} | AMP={amp_on}")
'''))
    cells.append(md(r'''
## 6. Training (resumable, NaN-guarded, time-boxed)
Same hygiene as v9.1/v10: per-epoch `last.pt`, `best_model.pt` selected on **val top-1**, early
stopping after `MIN_EPOCHS_BEFORE_STOPPING` with `EARLY_STOPPING_PATIENCE`, NaN rollback.
Added: a time guard that will not start an epoch it cannot finish (plus a reserve for the
test evaluation) inside `TIME_BUDGET_HOURS`.
'''))
    cells.append(code(EVAL_HELPERS + r'''

def model_has_nan(m):
    return any(not torch.isfinite(p).all() for p in m.parameters())


def topk_correct(logits, y, k):
    k = min(k, logits.shape[1])
    return logits.topk(k, dim=1).indices.eq(y.unsqueeze(1)).any(dim=1).sum().item()


def run_epoch(loader, train, epoch=None):
    model.train(train)
    tot_loss, c1, c5, n, skipped = 0.0, 0, 0, 0, 0
    t0 = time.time()
    with torch.set_grad_enabled(train):
        for step, (x, y) in enumerate(loader):
            x = gpu_preprocess(x.to(DEVICE, non_blocking=True), train)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on):
                logits = model(x)
            loss = criterion(logits.float(), y)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(base_model.parameters(), CONFIG.GRAD_CLIP_NORM)
                if torch.isfinite(gn):
                    scaler.step(optimizer)
                else:
                    skipped += 1
                scaler.update()
                scheduler.step()
            bs = y.size(0)
            tot_loss += loss.item() * bs; c1 += topk_correct(logits, y, 1); c5 += topk_correct(logits, y, 5); n += bs
            if train and step in (50, 200) or (train and step > 0 and step % 500 == 0):
                rate = n / (time.time() - t0)
                print(f"   epoch {epoch} step {step}/{len(loader)} | {rate:.1f} clips/s | "
                      f"epoch ETA {(len(loader.sampler) - n) / max(rate, 1e-6) / 60:.1f} min | "
                      f"loss {tot_loss / n:.3f}")
    if skipped:
        print(f"   [warning] {skipped} non-finite batches skipped")
    return {"loss": tot_loss / max(n, 1), "top1": c1 / max(n, 1), "top5": c5 / max(n, 1),
            "seconds": time.time() - t0}


LAST_PATH = CONFIG.OUT_DIR / "last.pt"
BEST_PATH = CONFIG.OUT_DIR / "best_model.pt"
history, best_val_top1, epochs_without_improvement, start_epoch = [], -1.0, 0, 1

resume_path = CONFIG.RESUME_FROM if CONFIG.RESUME_FROM else (LAST_PATH if LAST_PATH.exists() else None)
if resume_path is not None and Path(resume_path).exists():
    st = torch.load(resume_path, map_location=DEVICE, weights_only=False)
    base_model.load_state_dict(st["model_state_dict"])
    optimizer.load_state_dict(st["optimizer_state_dict"])
    scheduler.load_state_dict(st["scheduler_state_dict"])
    scaler.load_state_dict(st["scaler_state_dict"])
    history, best_val_top1 = st["history"], st["best_val_top1"]
    epochs_without_improvement, start_epoch = st["epochs_without_improvement"], st["epoch"] + 1
    prev_best = Path(resume_path).parent / "best_model.pt"
    if prev_best.exists() and prev_best.resolve() != BEST_PATH.resolve():
        BEST_PATH.write_bytes(prev_best.read_bytes())   # carry the previous version's best forward
    print(f"Resumed from {resume_path}: continuing at epoch {start_epoch}, best val_top1 {best_val_top1:.3f}")
else:
    print("Starting fresh.")

budget_s = CONFIG.TIME_BUDGET_HOURS * 3600
test_reserve_s = 15 * 60
best_epoch = max((h["epoch"] for h in history if h["val_top1"] >= best_val_top1), default=0)

for epoch in range(start_epoch, CONFIG.EPOCHS + 1):
    elapsed = time.time() - NOTEBOOK_START
    if history:
        per_epoch = max(h["epoch_seconds"] for h in history[-3:])
        if elapsed + per_epoch * 1.1 + test_reserve_s > budget_s:
            print(f"[time guard] {elapsed / 3600:.2f} h used; next epoch (~{per_epoch / 60:.0f} min) + test "
                  f"evaluation would exceed {CONFIG.TIME_BUDGET_HOURS} h -- stopping training here.")
            break

    tr = run_epoch(train_loader, True, epoch)
    va = run_epoch(val_loader, False)
    test_reserve_s = va["seconds"] * max(1.0, len(test_samples) / max(1, len(val_samples))) * 1.5 + 10 * 60

    if model_has_nan(base_model):
        print(f"[STOPPING] epoch {epoch} produced NaN weights; best_model.pt (val_top1={best_val_top1:.3f}) is intact.")
        break

    rec = {"epoch": epoch, "train_loss": tr["loss"], "train_top1": tr["top1"], "train_top5": tr["top5"],
           "val_loss": va["loss"], "val_top1": va["top1"], "val_top5": va["top5"],
           "epoch_seconds": tr["seconds"] + va["seconds"], "lr_head": optimizer.param_groups[2]["lr"]}
    history.append(rec)
    print(f"Epoch {epoch}/{CONFIG.EPOCHS} | train loss {tr['loss']:.4f} top1 {tr['top1']:.3f} | "
          f"val loss {va['loss']:.4f} top1 {va['top1']:.3f} top5 {va['top5']:.3f} | "
          f"{rec['epoch_seconds'] / 60:.1f} min | total {(time.time() - NOTEBOOK_START) / 3600:.2f} h")

    is_best = va["top1"] > best_val_top1
    if is_best:
        best_val_top1, best_epoch, epochs_without_improvement = va["top1"], epoch, 0
        torch.save({"epoch": epoch, "model_state_dict": base_model.state_dict(),
                    "val_top1": best_val_top1, "val_loss": va["loss"]}, BEST_PATH)
        print(f"   -> new best val_top1 {best_val_top1:.3f}, saved best_model.pt")
    else:
        epochs_without_improvement += 1

    torch.save({"epoch": epoch, "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(), "history": history, "best_val_top1": best_val_top1,
                "best_epoch": best_epoch, "epochs_without_improvement": epochs_without_improvement}, LAST_PATH)
    with open(CONFIG.OUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=1)

    if (not is_best and epoch >= CONFIG.MIN_EPOCHS_BEFORE_STOPPING
            and epochs_without_improvement >= CONFIG.EARLY_STOPPING_PATIENCE):
        print(f"Early stopping at epoch {epoch}: no val_top1 improvement for {CONFIG.EARLY_STOPPING_PATIENCE} epochs.")
        break

best = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
base_model.load_state_dict(best["model_state_dict"])
print(f"\nTraining done in {(time.time() - NOTEBOOK_START) / 3600:.2f} h. "
      f"Restored best epoch {best['epoch']} (val_top1 {best['val_top1']:.3f}).")
plot_history(history, "Swin3D-T (1000 cls)", CONFIG.OUT_DIR / "training_curves.png")
'''))
    cells.append(md("## 7. Test-set evaluation (touched once, best checkpoint, no TTA)"))
    cells.append(code(r'''
@torch.inference_mode()
def collect_predictions(loader):
    model.eval()
    yt, yp, conf, top5 = [], [], [], []
    for x, y in loader:
        x = gpu_preprocess(x.to(DEVICE, non_blocking=True), False)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on):
            logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)
        c, p = probs.max(dim=1)
        yt.append(y.numpy()); yp.append(p.cpu().numpy()); conf.append(c.cpu().numpy())
        top5.append(probs.topk(min(5, NUM_CLASSES), dim=1).indices.cpu().numpy())
    return {"y_true": np.concatenate(yt), "y_pred": np.concatenate(yp),
            "confidence": np.concatenate(conf), "top5": np.concatenate(top5)}


t0 = time.time()
test_preds = collect_predictions(test_loader)
test_seconds = time.time() - t0
test_metrics = summarize_predictions(test_preds["y_true"], test_preds["y_pred"], test_preds["top5"], NUM_CLASSES)
cm = save_evaluation(CONFIG.OUT_DIR, "swin3d_t", test_preds, [v for v, _ in test_samples], CLASS_NAMES)
print(f"\nSwin3D-T 1000-class TEST: top-1 {test_metrics['top1']:.3f} | top-5 {test_metrics['top5']:.3f} | "
      f"macro-F1 {test_metrics['macro_f1']:.3f} | {len(test_samples)} clips in {test_seconds / 60:.1f} min")
'''))
    cells.append(md(r'''
## 8. Export for the web app
`app_export/` contains everything the Streamlit app needs and nothing else:
`model_weights.pt` (plain fp32 `state_dict`), `class_list.json`, `model_config.json`
(exact preprocessing), `metrics.json`. CPU latency is measured here so the app's comparison
page shows a real number.
'''))
    cells.append(code(r'''
export = CONFIG.OUT_DIR / "app_export"
export.mkdir(exist_ok=True)
state = {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
         for k, v in base_model.state_dict().items()}
torch.save(state, export / "model_weights.pt")

cpu_model = build_model(NUM_CLASSES, pretrained=False)
cpu_model.load_state_dict(state, strict=True)
example = torch.randn(1, 3, CONFIG.FRAME_COUNT, CONFIG.IMG_SIZE, CONFIG.IMG_SIZE)
assert cpu_model.eval()(example).shape == (1, NUM_CLASSES)
latency = measure_cpu_latency(cpu_model, example)
print(f"CPU latency (2 threads, 1 clip): {latency['mean_s']:.2f} s")
del cpu_model

model_config = {
    "model_id": "swin3d_t_1000",
    "display_name": "Video Swin-T (RGB)",
    "architecture": "torchvision.models.video.swin3d_t",
    "head": "Sequential(Dropout(p), Linear(768, num_classes))  # state_dict keys head.1.*",
    "head_dropout": CONFIG.HEAD_DROPOUT,
    "init_weights": "Swin3D_T_Weights.KINETICS400_V1" if CONFIG.PRETRAINED else None,
    "num_classes": NUM_CLASSES,
    "class_fingerprint": CLASS_FINGERPRINT,
    "input_stream": "cached RGB appearance crop (appearance[:, 0]) -- same crop as the cache builder",
    "input_layout": "B,C,T,H,W",
    "frame_count": CONFIG.FRAME_COUNT,
    "native_crop_hw": list(map(int, REF_HW)),
    "img_size": CONFIG.IMG_SIZE,
    "resize": "full-frame trilinear resize to img_size x img_size, no crop (eval)",
    "pixel_scale": "uint8 / 255",
    "appearance_is_bgr_in_cache": APPEARANCE_IS_BGR,
    "mean": CONFIG.MEAN, "std": CONFIG.STD,
    "train_hparams": {k: getattr(CONFIG, k) for k in ["BATCH_SIZE", "EPOCHS", "LR_BACKBONE", "LR_HEAD",
                      "WEIGHT_DECAY", "LABEL_SMOOTHING", "STOCHASTIC_DEPTH", "AUG_SCALE_MIN", "AUG_HFLIP"]},
}
metrics = {
    "model_id": "swin3d_t_1000", "num_classes": NUM_CLASSES, "class_fingerprint": CLASS_FINGERPRINT,
    "best_epoch": int(best["epoch"]), "epochs_ran": len(history), "val_top1_best": float(best["val_top1"]),
    "test": test_metrics, "trainable_params": int(n_params),
    "train_hours": round(sum(h["epoch_seconds"] for h in history) / 3600, 2),
    "gpu": [torch.cuda.get_device_name(i) for i in range(N_GPUS)],
    "gpu_test_seconds_total": round(test_seconds, 1), "cpu_latency": latency,
    "smoke_test": CONFIG.SMOKE_TEST,
}
for name, obj in [("model_config.json", model_config), ("metrics.json", metrics)]:
    with open(export / name, "w") as f:
        json.dump(obj, f, indent=1)
for name in ["class_list.json", "per_class_metrics.csv", "confusion_matrix.npz", "training_history.json"]:
    src = CONFIG.OUT_DIR / name
    if src.exists():
        (export / name).write_bytes(src.read_bytes())

print("\nExported:", *sorted(f"{p.name}  ({p.stat().st_size / 1e6:.1f} MB)" for p in export.iterdir()), sep="\n  ")
print("\n" + json.dumps(metrics["test"], indent=1))
if CONFIG.SMOKE_TEST:
    print("\n*** SMOKE TEST finished OK -- set ASL_SMOKE_TEST/CONFIG.SMOKE_TEST False and Save & Run All. ***")
'''))
    return cells


# --------------------------------------------------------------------------------------
# Notebook B -- landmark models (v10 Geometry Transformer @1000 + ST-GCN)
# --------------------------------------------------------------------------------------

def build_landmark_notebook():
    cells = []
    cells.append(md(r'''
# ASL Citizen -- 1000 classes -- Landmark models: v10 Geometry Transformer + ST-GCN
### Capstone C -- model comparison track (new architecture #2 of 2, plus the 1000-class baseline)

This notebook trains **two** models on the cached MediaPipe landmarks (16 frames x 116 features),
one after the other:

| # | Model | Why |
|---|---|---|
| 1 | **v10 Geometry-Only Transformer** -- the team's model, code + hyperparameters copied unchanged | The 1000-class point of the v10 series, so every architecture in the comparison has a 1000-class number on the identical split |
| 2 | **ST-GCN** (spatial-temporal graph convolutional network) | The standard skeleton-based action/sign-recognition architecture, and the v10 notebook's own suggested future work ("a small temporal-conv or ST-GCN-style module"). Treats the 58 landmarks as a graph (hand bones, arm chain, eyebrows) + a learned adjacency, instead of flattening them into one vector |

Both models read only landmarks, so all landmarks are **pre-loaded into RAM once** (~0.2 GB) --
same numbers, just no repeated disk I/O. Each model trains in well under an hour on one GPU.

---
### How to run on Kaggle
1. *File -> Import Notebook* -> upload this `.ipynb`.
2. **Add Input**: the 3 `asl-shared-cache-building-id0x` notebook outputs + the `asl-citizen` dataset
   (paths auto-discovered).
3. Accelerator = **GPU T4 x1 or P100** (one GPU is plenty). Internet not required.
4. *(Optional)* `SMOKE_TEST = True` -> Run All (~3-5 min) -> back to `False`.
5. **Save Version -> Save & Run All (Commit)**. Estimated **1-2 h total** (dominated by
   preloading ~31k `.npz` files and the batch-size-8 v10 schedule).
6. Download `landmark_models_1000cls/` from the Output tab.

Run this **at the same time** as `train_1000_swin3d_t.ipynb` -- they don't depend on each other.
'''))
    cells.append(md("## 1. Setup"))
    cells.append(code(SETUP))
    cells.append(md("## 2. Configuration"))
    cells.append(code(r'''
class CONFIG:
    WORK_DIR = Path(_env("ASL_WORK_DIR", "/kaggle/working"))

    CACHE_DIRS = [
        Path("/kaggle/input/notebooks/aniruddha2022/asl-shared-cache-building-id00/feature_cache_shard_0"),
        Path("/kaggle/input/notebooks/redoanmozumder/asl-shared-cache-building-id02/feature_cache_shard_2"),
        Path("/kaggle/input/notebooks/pranaysahaps/asl-shared-cache-building-id01/feature_cache_shard_1"),
    ]
    SPLITS_DIR = Path("/kaggle/input/datasets/pranaysahaps/asl-citizen/splits")
    OUT_DIR = WORK_DIR / "landmark_models_1000cls"

    TOP_K_CLASSES = int(_env("ASL_TOP_K", 1000))

    # ---- cache layout (v10 CONFIG, unchanged) ----
    FRAME_COUNT = 16
    N_HAND_LANDMARKS = 21
    POSE_SUBSET_IDX = [11, 12, 13, 14, 15, 16]
    FACE_SUBSET_IDX = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]

    # ---- which models to train (in order) ----
    MODELS_TO_RUN = ["geometry_transformer_v10", "stgcn"]

    SMOKE_TEST = _env("ASL_SMOKE_TEST", "0") == "1"
    SMOKE_SAMPLES = {"train": 512, "val": 256, "test": 256}
    PRELOAD_THREADS = 16
    SEED = 42


N_POINTS = CONFIG.N_HAND_LANDMARKS * 2 + len(CONFIG.POSE_SUBSET_IDX) + len(CONFIG.FACE_SUBSET_IDX)  # 58
LANDMARK_VECTOR_DIM = N_POINTS * 2                                                                 # 116
print(f"Landmark points per frame: {N_POINTS} -> vector dim {LANDMARK_VECTOR_DIM}")
''' + "\n" + RESOLVE_INPUTS))
    cells.append(md(MANIFEST_MD))
    cells.append(code(MANIFEST))
    cells.append(md(r'''
## 3. Pre-load landmarks into memory
Exactly the tensor v10's `WLASLCachedDataset` returned: `data["landmarks"]` as float32 with
`nan_to_num(nan=0, posinf=0, neginf=0)`. Only the `landmarks` member of each `.npz` is read.
'''))
    cells.append(code(r'''
def load_landmarks(vid):
    with np.load(cache_index[vid]) as d:
        x = np.asarray(d["landmarks"], dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def preload(samples, name):
    t0 = time.time()
    with ThreadPoolExecutor(CONFIG.PRELOAD_THREADS) as ex:
        arr = list(ex.map(load_landmarks, [v for v, _ in samples]))
    X = torch.from_numpy(np.stack(arr))
    y = torch.tensor([c for _, c in samples], dtype=torch.long)
    print(f"{name}: {tuple(X.shape)} loaded in {time.time() - t0:.0f}s")
    return X, y


X_train, y_train = preload(train_samples, "train")
X_val, y_val = preload(val_samples, "val")
X_test, y_test = preload(test_samples, "test")
assert X_train.shape[1:] == (CONFIG.FRAME_COUNT, LANDMARK_VECTOR_DIM), X_train.shape
print(f"value range: [{X_train.min():.3f}, {X_train.max():.3f}] | "
      f"all-zero frames: {(X_train.abs().sum(-1) == 0).float().mean():.1%}")
'''))
    cells.append(md(r'''
## 4. Verify the landmark layout (needed only for the ST-GCN graph)
The v10 Transformer consumes the flat 116-vector, so it doesn't care how points are packed. ST-GCN
needs to know which numbers form each `(x, y)` point. Rather than assume, this cell **tests both
plausible packings** against hand anatomy: in the correct one, MediaPipe's 21 hand-bone pairs are much
shorter than the average distance between arbitrary hand points.
'''))
    cells.append(code(r'''
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def to_points(X, packing):
    # X: (..., 116) -> (..., 58, 2)
    if packing == "interleaved":            # [x0, y0, x1, y1, ...]
        return X.reshape(*X.shape[:-1], N_POINTS, 2)
    if packing == "planar":                 # [x0..x57, y0..y57]
        return X.reshape(*X.shape[:-1], 2, N_POINTS).transpose(-1, -2)
    raise ValueError(packing)


def hand_bone_ratio(X, packing, hand_offset):
    P = to_points(X.reshape(-1, LANDMARK_VECTOR_DIM), packing)[:, hand_offset:hand_offset + 21]
    P = P[P.abs().sum(dim=(1, 2)) > 0]                       # frames where this hand was detected
    if len(P) < 50:
        return float("nan")
    D = torch.cdist(P, P)                                    # (M, 21, 21)
    bone = torch.stack([D[:, i, j] for i, j in HAND_EDGES], 1).mean()
    iu = torch.triu_indices(21, 21, 1)
    return float(bone / D[:, iu[0], iu[1]].mean())


sample_X = X_train[:min(len(X_train), 4000)]
ratios = {pk: [hand_bone_ratio(sample_X, pk, off) for off in (0, 21)] for pk in ("interleaved", "planar")}
for pk, r in ratios.items():
    print(f"{pk:12s} bone/avg-distance ratio -- hand block 0: {r[0]:.3f} | hand block 1: {r[1]:.3f}")
score = {pk: np.nanmean(r) for pk, r in ratios.items()}
LANDMARK_PACKING = min(score, key=score.get)
other = max(score, key=score.get)
LAYOUT_CONFIDENT = bool(score[LANDMARK_PACKING] < 0.7 * score[other])
print(f"\n-> using packing '{LANDMARK_PACKING}' "
      f"({'clear anatomical match' if LAYOUT_CONFIDENT else 'NOT conclusive -- ST-GCN will lean on its learned adjacency'})")
'''))
    cells.append(md(r'''
## 5. Models

**Model 1 -- v10 Geometry-Only Transformer:** `GeometryStream`, `PositionalEncoding`,
`GeometryOnlySLRModel` copied unchanged from the v10 notebook (D_MODEL 256, 4 heads, 3 layers,
FFN 512, dropout 0.5; input = coords + frame-to-frame velocity).

**Model 2 -- ST-GCN (adaptive graph):** input per frame = 58 nodes x (x, y, vx, vy). Graph =
self-loops + MediaPipe hand bones (both hands) + shoulder-elbow-wrist chains + eyebrow chains,
symmetrically normalised, **plus a learnable adjacency** per layer (2s-AGCN style) so the network can
learn hand-to-wrist, hand-to-face and two-hand relations the fixed skeleton does not encode.
7 blocks (64-64-64-128-128-256-256 channels, temporal kernel 5, temporal stride 2 at 128 and 256),
global average pooling, dropout 0.5, linear head.
'''))
    cells.append(code(r'''
# ---------------- Model 1: v10, unchanged ----------------
class V10:
    LANDMARK_EMBED_DIM = 128
    D_MODEL = 256
    N_HEADS = 4
    N_ENCODER_LAYERS = 3
    FFN_DIM = 512
    DROPOUT = 0.5


class GeometryStream(nn.Module):
    def __init__(self, input_dim=LANDMARK_VECTOR_DIM, embed_dim=V10.LANDMARK_EMBED_DIM):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim * 2, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))

    def forward(self, x):
        velocity = torch.zeros_like(x)
        velocity[:, 1:] = x[:, 1:] - x[:, :-1]
        return self.project(torch.cat([x, velocity], dim=-1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=CONFIG.FRAME_COUNT):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GeometryOnlySLRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.geometry_stream = GeometryStream()
        self.to_d_model = nn.Linear(V10.LANDMARK_EMBED_DIM, V10.D_MODEL)
        self.pos_encoding = PositionalEncoding(V10.D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(d_model=V10.D_MODEL, nhead=V10.N_HEADS,
                                                   dim_feedforward=V10.FFN_DIM, dropout=V10.DROPOUT,
                                                   batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=V10.N_ENCODER_LAYERS)
        self.word_head = nn.Linear(V10.D_MODEL, num_classes)

    def forward(self, landmarks):
        fused = self.pos_encoding(self.to_d_model(self.geometry_stream(landmarks)))
        return self.word_head(self.temporal_encoder(fused).mean(dim=1))


# ---------------- Model 2: ST-GCN ----------------
def skeleton_edges():
    e = []
    for off in (0, 21):                                         # two hands
        e += [(i + off, j + off) for i, j in HAND_EDGES]
    p = {idx: 42 + k for k, idx in enumerate(CONFIG.POSE_SUBSET_IDX)}   # pose 11..16 -> nodes 42..47
    e += [(p[11], p[12]), (p[11], p[13]), (p[13], p[15]), (p[12], p[14]), (p[14], p[16])]
    f0 = 42 + len(CONFIG.POSE_SUBSET_IDX)                        # eyebrows -> nodes 48..57
    e += [(f0 + i, f0 + i + 1) for i in range(4)] + [(f0 + 5 + i, f0 + 6 + i) for i in range(4)]
    return e


def normalized_adjacency(V, edges):
    A = torch.eye(V)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]


class AdaptiveGraphConv(nn.Module):
    def __init__(self, cin, cout, A):
        super().__init__()
        self.register_buffer("A", A)
        self.B = nn.Parameter(torch.zeros_like(A))              # learned adjacency, starts at 0
        self.conv = nn.Conv2d(cin, cout, 1)

    def forward(self, x):                                       # x: (N, C, T, V)
        return torch.einsum("nctv,vw->nctw", self.conv(x), self.A + self.B)


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, dropout=0.1, kt=5):
        super().__init__()
        self.gcn = AdaptiveGraphConv(cin, cout, A)
        self.bn1 = nn.BatchNorm2d(cout)
        self.tcn = nn.Sequential(nn.Conv2d(cout, cout, (kt, 1), (stride, 1), ((kt - 1) // 2, 0)),
                                 nn.BatchNorm2d(cout), nn.Dropout(dropout))
        if cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.tcn(F.relu(self.bn1(self.gcn(x)))) + self.res(x))


class STGCN(nn.Module):
    def __init__(self, num_classes, packing, in_ch=4, dropout=0.5):
        super().__init__()
        self.packing = packing
        A = normalized_adjacency(N_POINTS, skeleton_edges())
        self.data_bn = nn.BatchNorm1d(in_ch * N_POINTS)
        cfg = [(in_ch, 64, 1), (64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1)]
        self.blocks = nn.ModuleList(STGCNBlock(i, o, A, s) for i, o, s in cfg)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, landmarks):                               # (N, T, 116) -- same input as v10
        N, T, _ = landmarks.shape
        pts = to_points(landmarks, self.packing)                # (N, T, V, 2)
        vel = torch.zeros_like(pts)
        vel[:, 1:] = pts[:, 1:] - pts[:, :-1]
        x = torch.cat([pts, vel], -1)                           # (N, T, V, 4)
        x = x.permute(0, 2, 3, 1).reshape(N, N_POINTS * 4, T)   # (N, V*C, T)
        x = self.data_bn(x).reshape(N, N_POINTS, 4, T).permute(0, 2, 3, 1)   # (N, C, T, V)
        for blk in self.blocks:
            x = blk(x)
        return self.fc(self.drop(x.mean(dim=(2, 3))))


def augment_landmarks(x):
    """ST-GCN only: per-clip random scale/rotate/shift of *detected* points (zeros = missing stay zero)."""
    N = x.shape[0]
    pts = to_points(x, LANDMARK_PACKING).clone()                # (N, T, V, 2)
    mask = (pts.abs().sum(-1, keepdim=True) > 0).float()
    centre = 0.5
    s = torch.empty(N, 1, 1, 1, device=x.device).uniform_(0.9, 1.1)
    th = torch.empty(N, device=x.device).uniform_(-math.radians(10), math.radians(10))
    R = torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], -2)
    shift = torch.empty(N, 1, 1, 2, device=x.device).uniform_(-0.05, 0.05)
    p = (pts - centre) * s
    p = torch.einsum("ntvc,ndc->ntvd", p, R) + centre + shift
    pts = p * mask
    if LANDMARK_PACKING == "interleaved":
        return pts.reshape(N, x.shape[1], -1)
    return pts.transpose(-1, -2).reshape(N, x.shape[1], -1)


# ---------------- per-model training recipes ----------------
MODEL_SPECS = {
    "geometry_transformer_v10": dict(   # v10 CONFIG values, unchanged
        display_name="Geometry Transformer (v10, landmarks)",
        build=lambda: GeometryOnlySLRModel(NUM_CLASSES),
        batch_size=8, lr=1e-4, weight_decay=3e-4, grad_clip=0.5, label_smoothing=0.1,
        epochs=50, warmup_epochs=0, scheduler="cosine_epoch",
        min_epochs=30, patience=8, augment=False),
    "stgcn": dict(
        display_name="ST-GCN (landmarks)",
        build=lambda: STGCN(NUM_CLASSES, LANDMARK_PACKING),
        batch_size=32, lr=1e-3, weight_decay=5e-4, grad_clip=1.0, label_smoothing=0.1,
        epochs=80, warmup_epochs=3, scheduler="warmup_cosine_step",
        min_epochs=30, patience=12, augment=True),
}
for name in CONFIG.MODELS_TO_RUN:
    m = MODEL_SPECS[name]["build"]()
    out = m(X_train[:2])
    print(f"{name:26s} params={sum(p.numel() for p in m.parameters()):>10,}  output={tuple(out.shape)}")
'''))
    cells.append(md(r'''
## 6. Train + evaluate each model
Same protocol as v10 for both: `WeightedRandomSampler`, label smoothing, gradient clipping,
NaN guard, best checkpoint on **val top-1**, early stopping with a minimum-epoch floor, then a single
pass over the held-out test split with the best checkpoint. fp32 throughout (as in v10).
'''))
    cells.append(code(EVAL_HELPERS + r'''

def batches(X, y, bs, sampler=None):
    idx = list(sampler) if sampler is not None else list(range(len(X)))
    for i in range(0, len(idx) - (bs - 1 if sampler is not None else 0), bs):
        j = torch.tensor(idx[i:i + bs])
        if len(j) == 0:
            break
        yield X[j], y[j]


def topk_correct(logits, y, k):
    k = min(k, logits.shape[1])
    return logits.topk(k, dim=1).indices.eq(y.unsqueeze(1)).any(dim=1).sum().item()


@torch.inference_mode()
def predict(model, X, bs=256):
    model.eval()
    probs = []
    for i in range(0, len(X), bs):
        probs.append(torch.softmax(model(X[i:i + bs].to(DEVICE)), 1).cpu())
    return torch.cat(probs)


def train_and_evaluate(name):
    spec = MODEL_SPECS[name]
    out_dir = CONFIG.OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(CONFIG.SEED)
    model = spec["build"]().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    epochs = 1 if CONFIG.SMOKE_TEST else spec["epochs"]
    min_epochs = 1 if CONFIG.SMOKE_TEST else spec["min_epochs"]

    criterion = nn.CrossEntropyLoss(label_smoothing=spec["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    counts = Counter(y_train.tolist())
    weights = [1.0 / counts[c] for c in y_train.tolist()]
    steps_per_epoch = len(X_train) // spec["batch_size"]
    if spec["scheduler"] == "cosine_epoch":                 # v10: CosineAnnealingLR(T_max=EPOCHS), per epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        step_sched = False
    else:
        total, warm = epochs * steps_per_epoch, spec["warmup_epochs"] * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / max(1, warm) if s < warm
                                                      else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
        step_sched = True

    history, best_val, best_epoch, ewi = [], -1.0, 0, 0
    best_path = out_dir / "best_model.pt"
    t_start = time.time()
    print(f"\n========== {name}: {n_params:,} params, up to {epochs} epochs ==========")
    for epoch in range(1, epochs + 1):
        model.train()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        tl, c1, c5, n = 0.0, 0, 0, 0
        for xb, yb in batches(X_train, y_train, spec["batch_size"], sampler):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            if spec["augment"]:
                xb = augment_landmarks(xb)
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["grad_clip"])
            if not torch.isfinite(gn):
                continue
            optimizer.step()
            if step_sched:
                scheduler.step()
            tl += loss.item() * len(yb); c1 += topk_correct(logits, yb, 1); c5 += topk_correct(logits, yb, 5); n += len(yb)
        if not step_sched:
            scheduler.step()

        model.eval()
        vl, v1, v5, vn = 0.0, 0, 0, 0
        with torch.no_grad():
            for xb, yb in batches(X_val, y_val, 256):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                vl += criterion(logits, yb).item() * len(yb); v1 += topk_correct(logits, yb, 1)
                v5 += topk_correct(logits, yb, 5); vn += len(yb)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            print(f"[STOPPING] NaN weights at epoch {epoch}; best checkpoint intact."); break

        rec = {"epoch": epoch, "train_loss": tl / max(n, 1), "train_top1": c1 / max(n, 1), "train_top5": c5 / max(n, 1),
               "val_loss": vl / vn, "val_top1": v1 / vn, "val_top5": v5 / vn}
        history.append(rec)
        is_best = rec["val_top1"] > best_val
        if is_best:
            best_val, best_epoch, ewi = rec["val_top1"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_top1": best_val,
                        "val_loss": rec["val_loss"]}, best_path)
        else:
            ewi += 1
        if epoch == 1 or epoch % 5 == 0 or is_best:
            print(f"epoch {epoch:3d} | train {rec['train_loss']:.3f}/{rec['train_top1']:.3f} | "
                  f"val {rec['val_loss']:.3f}/{rec['val_top1']:.3f}/{rec['val_top5']:.3f}"
                  f"{'  *best*' if is_best else ''} | {(time.time() - t_start) / 60:.1f} min")
        if not is_best and epoch >= min_epochs and ewi >= spec["patience"]:
            print(f"early stopping at epoch {epoch} (no val_top1 gain for {spec['patience']} epochs)"); break

    train_minutes = (time.time() - t_start) / 60
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=1)
    best = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    plot_history(history, spec["display_name"], out_dir / "training_curves.png")

    # ---- test: touched once ----
    t0 = time.time()
    probs = predict(model, X_test)
    gpu_test_s = time.time() - t0
    preds = {"y_true": y_test.numpy(), "y_pred": probs.argmax(1).numpy(), "confidence": probs.max(1).values.numpy(),
             "top5": probs.topk(min(5, NUM_CLASSES), 1).indices.numpy()}
    test_metrics = summarize_predictions(preds["y_true"], preds["y_pred"], preds["top5"], NUM_CLASSES)
    save_evaluation(out_dir, name, preds, [v for v, _ in test_samples], CLASS_NAMES)

    # ---- export for the app ----
    export = out_dir / "app_export"
    export.mkdir(exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, export / "model_weights.pt")
    cpu_model = spec["build"]()
    cpu_model.load_state_dict(state, strict=True)
    latency = measure_cpu_latency(cpu_model, X_test[:1].clone(), n_runs=10)
    model_config = {
        "model_id": f"{name}_1000", "display_name": spec["display_name"],
        "architecture": {"geometry_transformer_v10": "GeometryOnlySLRModel (v10)", "stgcn": "STGCN (adaptive graph)"}[name],
        "num_classes": NUM_CLASSES, "class_fingerprint": CLASS_FINGERPRINT,
        "input_stream": "MediaPipe landmarks from the cache builder: 16 frames x 116 = 58 points x (x, y); NaN -> 0",
        "input_layout": "B,T,116", "frame_count": CONFIG.FRAME_COUNT,
        "landmark_points": {"hands": "2 x 21 MediaPipe hand landmarks", "pose_subset_idx": CONFIG.POSE_SUBSET_IDX,
                            "face_subset_idx": CONFIG.FACE_SUBSET_IDX},
        "landmark_packing_detected": LANDMARK_PACKING, "landmark_packing_confident": LAYOUT_CONFIDENT,
        "hparams": {k: v for k, v in spec.items() if k not in ("build", "display_name")},
        "v10_dims": {k: getattr(V10, k) for k in dir(V10) if k.isupper()} if name == "geometry_transformer_v10" else None,
    }
    metrics = {"model_id": f"{name}_1000", "num_classes": NUM_CLASSES, "class_fingerprint": CLASS_FINGERPRINT,
               "best_epoch": int(best["epoch"]), "epochs_ran": len(history), "val_top1_best": float(best["val_top1"]),
               "test": test_metrics, "trainable_params": int(n_params), "train_minutes": round(train_minutes, 1),
               "gpu": torch.cuda.get_device_name(0) if N_GPUS else "cpu",
               "gpu_test_seconds_total": round(gpu_test_s, 2), "cpu_latency": latency, "smoke_test": CONFIG.SMOKE_TEST}
    for fname, obj in [("model_config.json", model_config), ("metrics.json", metrics)]:
        with open(export / fname, "w") as f:
            json.dump(obj, f, indent=1)
    for fname in ["per_class_metrics.csv", "confusion_matrix.npz", "training_history.json"]:
        (export / fname).write_bytes((out_dir / fname).read_bytes())
    (export / "class_list.json").write_bytes((CONFIG.OUT_DIR / "class_list.json").read_bytes())

    print(f"\n{name} TEST: top-1 {test_metrics['top1']:.3f} | top-5 {test_metrics['top5']:.3f} | "
          f"macro-F1 {test_metrics['macro_f1']:.3f} | best epoch {best['epoch']} | {train_minutes:.1f} min | "
          f"CPU latency {latency['mean_s'] * 1000:.1f} ms")
    del model, cpu_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return metrics


all_metrics = {name: train_and_evaluate(name) for name in CONFIG.MODELS_TO_RUN}
'''))
    cells.append(md(r'''
## 7. 1000-class comparison summary
The dual-stream (`r3d_18` + geometry) row is copied from the team's own record
(`BASELINE_BY_CLASS_COUNT[1000]` in the v10 notebooks, from `with_1000_class.ipynb`'s executed run):
it never had precision/recall/F1 computed, so those cells stay empty. Swin3D-T comes from the
other notebook's `metrics.json`.
'''))
    cells.append(code(r'''
rows = [{"Model": "v9 Dual-Stream (r3d_18 + geometry) [recorded]", "Val Top-1": 0.362, "Test Top-1": 0.282,
         "Test Top-5": 0.551, "Macro P": None, "Macro R": None, "Macro F1": None, "Params": None}]
for name, m in all_metrics.items():
    t = m["test"]
    rows.append({"Model": MODEL_SPECS[name]["display_name"], "Val Top-1": m["val_top1_best"], "Test Top-1": t["top1"],
                 "Test Top-5": t["top5"], "Macro P": t["macro_precision"], "Macro R": t["macro_recall"],
                 "Macro F1": t["macro_f1"], "Params": m["trainable_params"]})
summary = pd.DataFrame(rows).set_index("Model")
display(summary.round(3))
summary.to_csv(CONFIG.OUT_DIR / "comparison_1000cls.csv")
print(f"\nClass-list fingerprint: {CLASS_FINGERPRINT}  -- compare with the Swin notebook's printout.")
if CONFIG.SMOKE_TEST:
    print("\n*** SMOKE TEST finished OK -- set SMOKE_TEST False and Save & Run All. ***")
'''))
    return cells


def write(cells, name):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    path = HERE / name
    nbf.write(nb, path)
    print("wrote", path)


SWIN_EVAL_HEADER = r'''
# ASL Citizen -- 1000 classes -- Swin3D-T: **evaluation + export only** (no training)

Use this when `train_1000_swin3d_t.ipynb` finished training (it saved `best_model.pt`) but stopped
before the test evaluation / `app_export/`. It rebuilds the identical manifest, class mapping and
preprocessing, loads `best_model.pt`, verifies the class-list fingerprint against the training run,
evaluates the held-out test split once, and writes `app_export/`.

### How to run on Kaggle (~15-25 min)
1. *File -> Import Notebook* -> upload this `.ipynb`.
2. **Add Input**: the 3 `asl-shared-cache-building-id0x` outputs + `asl-citizen` dataset (as before)
   **plus the output of your finished Swin training notebook** (Add Input -> Your Work -> the Swin
   training notebook). Its `swin3d_t_1000cls/best_model.pt` is found automatically.
3. Accelerator = **GPU T4 x2**. Internet not required.
4. **Save Version -> Save & Run All (Commit)**. Download `swin3d_t_1000cls/` from the Output tab.
'''

SWIN_EVAL_LOAD_MD = r'''
## Load the trained run
Finds `best_model.pt`, `training_history.json` and `class_list.json` in the attached training output,
checks that the class fingerprint matches the one rebuilt above (same 1000 classes, same index order),
and loads the weights with `strict=True`.
'''

SWIN_EVAL_LOAD = r'''
TRAINED_RUN_DIR = None   # set explicitly if auto-discovery picks the wrong folder
if TRAINED_RUN_DIR is None:
    env_dir = os.environ.get("ASL_TRAINED_RUN_DIR")
    hits = [Path(env_dir) / "best_model.pt"] if env_dir else sorted(
        Path(p) for p in glob.glob("/kaggle/input/**/swin3d_t_1000cls/best_model.pt", recursive=True))
    assert hits and hits[0].exists(), ("best_model.pt from the Swin training run not found -- attach that "
                                       "notebook's output as an input (see top of notebook).")
    TRAINED_RUN_DIR = hits[0].parent
print("Trained run:", TRAINED_RUN_DIR)

with open(TRAINED_RUN_DIR / "class_list.json", encoding="utf-8") as f:
    trained_classes = json.load(f)
assert trained_classes["fingerprint"] == CLASS_FINGERPRINT and trained_classes["classes"] == CLASS_NAMES, (
    f"Class list mismatch: training run {trained_classes['fingerprint']} vs rebuilt {CLASS_FINGERPRINT}. "
    "The inputs differ from the training run -- do not evaluate with this mapping.")
print(f"Class-list fingerprint matches the training run: {CLASS_FINGERPRINT}")

with open(TRAINED_RUN_DIR / "training_history.json") as f:
    history = json.load(f)
best = torch.load(TRAINED_RUN_DIR / "best_model.pt", map_location="cpu", weights_only=False)
base_model = build_model(NUM_CLASSES, pretrained=False)
base_model.load_state_dict(best["model_state_dict"], strict=True)
base_model = base_model.to(DEVICE).eval()
model = nn.DataParallel(base_model) if N_GPUS > 1 else base_model
n_params = sum(p.numel() for p in base_model.parameters())
amp_on = CONFIG.USE_AMP and DEVICE == "cuda"
with open(CONFIG.OUT_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=1)
print(f"Loaded best epoch {best['epoch']} (val_top1 {best['val_top1']:.3f}); {len(history)} epochs in history; "
      f"{n_params:,} params")
plot_history(history, "Swin3D-T (1000 cls)", CONFIG.OUT_DIR / "training_curves.png")
'''


def build_swin_eval_notebook():
    """Eval/export only: reuses the training notebook's cells (identical preprocessing) but loads
    best_model.pt from a finished training run instead of training."""
    t = build_swin_notebook()
    # 0 hdr | 1-2 setup | 3-4 config | 5-6 manifest | 7-8 probe | 9-10 data | 11 model md
    # 12 build_model | 13 optimizer | 14-15 training | 16-17 test eval | 18-19 export
    assert "def build_model" in t[12].source and "def collect_predictions" in t[17].source
    assert "app_export" in t[19].source
    return ([md(SWIN_EVAL_HEADER)] + t[1:11] + [t[11], t[12], md(SWIN_EVAL_LOAD_MD),
            code(EVAL_HELPERS + "\n\n" + SWIN_EVAL_LOAD)] + t[16:20])


ARCH_SOURCE = (HERE.parent / "inference" / "architectures.py").read_text(encoding="utf-8")


def build_extras_notebook():
    cells = [md(r'''
# Extras for the web app: (A) full R3D-18 dual-stream evaluation, (B) parity sample pack
### Capstone C -- no training, ~20-30 min on one GPU

**(A)** The Capstone B dual-stream model (`r3d_18` + geometry Transformer, 1000 classes) only has
recorded top-1/top-5. This re-evaluates its saved `best_model.pt` on the same 12,560-clip test split
to get **precision / recall / F1, confusion matrix, per-class metrics and CPU latency** -- the same
artifacts the other three 1000-class models already have -- and exports `app_export/`.
Preprocessing is copied from that notebook (`/255`, Kinetics mean/std, no TTA). The reproduced
top-1 is checked against the recorded 0.282.

**(B)** Copies a small set of **raw test-split videos together with their cached `.npz`** so the web
app's video -> landmarks / signer-crop pipeline can be verified offline against exactly what the
models were trained on. (These files stay private to the team; they are not published in the app.)

### How to run on Kaggle
1. *File -> Import Notebook* -> upload this `.ipynb`.
2. **Add Input**: the 3 `asl-shared-cache-building-id0x` outputs, the `asl-citizen` dataset
   (needs `splits/` **and** `videos/`), and **the output of the previous r3d_18 notebook**
   (the one with `checkpoints/pilot_1000cls/best_model.pt`). Paths are auto-discovered.
3. Accelerator = **GPU T4 x1** (or P100). Internet not required.
4. **Save Version -> Save & Run All**, then download `extras/` from the Output tab.
'''), md("## 1. Setup"), code(SETUP), md("## 2. Configuration")]
    cells.append(code(r'''
class CONFIG:
    WORK_DIR = Path(_env("ASL_WORK_DIR", "/kaggle/working"))
    CACHE_DIRS = [
        Path("/kaggle/input/notebooks/aniruddha2022/asl-shared-cache-building-id00/feature_cache_shard_0"),
        Path("/kaggle/input/notebooks/redoanmozumder/asl-shared-cache-building-id02/feature_cache_shard_2"),
        Path("/kaggle/input/notebooks/pranaysahaps/asl-shared-cache-building-id01/feature_cache_shard_1"),
    ]
    SPLITS_DIR = Path("/kaggle/input/datasets/pranaysahaps/asl-citizen/splits")
    OUT_DIR = WORK_DIR / "extras"
    TOP_K_CLASSES = int(_env("ASL_TOP_K", 1000))
    EXPECTED_FINGERPRINT = _env("ASL_EXPECTED_FP", "72d3ef78291c")   # the 1000-class list all models share
    R3D_CHECKPOINT = None            # auto-discovered: /kaggle/input/**/pilot_1000cls/best_model.pt
    RECORDED_TEST_TOP1 = 0.282       # from the r3d_18 notebook's own test cell (no TTA)
    N_SAMPLE_VIDEOS = int(_env("ASL_N_SAMPLES", 30))
    BATCH_SIZE = 32
    NUM_WORKERS = int(_env("ASL_NUM_WORKERS", 4))
    SMOKE_TEST = _env("ASL_SMOKE_TEST", "0") == "1"
    SMOKE_SAMPLES = {"train": 16, "val": 16, "test": 64}
    SEED = 42
''' + "\n" + RESOLVE_INPUTS + r'''

VIDEOS_DIR = CONFIG.SPLITS_DIR.parent / "videos"
if os.environ.get("ASL_VIDEOS_DIR"):
    VIDEOS_DIR = Path(os.environ["ASL_VIDEOS_DIR"])
if not VIDEOS_DIR.exists():
    hits = sorted(glob.glob("/kaggle/input/**/asl-citizen/videos", recursive=True))
    VIDEOS_DIR = Path(hits[0]) if hits else VIDEOS_DIR
print("Videos dir:  ", VIDEOS_DIR, "(found)" if VIDEOS_DIR.exists() else "(NOT FOUND -- part B will be skipped)")
'''))
    cells += [md(MANIFEST_MD), code(MANIFEST + r'''

assert CLASS_FINGERPRINT == CONFIG.EXPECTED_FINGERPRINT, (
    f"class list {CLASS_FINGERPRINT} != expected {CONFIG.EXPECTED_FINGERPRINT}: inputs differ from the other runs")
print("Class list matches the other 1000-class models.")
''')]
    cells += [md("## 3. Model definitions (identical to the web app's `inference/architectures.py`)"),
              code(ARCH_SOURCE)]
    cells.append(md("## 4. (A) R3D-18 dual-stream -- test evaluation + export"))
    cells.append(code(EVAL_HELPERS + r'''

KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 1, 3, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 1, 3, 1, 1)


class DualStreamEvalDataset(Dataset):
    # Exact eval-time __getitem__ of the r3d_18 notebook (augment=False).
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        vid, label = self.samples[i]
        with np.load(cache_index[vid]) as d:
            appearance = torch.from_numpy(d["appearance"]).float() / 255.0
            landmarks = torch.from_numpy(d["landmarks"]).float()
        appearance = appearance.permute(0, 1, 4, 2, 3)                 # T, S, C, H, W
        appearance = (appearance - KINETICS_MEAN) / KINETICS_STD
        landmarks = torch.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)
        return appearance, landmarks, label


ckpt_path = CONFIG.R3D_CHECKPOINT or (os.environ.get("ASL_R3D_CKPT") and Path(os.environ["ASL_R3D_CKPT"]))
if not ckpt_path:
    hits = sorted(glob.glob("/kaggle/input/**/pilot_1000cls/best_model.pt", recursive=True))
    assert hits, "r3d_18 best_model.pt not found -- attach the previous r3d_18 notebook's output."
    ckpt_path = Path(hits[0])
print("R3D checkpoint:", ckpt_path)
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
r3d = DualStreamSLRModel(NUM_CLASSES)
r3d.load_state_dict(ck["model_state_dict"], strict=True)
r3d = r3d.to(DEVICE).eval()
print(f"loaded epoch {ck['epoch']} (val_top1 {ck['val_top1']:.3f}) -- strict load OK")

loader = DataLoader(DualStreamEvalDataset(test_samples), batch_size=CONFIG.BATCH_SIZE, shuffle=False,
                    num_workers=CONFIG.NUM_WORKERS, pin_memory=DEVICE == "cuda")
yt, yp, conf, top5 = [], [], [], []
t0 = time.time()
with torch.inference_mode():
    for app, lm, y in loader:
        probs = torch.softmax(r3d(app.to(DEVICE), lm.to(DEVICE)).float(), 1)
        c, p = probs.max(1)
        yt.append(y.numpy()); yp.append(p.cpu().numpy()); conf.append(c.cpu().numpy())
        top5.append(probs.topk(min(5, NUM_CLASSES), 1).indices.cpu().numpy())
test_seconds = time.time() - t0
preds = {"y_true": np.concatenate(yt), "y_pred": np.concatenate(yp), "confidence": np.concatenate(conf),
         "top5": np.concatenate(top5)}
test_metrics = summarize_predictions(preds["y_true"], preds["y_pred"], preds["top5"], NUM_CLASSES)
r3d_dir = CONFIG.OUT_DIR / "r3d18_dualstream_1000"
save_evaluation(r3d_dir, "r3d18_dualstream", preds, [v for v, _ in test_samples], CLASS_NAMES)
print(f"\nR3D-18 dual-stream TEST: top-1 {test_metrics['top1']:.3f} (recorded {CONFIG.RECORDED_TEST_TOP1}) | "
      f"top-5 {test_metrics['top5']:.3f} | macro-F1 {test_metrics['macro_f1']:.3f} | {test_seconds / 60:.1f} min")
if not CONFIG.SMOKE_TEST and abs(test_metrics["top1"] - CONFIG.RECORDED_TEST_TOP1) > 0.006:
    print("WARNING: reproduced top-1 differs from the recorded value by more than rounding -- investigate.")

export = r3d_dir / "app_export"
export.mkdir(parents=True, exist_ok=True)
state = {k: v.detach().cpu() for k, v in r3d.state_dict().items()}
torch.save(state, export / "model_weights.pt")
cpu_model = DualStreamSLRModel(NUM_CLASSES)
cpu_model.load_state_dict(state, strict=True)
app_ex, lm_ex, _ = DualStreamEvalDataset(test_samples[:1])[0]
prev =torch.get_num_threads(); torch.set_num_threads(2); cpu_model.eval()
with torch.inference_mode():
    cpu_model(app_ex[None], lm_ex[None]); ts = []
    for _ in range(3):
        t1 = time.perf_counter(); cpu_model(app_ex[None], lm_ex[None]); ts.append(time.perf_counter() - t1)
torch.set_num_threads(prev)
latency = {"cpu_threads": 2, "mean_s": float(np.mean(ts)), "runs": 3}
print(f"CPU latency (2 threads, 1 clip): {latency['mean_s']:.2f} s")

metrics = {"model_id": "r3d18_dualstream_1000", "num_classes": NUM_CLASSES, "class_fingerprint": CLASS_FINGERPRINT,
           "best_epoch": int(ck["epoch"]), "val_top1_best": float(ck["val_top1"]),
           "selection": "best val_loss (v9 notebook)", "test": test_metrics,
           "trainable_params_during_training": 27007080, "total_params": int(sum(p.numel() for p in state.values() if p.is_floating_point())),
           "gpu": torch.cuda.get_device_name(0) if N_GPUS else "cpu", "gpu_test_seconds_total": round(test_seconds, 1),
           "cpu_latency": latency, "smoke_test": CONFIG.SMOKE_TEST}
model_config = {"model_id": "r3d18_dualstream_1000", "display_name": "R3D-18 Dual-Stream (RGB + landmarks)",
                "architecture": "DualStreamSLRModel (v9): r3d_18 appearance stream + geometry stream -> Transformer",
                "num_classes": NUM_CLASSES, "class_fingerprint": CLASS_FINGERPRINT,
                "inputs": {"appearance": "B,T,1,3,112,112 -- cached signer crop /255, Kinetics mean/std",
                           "landmarks": "B,T,116 -- cached MediaPipe vector, NaN -> 0"},
                "frame_count": 16, "img_size": 112,
                "mean": [0.43216, 0.394666, 0.37645], "std": [0.22803, 0.22145, 0.216989]}
for name, obj in [("metrics.json", metrics), ("model_config.json", model_config)]:
    with open(export / name, "w") as f:
        json.dump(obj, f, indent=1)
for name in ["per_class_metrics.csv", "confusion_matrix.npz"]:
    (export / name).write_bytes((r3d_dir / name).read_bytes())
(export / "class_list.json").write_bytes((CONFIG.OUT_DIR / "class_list.json").read_bytes())
del r3d, cpu_model
'''))
    cells.append(md(r'''
## 5. (B) Parity sample pack
`CONFIG.N_SAMPLE_VIDEOS` test-split videos from distinct classes (fixed seed), each copied with its
cached `.npz`, plus `samples.csv` (video id, file, gloss, 1000-class index).
'''))
    cells.append(code(r'''
import shutil

video_files = {}
with open(CONFIG.SPLITS_DIR / "test.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        video_files[Path(row["Video file"]).stem] = row["Video file"]

pack = CONFIG.OUT_DIR / "parity_samples"
pack.mkdir(parents=True, exist_ok=True)
rng = random.Random(CONFIG.SEED)
pool = test_samples[:]
rng.shuffle(pool)
chosen, seen = [], set()
for vid, label in pool:
    if label in seen or vid not in video_files or not (VIDEOS_DIR / video_files[vid]).exists():
        continue
    chosen.append((vid, label)); seen.add(label)
    if len(chosen) >= CONFIG.N_SAMPLE_VIDEOS:
        break

rows = []
for vid, label in chosen:
    src = VIDEOS_DIR / video_files[vid]
    shutil.copy2(src, pack / src.name)
    shutil.copy2(cache_index[vid], pack / f"{vid}.npz")
    rows.append({"video_id": vid, "video_file": src.name, "gloss": CLASS_NAMES[label], "class_idx_1000": label})
pd.DataFrame(rows).to_csv(pack / "samples.csv", index=False)
size_mb = sum(p.stat().st_size for p in pack.iterdir()) / 1e6
print(f"Parity pack: {len(rows)} videos + npz ({size_mb:.1f} MB) -> {pack}")
print(pd.DataFrame(rows).head(10).to_string(index=False))
'''))
    return cells


if __name__ == "__main__":
    write(build_swin_notebook(), "train_1000_swin3d_t.ipynb")
    write(build_landmark_notebook(), "train_1000_landmark_models.ipynb")
    write(build_swin_eval_notebook(), "eval_1000_swin3d_t.ipynb")
    write(build_extras_notebook(), "extras_r3d_eval_and_samples.ipynb")
