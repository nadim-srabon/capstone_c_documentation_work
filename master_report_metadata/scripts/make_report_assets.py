"""Build the report-asset folder: generated figures, renamed notebook figures, tables, data, metadata.

    .venv/Scripts/python scripts/make_report_assets.py <out_dir>

Every number comes from results/ (copied from the Kaggle evaluation outputs) or docs/*.json
(local measurements). Original notebook figures are read from the Kaggle output zips.
"""
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inference.config import COMPARISON_1000, MODELS, SCALING_SERIES  # noqa: E402

OUT = Path(sys.argv[1])
RES = ROOT / "results"
THESIS = ("Building a Word-Level American Sign Language Recognition System: "
          "A Multi-Architecture Design and Evaluation Study")

# Validated categorical palette (light mode), fixed per model -- never by rank.
COL = {"swin3d_t_1000": "#2a78d6", "geometry_transformer_1000": "#eb6834", "stgcn_1000": "#1baf7a",
       "r3d18_dualstream_1000": "#eda100"}
SEQ3 = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#1f2430", "#5a6170", "#e6e8ee"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": "#c9ced8", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 100,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})

INDEX = []   # rows for figure_index.csv
TICK = {"swin3d_t_1000": "Video\nSwin-T", "geometry_transformer_1000": "Geometry\nTransformer",
        "stgcn_1000": "ST-GCN", "r3d18_dualstream_1000": "R3D-18\nDual-Stream"}


def metrics(mid):
    return json.loads((RES / mid / "metrics.json").read_text())


def history(mid):
    return pd.DataFrame(json.loads((RES / mid / "training_history.json").read_text()))


def name(mid):
    s = MODELS[mid]
    return f"{s.display_name} ({s.num_classes})" if mid in SCALING_SERIES else s.display_name


def save(fig, folder, fname, title, caption, source, pdf=True):
    d = OUT / "A_generated_figures" / folder
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{fname}.png")
    if pdf:
        fig.savefig(d / f"{fname}.pdf")
    plt.close(fig)
    INDEX.append({"file": f"A_generated_figures/{folder}/{fname}.png", "pdf_vector": "yes" if pdf else "no",
                  "title": title, "caption": caption, "source": source, "contains_people": "no"})


def bar_labels(ax, bars, fmt="{:.1%}", dy=0.01):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=8.5, color=INK2)


# ======================================================================== A1 dataset
def fig_dataset():
    ds = json.loads((RES / "dataset.json").read_text())
    s = ds["subset_1000"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(["Train", "Validation", "Test"], [s["train"], s["val"], s["test"]],
                  color=["#2a78d6", "#eb6834", "#1baf7a"], width=0.55, edgecolor="white", linewidth=2)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 200, f"{int(b.get_height()):,}", ha="center",
                fontsize=9, color=INK2)
    ax.set_ylabel("Video clips")
    ax.set_title("ASL Citizen 1,000-sign subset: clips per split")
    ax.set_ylim(0, s["train"] * 1.15)
    save(fig, "01_dataset", "fig01_01_dataset_split_sizes_1000",
         "Clips per split, 1,000-sign subset",
         f"Number of video clips in the official signer-independent train / validation / test splits of the "
         f"1,000-sign ASL Citizen subset ({s['videos']:,} clips in total, 30-32 clips per sign).",
         "results/dataset.json (r3d_18 notebook manifest output)")

    ks, tests = [], []
    for mid in SCALING_SERIES:
        pc = pd.read_csv(RES / mid / "per_class_metrics.csv")
        ks.append(MODELS[mid].num_classes)
        tests.append(int(metrics(mid)["test"].get("n_samples") or pc["support"].sum()))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar([str(k) for k in ks], tests, color="#2a78d6", width=0.55, edgecolor="white", linewidth=2)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 200, f"{int(b.get_height()):,}", ha="center",
                fontsize=9, color=INK2)
    ax.set_xlabel("Vocabulary size (number of signs)")
    ax.set_ylabel("Test clips")
    ax.set_title("Test-set size per vocabulary size")
    save(fig, "01_dataset", "fig01_02_test_clips_per_vocabulary_size",
         "Test clips per vocabulary size",
         "Number of held-out test clips for each vocabulary size used in the scaling study (the 300, 500, "
         "1,000 and 1,200 most frequent glosses).",
         "results/*/per_class_metrics.csv (support column) and metrics.json")


# ======================================================================== A2 pipeline / architectures
def _box(ax, x, y, w, h, title, sub, fc="#eef3fc", ec="#2a78d6"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=7.6, color=INK2, wrap=True)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#7a8394", lw=1.3))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 2.6); ax.axis("off")
    steps = [("Video input", "one isolated sign\n(up to 20 s)"),
             ("Frame sampling", "16 frames,\nlinspace(0, n-1, 16)"),
             ("MediaPipe", "hands, pose, face\nlandmarks"),
             ("Preprocessing", "normalise 58 points;\n112x112 signer crop"),
             ("Model", "3D-CNN / video\nTransformer / GCN"),
             ("Classifier", "linear head +\nsoftmax"),
             ("Prediction", "top-1 sign +\nconfidence, top-5")]
    w, gap = 1.6, 0.25
    for i, (t, s) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        _box(ax, x, 0.55, w, 1.5, t, s)
        if i:
            _arrow(ax, x - gap + 0.02, 1.3, x - 0.02, 1.3)
    save(fig, "02_methodology", "fig02_01_recognition_pipeline",
         "End-to-end recognition pipeline",
         "Pipeline shared by training-data preparation and the web application: 16 uniformly sampled frames, "
         "MediaPipe landmark extraction, landmark normalisation and signer cropping, a recognition model, and a "
         "softmax classifier over the sign vocabulary.", "Design figure (inference/video_processor.py)")

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")
    rows = [
        ("swin3d_t_1000", "RGB crops\n16 x 3 x 224 x 224", "Video Swin-T\n3D shifted-window attention",
         "Linear 768 -> 1000"),
        ("r3d18_dualstream_1000", "RGB crops 16x112x112\n+ landmarks 16 x 116",
         "R3D-18 (RGB) + geometry MLP\n-> 3-layer Transformer", "Linear 256 -> 1000"),
        ("geometry_transformer_1000", "Landmarks\n16 x 116 (+ velocity)", "Geometry Transformer\n3 layers, 4 heads",
         "Linear 256 -> 1000"),
        ("stgcn_1000", "Landmarks\n58 joints x 16 frames", "ST-GCN\n7 graph/temporal blocks", "Linear 256 -> 1000"),
    ]
    for r, (mid, inp, body, head) in enumerate(rows):
        y = 4.2 - r * 1.3
        c = COL[mid]
        ax.text(0.05, y + 0.45, MODELS[mid].display_name, fontsize=10, fontweight="bold", color=c, va="center")
        ax.text(0.05, y + 0.12, MODELS[mid].family, fontsize=8, color=INK2, va="center")
        _box(ax, 2.6, y, 2.5, 0.95, "Input", inp, fc="#f7f8fa", ec="#c9ced8")
        _box(ax, 5.6, y, 3.2, 0.95, "Backbone", body, fc="#f7f8fa", ec=c)
        _box(ax, 9.3, y, 2.5, 0.95, "Classifier", head, fc="#f7f8fa", ec="#c9ced8")
        _arrow(ax, 5.1, y + 0.47, 5.58, y + 0.47)
        _arrow(ax, 8.8, y + 0.47, 9.28, y + 0.47)
    save(fig, "02_methodology", "fig02_02_architecture_overview",
         "Overview of the four 1,000-sign architectures",
         "Input representation, backbone and classification head of the four architectures compared on the "
         "1,000-sign vocabulary.", "inference/architectures.py, model_config.json files")


# ======================================================================== A3 training curves
def fig_training():
    for i, mid in enumerate(COMPARISON_1000 + SCALING_SERIES[:2] + SCALING_SERIES[3:], start=1):
        h = history(mid)
        m = metrics(mid)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
        a1.plot(h.epoch, h.train_loss, color=SEQ3[0], lw=2, label="Train")
        a1.plot(h.epoch, h.val_loss, color=SEQ3[1], lw=2, label="Validation")
        a1.set_title("Loss (label-smoothed cross-entropy)"); a1.set_xlabel("Epoch"); a1.legend()
        a2.plot(h.epoch, h.train_top1, color=SEQ3[0], lw=2, label="Train top-1")
        a2.plot(h.epoch, h.val_top1, color=SEQ3[1], lw=2, label="Validation top-1")
        be = m.get("best_epoch")
        if be:
            a2.axvline(be, color="#9aa1ad", ls="--", lw=1)
            a2.text(be, 0.02, f" selected epoch {be}", color=INK2, fontsize=8, va="bottom")
        a2.set_ylim(0, 1.02); a2.set_title("Top-1 accuracy"); a2.set_xlabel("Epoch"); a2.legend(loc="lower right")
        fig.suptitle(f"{name(mid)} -- training history", fontweight="bold", y=1.03)
        sel = "lowest validation loss" if mid == "r3d18_dualstream_1000" else "highest validation top-1"
        save(fig, "03_training_curves", f"fig03_{i:02d}_training_curves_{mid}",
             f"Training curves: {name(mid)}",
             f"Loss and top-1 accuracy per epoch on the training and validation splits for {name(mid)}. The dashed "
             f"line marks the checkpoint used for testing (selected on {sel}).",
             f"results/{mid}/training_history.json")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for mid in COMPARISON_1000:
        h = history(mid)
        ax.plot(h.epoch, h.val_top1, color=COL[mid], lw=2, label=name(mid))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation top-1"); ax.set_ylim(0, 0.8)
    ax.set_title("Validation top-1 during training, 1,000 signs"); ax.legend()
    save(fig, "03_training_curves", "fig03_08_validation_top1_all_1000_models",
         "Validation accuracy of the four 1,000-sign models",
         "Validation top-1 accuracy per epoch for all four 1,000-sign architectures (same validation clips).",
         "results/*_1000/training_history.json")


# ======================================================================== A4 comparison
def fig_comparison():
    ms = {k: metrics(k) for k in COMPARISON_1000}
    labels = [TICK[k] for k in COMPARISON_1000]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - 0.19, [ms[k]["test"]["top1"] for k in COMPARISON_1000], 0.36, color="#2a78d6", label="Top-1",
                edgecolor="white", linewidth=2)
    b2 = ax.bar(x + 0.19, [ms[k]["test"]["top5"] for k in COMPARISON_1000], 0.36, color="#9cc2f0", label="Top-5",
                edgecolor="white", linewidth=2)
    bar_labels(ax, b1); bar_labels(ax, b2)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.05); ax.set_ylabel("Test accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Test accuracy, 1,000 signs (12,560 clips)"); ax.legend()
    save(fig, "04_model_comparison_1000", "fig04_01_test_top1_top5_accuracy",
         "Top-1 and top-5 test accuracy, 1,000 signs",
         "Top-1 and top-5 accuracy of the four architectures on the 12,560-clip signer-independent test split "
         "(identical class list and clips for all models).", "results/*_1000/metrics.json")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j, (key, lab) in enumerate([("macro_precision", "Precision"), ("macro_recall", "Recall"),
                                    ("macro_f1", "F1-score")]):
        bars = ax.bar(x + (j - 1) * 0.26, [ms[k]["test"][key] for k in COMPARISON_1000], 0.25, color=SEQ3[j],
                      label=lab, edgecolor="white", linewidth=2)
        bar_labels(ax, bars, "{:.3f}", 0.008)
    ax.set_xticks(x, labels); ax.set_ylim(0, 0.85); ax.set_ylabel("Macro-averaged score")
    ax.set_title("Macro precision, recall and F1, 1,000 signs"); ax.legend(ncol=3)
    save(fig, "04_model_comparison_1000", "fig04_02_macro_precision_recall_f1",
         "Macro precision, recall and F1, 1,000 signs",
         "Precision, recall and F1-score macro-averaged over the 1,000 classes on the test split.",
         "results/*_1000/metrics.json")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for k in COMPARISON_1000:
        lat = ms[k]["cpu_latency"]["mean_s"] * 1000
        p = ms[k].get("total_params") or ms[k].get("trainable_params")
        ax.scatter(lat, ms[k]["test"]["top1"], s=60 + 9 * p / 1e6, color=COL[k], edgecolor="white", lw=2, zorder=3)
        ax.annotate(f"{MODELS[k].display_name}\n{p / 1e6:.1f} M params", (lat, ms[k]["test"]["top1"]), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=8.5, color=INK)
    ax.set_xscale("log"); ax.set_xlabel("CPU latency per clip, ms (2 threads, model forward only; log scale)")
    ax.set_ylabel("Test top-1"); ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Accuracy versus CPU cost")
    save(fig, "04_model_comparison_1000", "fig04_03_accuracy_vs_cpu_latency",
         "Accuracy versus CPU latency",
         "Test top-1 accuracy against single-clip CPU inference latency (Kaggle CPU, 2 threads, forward pass only; "
         "MediaPipe extraction excluded). Marker area grows with parameter count.",
         "results/*_1000/metrics.json (cpu_latency, total_params)")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    data = [pd.read_csv(RES / k / "per_class_metrics.csv")["f1"].values for k in COMPARISON_1000]
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.5,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=INK, markersize=5),
                    medianprops=dict(color=INK, lw=1.5))
    for patch, k in zip(bp["boxes"], COMPARISON_1000):
        patch.set_facecolor(COL[k]); patch.set_alpha(0.85)
    ax.set_xticks(range(1, 5), labels); ax.set_ylabel("Per-class F1 (test)"); ax.set_ylim(0, 1.02)
    ax.set_title("Distribution of per-class F1 over the 1,000 signs")
    save(fig, "04_model_comparison_1000", "fig04_04_per_class_f1_distribution",
         "Per-class F1 distribution",
         "Box plots of the test F1-score of each of the 1,000 classes (box = quartiles, line = median, "
         "diamond = mean).", "results/*_1000/per_class_metrics.csv")


# ======================================================================== A5 scaling
def fig_scaling():
    ks = [MODELS[m].num_classes for m in SCALING_SERIES]
    ms = [metrics(m)["test"] for m in SCALING_SERIES]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (key, lab), c in zip([("top5", "Test top-5"), ("top1", "Test top-1"), ("macro_f1", "Macro F1")], SEQ3):
        ys = [m[key] for m in ms]
        ax.plot(ks, ys, "-o", color=c, lw=2, ms=7, mec="white", mew=1.5, label=lab)
        below = key == "macro_f1"                  # F1 tracks top-1 closely: label it underneath
        for kx, yv in zip(ks, ys):
            ax.text(kx, yv - 0.022 if below else yv + 0.012, f"{yv:.3f}" if below else f"{yv:.1%}", ha="center",
                    va="top" if below else "bottom", fontsize=8, color=c)
    ax.set_xticks(ks); ax.set_xlabel("Vocabulary size (number of signs)"); ax.set_ylabel("Test score")
    ax.set_ylim(0.45, 0.92); ax.set_title("Geometry Transformer: effect of vocabulary size"); ax.legend()
    save(fig, "05_vocabulary_scaling", "fig05_01_scaling_accuracy_vs_vocabulary",
         "Accuracy versus vocabulary size",
         "Test top-1, top-5 and macro-F1 of the Geometry Transformer trained separately on the 300, 500, 1,000 and "
         "1,200 most frequent signs.", "results/geometry_transformer_*/metrics.json")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for mid, c in zip(SCALING_SERIES, ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]):
        h = history(mid)
        ax.plot(h.epoch, h.val_top1, color=c, lw=2, label=f"{MODELS[mid].num_classes} signs")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation top-1"); ax.set_ylim(0, 0.8)
    ax.set_title("Geometry Transformer: validation accuracy per vocabulary size"); ax.legend()
    save(fig, "05_vocabulary_scaling", "fig05_02_scaling_validation_curves",
         "Validation curves per vocabulary size",
         "Validation top-1 accuracy per epoch of the Geometry Transformer for each vocabulary size.",
         "results/geometry_transformer_*/training_history.json")


# ======================================================================== A6 error analysis
def fig_errors():
    classes = json.loads((RES / "class_lists" / "classes_1000.json").read_text(encoding="utf-8"))["classes"]
    for i, mid in enumerate(COMPARISON_1000, start=1):
        cm = np.load(RES / mid / "confusion_matrix.npz")["cm"].astype(np.int64)
        off = cm.copy(); np.fill_diagonal(off, 0)
        flat = np.argsort(off, axis=None)[::-1][:15]
        ti, pi = np.unravel_index(flat, off.shape)
        labs = [f"{classes[a]} -> {classes[b]}" for a, b in zip(ti, pi)][::-1]
        vals = [int(off[a, b]) for a, b in zip(ti, pi)][::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(labs, vals, color=COL[mid], edgecolor="white", linewidth=2)
        ax.set_xlabel("Misclassified test clips (true -> predicted)")
        ax.set_title(f"{name(mid)}: 15 most-confused sign pairs"); ax.grid(axis="y", visible=False)
        save(fig, "06_error_analysis", f"fig06_{i}a_top_confused_pairs_{mid}",
             f"Most-confused sign pairs: {name(mid)}",
             f"The 15 (true, predicted) sign pairs with the most test-set errors for {name(mid)}.",
             f"results/{mid}/confusion_matrix.npz")

        involved = off.sum(0) + off.sum(1)
        idx = np.argsort(involved)[::-1][:25]
        sub = cm[np.ix_(idx, idx)]
        fig, ax = plt.subplots(figsize=(8.2, 7))
        im = ax.imshow(sub, cmap="Blues")
        ax.set_xticks(range(25), [classes[j] for j in idx], rotation=70, ha="right", fontsize=7)
        ax.set_yticks(range(25), [classes[j] for j in idx], fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04, label="Test clips")
        ax.set_title(f"{name(mid)}: confusion matrix of the 25 classes most involved in errors", fontsize=10.5)
        save(fig, "06_error_analysis", f"fig06_{i}b_confusion_submatrix_{mid}",
             f"Confusion sub-matrix: {name(mid)}",
             f"Rows/columns of the 1,000 x 1,000 test confusion matrix of {name(mid)} for the 25 classes with the "
             f"most errors (as true or predicted class).", f"results/{mid}/confusion_matrix.npz")

        pc = pd.read_csv(RES / mid / "per_class_metrics.csv").sort_values(["f1", "support"], ascending=[True, False])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
        worst, best = pc.head(15)[::-1], pc.tail(15)
        a1.barh(worst["class_name"], worst["f1"], color="#e34948", edgecolor="white")
        a1.set_title("15 hardest signs (lowest F1)"); a1.set_xlim(0, 1)
        a2.barh(best["class_name"], best["f1"], color="#1baf7a", edgecolor="white")
        a2.set_title("15 easiest signs (highest F1)"); a2.set_xlim(0, 1.12)
        for a, dfp in ((a1, worst), (a2, best)):
            for yi, v in enumerate(dfp["f1"]):
                a.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7, color=INK2)
            a.set_xlabel("Test F1"); a.grid(axis="y", visible=False); a.tick_params(axis="y", labelsize=7.5)
        fig.suptitle(f"{name(mid)}: easiest and hardest signs", fontweight="bold", y=1.02)
        save(fig, "06_error_analysis", f"fig06_{i}c_best_worst_classes_{mid}",
             f"Easiest and hardest signs: {name(mid)}",
             f"Per-class test F1 of the 15 lowest- and 15 highest-scoring signs for {name(mid)} (ties broken by "
             f"support).", f"results/{mid}/per_class_metrics.csv")


# ======================================================================== A7 deployment / testing
def fig_deployment():
    fresh = json.loads((ROOT / "docs/parity_results_fresh.json").read_text())
    shared = json.loads((ROOT / "docs/parity_results_shared.json").read_text())
    labels = [TICK[k] for k in COMPARISON_1000]
    x = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, title in [(a1, "top1_agreement_cache_vs_app", "Same top-1 as on training-cache features"),
                          (a2, "repeatable_top1_pct", "Same top-1 when the same video is run twice")]:
        b1 = a.bar(x - 0.19, [fresh["models"][k][key] for k in COMPARISON_1000], 0.36, color="#2a78d6",
                   label="Fresh MediaPipe per video (app)", edgecolor="white", linewidth=2)
        b2 = a.bar(x + 0.19, [shared["models"][k][key] for k in COMPARISON_1000], 0.36, color="#c3c9d4",
                   label="Shared MediaPipe (as in cache builder)", edgecolor="white", linewidth=2)
        bar_labels(a, b1, "{:.0%}"); bar_labels(a, b2, "{:.0%}")
        a.set_xticks(x, labels, fontsize=8.5); a.set_ylim(0, 1.15); a.set_title(title, fontsize=10.5)
        a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a1.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Preprocessing parity test on {fresh['n_videos']} held-out test videos", fontweight="bold", y=1.02)
    save(fig, "07_deployment_testing", "fig07_01_preprocessing_parity_test",
         "Parity of the web-app pipeline with the training cache",
         f"Left: share of {fresh['n_videos']} raw test videos for which each model's top-1 prediction from the "
         "app's feature extraction matches its prediction from the cached training features. Right: determinism "
         "when the same video is processed twice. The app uses fresh MediaPipe trackers per video.",
         "docs/parity_results_fresh.json, docs/parity_results_shared.json (scripts/parity_test.py)")

    ml = json.loads((ROOT / "docs/memory_latency_local.json").read_text())
    order = [k for k in ml["models"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [ml["models"][k]["infer_ms_2threads"] for k in order]
    bars = ax.barh([name(k) for k in order][::-1], vals[::-1],
                   color=[COL.get(k, "#eb6834") for k in order][::-1], edgecolor="white", linewidth=2)
    for b in bars:
        ax.text(b.get_width() * 1.08, b.get_y() + b.get_height() / 2, f"{b.get_width():,.0f} ms", va="center",
                fontsize=8.5, color=INK2)
    ax.set_xscale("log"); ax.set_xlabel("Model inference per clip on local CPU, ms (2 threads, log scale)")
    ax.set_title("Inference time of each deployed model"); ax.grid(axis="y", visible=False)
    ax.set_xlim(1, max(vals) * 6)
    save(fig, "07_deployment_testing", "fig07_02_local_cpu_inference_time",
         "Local CPU inference time per model",
         f"Median forward-pass time for one 16-frame clip on the development laptop CPU (2 threads), excluding "
         f"MediaPipe feature extraction ({ml['feature_extraction_s']:.1f} s per clip).",
         "docs/memory_latency_local.json")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    stages = ["Python + PyTorch", "+ MediaPipe\n+ one video", "+ Video Swin-T\n(largest)", "+ landmark model"]
    vals = [ml["baseline_mb"], ml["with_mediapipe_mb"], ml["models"]["swin3d_t_1000"]["rss_peak_mb"],
            ml["models"]["geometry_transformer_1000"]["rss_peak_mb"]]
    bars = ax.bar(stages, vals, color=["#c3c9d4", "#9cc2f0", "#2a78d6", "#eb6834"], width=0.55, edgecolor="white",
                  linewidth=2)
    bar_labels(ax, bars, "{:,.0f} MB", 15)
    ax.set_ylabel("Process memory (RSS, MB)"); ax.set_ylim(0, max(vals) * 1.2)
    ax.set_title("Memory footprint of the web application (one model loaded at a time)")
    save(fig, "07_deployment_testing", "fig07_03_memory_footprint",
         "Memory footprint of the deployed application",
         "Resident memory of the application process after each stage, measured locally. Only one model is kept "
         "in memory at a time.", "docs/memory_latency_local.json")


# ======================================================================== B original notebook figures
V10_FILES = [
    ("v10_train_val_curves.png", "training_curves", "Training/validation loss and top-1 accuracy (notebook figure)."),
    ("v10_top1_vs_top5.png", "test_top1_vs_top5", "Test top-1 versus top-5 accuracy."),
    ("v10_geometry_only_{k}cls_vs_v9_head_to_head.png", "head_to_head_vs_v9_dualstream",
     "Head-to-head of the v10 Geometry Transformer and the v9 dual-stream baseline at the same vocabulary size."),
    ("v10_geometry_only_top_confusion_pairs.png", "top_confusion_pairs", "Most-confused class pairs on the test split."),
    ("v10_geometry_only_confusion_heatmap.png", "confusion_heatmap_top_confused_classes",
     "Confusion heatmap restricted to the classes most involved in errors."),
    ("v10_geometry_only_best_18_performing_classes.png", "best_18_classes", "The 18 best-performing classes."),
    ("v10_geometry_only_worst_18_performing_classes.png", "worst_18_classes", "The 18 worst-performing classes."),
    ("v10_geometry_only_support_vs_performance.png", "support_vs_performance",
     "Per-class test support versus performance."),
    ("v10_geometry_only_confidence_analysis.png", "confidence_correct_vs_incorrect",
     "Distribution of prediction confidence for correct and incorrect predictions."),
]
V10_PEOPLE = [("[v10_geometry_only]_Correct_Predictions.png", "examples_correct_predictions"),
              ("[v10_geometry_only]_Incorrect_Predictions.png", "examples_incorrect_predictions"),
              ("[v10_geometry_only]_Hardest_Errors_(High-Confidence_Mistakes).png", "examples_hardest_errors")]


def copy_originals():
    zips = {"300": "v10_checkpoints/300 class.zip", "500": "v10_checkpoints/500 class.zip",
            "1200": "v10_checkpoints/1200 class.zip"}
    jobs = []   # (zip, member, folder, fname, title, caption, people)
    for k, z in zips.items():
        for src, short, cap in V10_FILES:
            jobs.append((z, "error_analysis_report/" + src.format(k=k), "geometry_transformer_v10_scaling",
                         f"geomtransformer_v10_{k}cls__{short}", f"Geometry Transformer v10, {k} signs: {short.replace('_', ' ')}",
                         cap, False))
        jobs.append((z, "error_analysis_report/v9_vs_v10_precision_recall_f1.png", "geometry_transformer_v10_scaling",
                     f"geomtransformer_v10_{k}cls__precision_recall_f1_vs_v9", f"Geometry Transformer v10, {k} signs: P/R/F1 vs v9",
                     "Macro precision/recall/F1 of v10 versus v9." + (
                         "" if k == "300" else " WARNING: the v9 bars in this figure are the 300-class v9 values "
                         "(the notebook reused them for every class count); do not cite them for this vocabulary."),
                     False))
        for src, short in V10_PEOPLE:
            jobs.append((z, "error_analysis_report/" + src, None, f"geomtransformer_v10_{k}cls__{short}",
                         f"Geometry Transformer v10, {k} signs: {short.replace('_', ' ')}",
                         "Example test clips with true (T) and predicted (P) labels; shows ASL Citizen signers.", True))
    r3d = "previous_model_r3d_18/previous_model_r3d_18.zip"
    for member, fname, title, cap, people in [
        ("__results___files/__results___7_0.png", "r3d18_v9__dataset_video_count_per_split",
         "Videos per split (1,000-sign subset)", "Video count per split for the 1,000-sign subset.", False),
        ("__results___files/__results___8_0.png", "r3d18_v9__dataset_videos_per_class_distribution",
         "Videos per class", "Distribution of videos per class and the 15 most-represented classes.", False),
        ("__results___files/__results___9_1.png", "r3d18_v9__dataset_video_properties_fps_frames_resolution",
         "Video properties", "FPS, frame-count and resolution of 300 randomly sampled ASL Citizen clips.", False),
        ("__results___files/__results___15_0.png", "r3d18_v9__mediapipe_hand_landmark_indices_schematic",
         "MediaPipe hand landmarks", "Schematic of the 21 MediaPipe hand landmark indices.", False),
        ("__results___files/__results___27_0.png", "r3d18_v9__training_curves",
         "R3D-18 dual-stream training curves", "Loss and top-1 accuracy per epoch (Capstone B, v9).", False),
        ("__results___files/__results___18_2.png", "r3d18_v9__sampled_signer_crops_example",
         "Sampled signer crops", "The 16 sampled signer-bounding-box crops of one clip.", True),
        ("__results___files/__results___19_1.png", "r3d18_v9__signer_crops_gloss_WEAR",
         "Signer crops, gloss WEAR", "Sampled signer crops for a clip of the sign WEAR.", True),
    ]:
        jobs.append((r3d, member, "r3d18_dualstream_v9", fname, title, cap, people))
    lm = "landmark_model_1000_class/landmark_models.zip"
    jobs.append((lm, "landmark_models_1000cls/geometry_transformer_v10/training_curves.png", "landmark_models_1000",
                 "geomtransformer_v10_1000cls__training_curves", "Geometry Transformer, 1,000 signs: training curves",
                 "Loss and accuracy per epoch (notebook figure).", False))
    jobs.append((lm, "landmark_models_1000cls/stgcn/training_curves.png", "landmark_models_1000",
                 "stgcn_1000cls__training_curves", "ST-GCN, 1,000 signs: training curves",
                 "Loss and accuracy per epoch (notebook figure).", False))
    sw = "swin_3d_1000_class/eval_swin3d.zip"
    jobs.append((sw, "swin3d_t_1000cls/training_curves.png", "video_swin_t_1000", "swin3d_t_1000cls__training_curves",
                 "Video Swin-T, 1,000 signs: training curves", "Loss and accuracy per epoch (notebook figure).", False))
    jobs.append((sw, "swin3d_t_1000cls/appearance_probe.png", "video_swin_t_1000", "swin3d_t_1000cls__appearance_probe",
                 "Video Swin-T input probe", "Frames of the cached RGB appearance stream used to verify colour order.",
                 True))
    for z, member, folder, fname, title, cap, people in jobs:
        with zipfile.ZipFile(ROOT / z) as f:
            data = f.read(member)
        if people:
            rel = Path("D_RESTRICTED_contains_dataset_signers") / f"{fname}.png"
        else:
            rel = Path("B_original_notebook_figures") / folder / f"{fname}.png"
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        (OUT / rel).write_bytes(data)
        INDEX.append({"file": rel.as_posix(), "pdf_vector": "no", "title": title, "caption": cap,
                      "source": f"{z} :: {member}", "contains_people": "YES -- ASL Citizen signers" if people else "no"})


# ======================================================================== tables / data / metadata
def write_table(df, stem, caption):
    d = OUT / "tables"; d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{stem}.csv", index=False)
    (d / f"{stem}.md").write_text(f"**{caption}**\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
    tex = df.to_latex(index=False, escape=True, caption=caption, label=f"tab:{stem}", position="htbp")
    (d / f"{stem}.tex").write_text(tex, encoding="utf-8")


def tables():
    rows = []
    for k in COMPARISON_1000 + [m for m in SCALING_SERIES if m not in COMPARISON_1000]:
        m, s = metrics(k), MODELS[k]
        t = m["test"]
        lat = m.get("cpu_latency", {}).get("mean_s")
        rows.append({"Model": s.display_name, "Input": s.family, "Classes": s.num_classes,
                     "Params (M)": round((m.get("total_params") or m.get("trainable_params")) / 1e6, 2),
                     "Val top-1": round(m["val_top1_best"], 4), "Test top-1": round(t["top1"], 4),
                     "Test top-5": round(t["top5"], 4), "Macro P": round(t["macro_precision"], 4),
                     "Macro R": round(t["macro_recall"], 4), "Macro F1": round(t["macro_f1"], 4),
                     "Weighted F1": round(t["weighted_f1"], 4), "Best epoch": m.get("best_epoch"),
                     "CPU latency (ms)": round(lat * 1000, 1) if lat else None})
    df = pd.DataFrame(rows)
    write_table(df[df["Classes"] == 1000].drop(columns=["Classes"]), "table01_model_comparison_1000",
                "Test results of the four architectures on 1,000 signs (12,560 test clips)")
    write_table(df[df["Model"] == "Geometry Transformer"][["Classes", "Params (M)", "Val top-1", "Test top-1",
                                                           "Test top-5", "Macro P", "Macro R", "Macro F1"]],
                "table02_vocabulary_scaling", "Geometry Transformer results per vocabulary size")
    write_table(df, "table00_all_models_all_metrics", "All metrics of all seven trained models")

    st = json.loads((ROOT / "kaggle_outputs/landmark_1000/landmark_models_1000cls/stgcn/app_export/model_config.json")
                    .read_text())["hparams"]
    gt = json.loads((ROOT / "kaggle_outputs/landmark_1000/landmark_models_1000cls/geometry_transformer_v10/app_export/"
                     "model_config.json").read_text())["hparams"]
    sw = json.loads((ROOT / "kaggle_outputs/swin3d_1000/swin3d_t_1000cls/app_export/model_config.json").read_text())
    hp = pd.DataFrame([
        {"Model": "Video Swin-T", "Init": "Kinetics-400", "Input": "16x3x224x224 RGB", "Batch": sw["train_hparams"]["BATCH_SIZE"],
         "Epochs": sw["train_hparams"]["EPOCHS"], "LR": f"{sw['train_hparams']['LR_BACKBONE']} backbone / {sw['train_hparams']['LR_HEAD']} head",
         "Weight decay": sw["train_hparams"]["WEIGHT_DECAY"], "Label smoothing": sw["train_hparams"]["LABEL_SMOOTHING"],
         "Other": f"stochastic depth {sw['train_hparams']['STOCHASTIC_DEPTH']}, head dropout {sw['head_dropout']}, "
                  f"scale aug >= {sw['train_hparams']['AUG_SCALE_MIN']}, no h-flip", "Selection": "best val top-1"},
        {"Model": "R3D-18 Dual-Stream (v9)", "Init": "Kinetics-400 (frozen to layer3)", "Input": "16x112x112 RGB + 16x116",
         "Batch": 8, "Epochs": 50, "LR": "1e-4 (backbone 1e-5)", "Weight decay": 3e-4, "Label smoothing": 0.1,
         "Other": "dropout 0.5, grad clip 0.5, cosine schedule, class-balanced sampler, rotation/colour aug",
         "Selection": "lowest val loss"},
        {"Model": "Geometry Transformer (v10)", "Init": "random", "Input": "16x116 landmarks", "Batch": gt["batch_size"],
         "Epochs": gt["epochs"], "LR": gt["lr"], "Weight decay": gt["weight_decay"], "Label smoothing": gt["label_smoothing"],
         "Other": f"grad clip {gt['grad_clip']}, {gt['scheduler']}, patience {gt['patience']}", "Selection": "best val top-1"},
        {"Model": "ST-GCN", "Init": "random", "Input": "58 joints x 16 frames", "Batch": st["batch_size"],
         "Epochs": st["epochs"], "LR": st["lr"], "Weight decay": st["weight_decay"], "Label smoothing": st["label_smoothing"],
         "Other": f"grad clip {st['grad_clip']}, {st['warmup_epochs']} warm-up epochs, {st['scheduler']}, "
                  f"augmentation {'on' if st['augment'] else 'off'}, patience {st['patience']}", "Selection": "best val top-1"},
    ])
    write_table(hp, "table03_training_hyperparameters", "Training configuration of the 1,000-sign models")

    ds = json.loads((RES / "dataset.json").read_text())
    s = ds["subset_1000"]
    write_table(pd.DataFrame([
        {"Item": "Dataset", "Value": f"{ds['name']} ({ds['publisher']})"},
        {"Item": "Videos in feature cache", "Value": f"{ds['cached_videos']:,}"},
        {"Item": "Glosses in cache", "Value": f"{ds['glosses_in_cache']:,}"},
        {"Item": "1,000-sign subset videos", "Value": f"{s['videos']:,}"},
        {"Item": "Train / val / test clips", "Value": f"{s['train']:,} / {s['val']:,} / {s['test']:,}"},
        {"Item": "Clips per sign", "Value": f"{s['videos_per_class_min']}-{s['videos_per_class_max']}"},
        {"Item": "Frames per clip (model input)", "Value": str(ds["frames_per_clip"])},
        {"Item": "Typical source video", "Value": f"{ds['typical_video']['resolution']}, ~{ds['typical_video']['fps_median']:.0f} fps, "
                                                  f"median {ds['typical_video']['frames_median']} frames"},
        {"Item": "Split policy", "Value": ds["split_policy"]},
    ]), "table04_dataset_summary", "Dataset summary")

    fresh = json.loads((ROOT / "docs/parity_results_fresh.json").read_text())
    shared = json.loads((ROOT / "docs/parity_results_shared.json").read_text())
    write_table(pd.DataFrame([{
        "Model": name(k),
        "Agreement (fresh)": f"{fresh['models'][k]['top1_agreement_cache_vs_app']:.0%}",
        "Agreement (shared)": f"{shared['models'][k]['top1_agreement_cache_vs_app']:.0%}",
        "Repeatable (fresh)": f"{fresh['models'][k]['repeatable_top1_pct']:.0%}",
        "Repeatable (shared)": f"{shared['models'][k]['repeatable_top1_pct']:.0%}",
        "Correct on cache / app (fresh, of 30)": f"{fresh['models'][k]['correct_on_cached_features']} / "
                                                 f"{fresh['models'][k]['correct_on_app_features']}",
    } for k in COMPARISON_1000]), "table05_preprocessing_parity", "Parity of the web-app pipeline with the training cache (30 test videos)")

    ml = json.loads((ROOT / "docs/memory_latency_local.json").read_text())
    write_table(pd.DataFrame([{"Model": name(k), "Load (s)": v["load_s"], "Inference, 2 threads (ms)": v["infer_ms_2threads"],
                               "Peak RSS (MB)": v["rss_peak_mb"]} for k, v in ml["models"].items()]),
                "table06_local_latency_memory", "Local CPU load time, inference time and memory per model")


def data_and_meta():
    d = OUT / "data"
    shutil.copytree(RES, d / "evaluation_results", dirs_exist_ok=True)
    for f in ["parity_results_fresh.json", "parity_results_shared.json", "memory_latency_local.json"]:
        (d / "deployment_measurements").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "docs" / f, d / "deployment_measurements" / f)
    (OUT / "documents").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "docs/testing_report.md", OUT / "documents/testing_report.md")
    shutil.copy2(ROOT / "README.md", OUT / "documents/project_README.md")
    team = json.loads((ROOT / "config/team.json").read_text(encoding="utf-8"))
    import platform
    import importlib.metadata as im
    meta = {
        "thesis_title": THESIS, "short_name": "ASL Vision",
        "task": "Word-level (isolated sign) American Sign Language recognition from video",
        "institution": {"university": team["university"], "department": team["department"], "location": team["location"]},
        "team": [p["name"] for p in team["members"]], "supervisor": team["supervisor"]["name"],
        "repository": team.get("github_url"), "model_weights": "https://huggingface.co/srabon12/asl-vision-models (private)",
        "dataset": json.loads((RES / "dataset.json").read_text()),
        "models": {k: {"display_name": s.display_name, "family": s.family, "classes": s.num_classes,
                       "origin": s.origin, "description": s.description} for k, s in MODELS.items()},
        "training_hardware": "Kaggle notebooks, NVIDIA Tesla T4 GPUs (Swin-T: 2x T4, 6.1 h)",
        "app_environment": {"python": platform.python_version(),
                            **{p: im.version(p) for p in ["torch", "torchvision", "mediapipe", "streamlit", "plotly",
                                                          "numpy", "pandas"]}},
        "deployment": "Streamlit Community Cloud (CPU), weights downloaded on demand from Hugging Face",
    }
    (OUT / "metadata").mkdir(exist_ok=True)
    (OUT / "metadata/project_metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def write_index_and_readme():
    with open(OUT / "figure_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(INDEX[0]))
        w.writeheader(); w.writerows(INDEX)


if __name__ == "__main__":
    if OUT.exists():
        import os
        import stat

        def _force(func, path, _exc):          # OneDrive/AV can leave read-only flags
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(OUT, onexc=_force)
    OUT.mkdir(parents=True)
    fig_dataset(); fig_pipeline(); fig_training(); fig_comparison(); fig_scaling(); fig_errors(); fig_deployment()
    copy_originals(); tables(); data_and_meta(); write_index_and_readme()
    print(f"{len(INDEX)} figures indexed ->", OUT)
