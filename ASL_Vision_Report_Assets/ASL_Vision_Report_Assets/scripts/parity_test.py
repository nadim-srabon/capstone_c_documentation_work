"""Parity test: does the app's raw-video pipeline reproduce the training cache?

For each raw ASL Citizen test video in the parity pack (kaggle_outputs/extras/parity_samples, private,
never committed) we run the app's FeatureExtractor and compare against the .npz the cache builder
produced for the same video:
  * landmarks: mean |diff|, and whether hand / pose presence per frame agrees
  * appearance: mean |pixel diff| of the 112x112 signer crops
  * predictions of every 1000-class model: on cached features vs on app-extracted features

    .venv/Scripts/python scripts/parity_test.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inference.config import COMPARISON_1000  # noqa: E402
from inference.model_loader import load_model  # noqa: E402
from inference.predictor import predict  # noqa: E402
from inference.video_processor import ClipFeatures, FeatureExtractor, VideoInfo  # noqa: E402

PACK = ROOT / "kaggle_outputs/extras/parity_samples"


def hand_mask(lm):
    pts = lm.reshape(16, 58, 2)
    # a missing hand is all-zero BEFORE normalisation -> after normalisation every point of it is identical
    return np.stack([np.ptp(pts[:, s:s + 21], axis=(1, 2)) > 1e-6 for s in (0, 21)], 1)


def main():
    rows = list(csv.DictReader(open(PACK / "samples.csv", encoding="utf-8")))
    mode = sys.argv[1] if len(sys.argv) > 1 else "fresh"          # fresh | shared
    extractor = FeatureExtractor(reset_each_video=(mode == "fresh"))
    models = {}
    per_video, t_extract = [], []
    for r in rows:
        cached = np.load(PACK / f"{r['video_id']}.npz")
        c_lm, c_app = cached["landmarks"], cached["appearance"]
        t0 = time.perf_counter()
        feats = extractor.extract(PACK / r["video_file"])
        t_extract.append(time.perf_counter() - t0)
        feats_again = extractor.extract(PACK / r["video_file"])     # repeatability: same upload twice
        cached_feats = ClipFeatures(landmarks=c_lm, appearance=c_app, info=VideoInfo(0, 0, 0, 0, 0))
        rec = {"gloss": r["gloss"], "frames": feats.info.total_frames,
               "lm_mad": float(np.abs(feats.landmarks - c_lm).mean()),
               "lm_exact_frames": int((np.abs(feats.landmarks - c_lm).max(1) < 1e-5).sum()),
               "hand_presence_agree": float((hand_mask(feats.landmarks) == hand_mask(c_lm)).mean()),
               "app_mad": float(np.abs(feats.appearance.astype(np.int16) - c_app.astype(np.int16)).mean())}
        for mid in COMPARISON_1000:
            if mid not in models:
                models[mid] = load_model(mid)
            p_cache = predict(models[mid], mid, cached_feats)
            p_app = predict(models[mid], mid, feats)
            rec[f"{mid}_same_top1"] = p_cache.label == p_app.label
            rec[f"{mid}_repeatable"] = predict(models[mid], mid, feats_again).label == p_app.label
            rec[f"{mid}_cache_correct"] = p_cache.label == r["gloss"]
            rec[f"{mid}_app_correct"] = p_app.label == r["gloss"]
            rec[f"{mid}_prob_diff"] = abs(p_cache.confidence - dict(zip(p_app.top_labels, p_app.top_probs)).get(p_cache.label, 0.0))
        per_video.append(rec)
        print(f"{r['gloss']:14s} lm_mad={rec['lm_mad']:.4f} exact_frames={rec['lm_exact_frames']:2d}/16 "
              f"app_mad={rec['app_mad']:5.2f} | same top-1: " +
              " ".join(f"{m.split('_')[0]}={'Y' if rec[f'{m}_same_top1'] else 'n'}" for m in COMPARISON_1000))
    extractor.close()

    n = len(per_video)
    summary = {"mode": mode, "n_videos": n,
               "landmarks_mean_abs_diff": float(np.mean([r["lm_mad"] for r in per_video])),
               "landmark_frames_bit_exact_pct": float(np.mean([r["lm_exact_frames"] / 16 for r in per_video])),
               "hand_presence_agreement_pct": float(np.mean([r["hand_presence_agree"] for r in per_video])),
               "appearance_mean_abs_pixel_diff": float(np.mean([r["app_mad"] for r in per_video])),
               "feature_extraction_s_mean": float(np.mean(t_extract)), "models": {}}
    for mid in COMPARISON_1000:
        summary["models"][mid] = {
            "top1_agreement_cache_vs_app": sum(r[f"{mid}_same_top1"] for r in per_video) / n,
            "repeatable_top1_pct": sum(r[f"{mid}_repeatable"] for r in per_video) / n,
            "correct_on_cached_features": sum(r[f"{mid}_cache_correct"] for r in per_video),
            "correct_on_app_features": sum(r[f"{mid}_app_correct"] for r in per_video),
            "mean_abs_prob_diff_of_cached_top1": float(np.mean([r[f"{mid}_prob_diff"] for r in per_video]))}
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    with open(out / f"parity_results_{mode}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("\n" + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
