# ASL Vision — Testing Report

**Date:** 2026-11-21
**Environment:** Windows 11, CPU only (no CUDA), Python 3.13.5, torch 2.15.0+cpu, torchvision 0.29.2, mediapipe 0.10.35, streamlit 1.67.0.

Each result was measured as described; nothing is estimated. The raw outputs are in `docs/parity_results_*.json` and `docs/memory_latency_local.json`.

## Summary

| # | Test | Result |
|---|---|---|
| T1 | All 7 checkpoints load | ✅ All load with every key matched (strict check), on CPU |
| T2 | Class mappings | ✅ All 1000-class models share fingerprint `72d3ef78291c`. The 300/500/1200 lists were rebuilt and cross-checked against each run's per-class report |
| T3 | Preprocessing parity with the training cache | ⚠️ Equivalent but not bit-exact (see §3) |
| T4 | Tensor dimensions | ✅ R3D `(1,16,1,3,112,112)`, Swin `(1,3,16,224,224)`, landmarks `(1,16,116)` |
| T5 | Output dimensions | ✅ `(1, num_classes)`, asserted at load time and on every prediction |
| T6 | Confidence | ✅ Softmax over all classes; top-k sorted |
| T7 | Model switching | ✅ `st.cache_resource(max_entries=1)`: only one model in memory at a time |
| T8 | Invalid videos | ✅ Corrupted file → friendly message. Too long (>20 s or >1200 frames) and no-frame cases are caught |
| T9 | CPU only | ✅ No CUDA anywhere; CPU wheels |
| T10 | Memory | ✅ Peak process RSS ≈ 830 MB (Swin), see §4 |
| T11 | Inference time | ✅ Measured, see §4 |
| T12 | Local Streamlit run | ✅ All 5 pages render with 0 exceptions (AppTest). The recognition flow passed end to end with real videos. The server health check passed |
| T13 | Cloud run | ⬜ Pending deployment |

## 1. Re-evaluation consistency

- **R3D-18:** re-evaluating the R3D-18 dual-stream checkpoint on Kaggle with the app's model code gave test top-1 **0.2815**, matching the original notebook's recorded **0.282**. This confirms the architecture rebuild and its RGB preprocessing.
- **Other models:** all other metrics shown in the app are copied from the Kaggle evaluation outputs.

## 2. Checkpoint loading

| Model | Load (s) | Inference, 2 threads (ms) | Same outputs as plain `load_state_dict` |
|---|---|---|---|
| Video Swin-T | 3.7 | 2551 | ✅ bit-identical |
| R3D-18 Dual-Stream | 0.1 | 710 | ✅ |
| Geometry Transformer (300/500/1000/1200) | 0.03 | 2–4 | ✅ |
| ST-GCN | 0.05 | 11 | ✅ |

The loader memory-maps the weights and builds the model without first creating random weights, so weights are never held twice in RAM. Outputs are bit-identical to conventional loading.

## 3. Parity: app pipeline vs the training cache (30 held-out test videos)

**Method:** the Kaggle extras notebook exported 30 raw test videos (distinct signs) together with the `.npz` features that the cache builder computed from them. Each video went through the app's `FeatureExtractor`, and the results were compared with the cache. Then every 1000-class model predicted from both versions of the features.

| Measure | Fresh MediaPipe per video (**app default**) | Shared MediaPipe (like the cache builder) |
|---|---|---|
| Landmark mean \|diff\| (shoulder-width units) | 0.112 | 0.124 |
| Signer-crop mean \|pixel diff\| (0–255) | 15.4 | 16.5 |
| Hand-presence agreement per frame | 100% | 100% |
| Same top-1 as cache: Swin / GeomTr / ST-GCN / R3D | 83% / 87% / 73% / 73% | 90% / 83% / 70% / 50% |
| Correct out of 30, cache → app: Swin | 19 → 17 | 19 → 17 |
| Correct, GeomTr | 19 → 19 | 19 → 21 |
| Correct, ST-GCN | 13 → 13 | 13 → 13 |
| Correct, R3D | 10 → 9 | 10 → 6 |
| **Same video twice → same top-1** | **100% (all models)** | 67–87% |
| Feature extraction time per video | 2.7 s | 1.4 s |

**Why it is not bit-exact.** The diagnosis below rules out frame selection and decoding:
- **Same frames:** frame counts from `CAP_PROP_FRAME_COUNT`, `grab()` and `read()` agree on all 30 videos, and a 3-frame clip matches its cache entry.
- **Same detections:** hand detections agree on 100% of frames.
- **The cause is tracking state.** MediaPipe Tasks in VIDEO mode tracks people from one frame to the next. The cache builder reused one landmarker for ~21,000 videos, so each cached clip depends on the clip processed before it.
- **Measured directly:** processing the same video after a different one changes the landmarks by 0.08–0.12, the same size as the cache-vs-app gap. The cache therefore contains this order-dependent noise, and it cannot be reproduced exactly.

**Decision.** The app starts fresh landmarkers for every video:
- **Deterministic:** identical uploads give identical predictions.
- **Faithful:** parity with the cache and accuracy are equivalent to the shared mode.

**Implication.** Demo predictions can differ slightly from the predictions the same model would make on the cached test features. Accuracy on these 30 clips is about the same either way. With only 30 clips, the cache-vs-app accuracy differences are within noise.

## 4. Memory and latency (local CPU, `torch.set_num_threads(2)`)

| Stage | Process RSS |
|---|---|
| Python + torch imported | 219 MB |
| + MediaPipe landmarkers and one video's features | 453 MB |
| + Swin-T loaded and run (largest) | ~810–830 MB peak |
| + R3D-18 dual-stream loaded and run | ~755–840 MB |
| + landmark models | ~720–750 MB |

- **Feature extraction:** about 1.1 s with shared landmarkers, or about 2.7 s with fresh landmarkers (the default). Most of the difference is creating the landmarkers.
- **End-to-end demo time on this machine:**
  - landmark models: about 3 s per clip
  - R3D-18: about 3.5 s per clip
  - Swin-T: about 5–6 s per clip (plus a 4 s one-time model load)
- **Comparing all four models:** about 26 s on a cold start.

## 5. Recognition page end-to-end (AppTest with real uploads)

| Case | Result |
|---|---|
| DONUT clip, Video Swin-T + "compare all 4 models" | Predicted **DONUT** ✅; comparison table rendered; 0 exceptions |
| AWAKE clip, Geometry Transformer 300 | Predicted **AWAKE** ✅; a detection-quality warning was shown, as designed |
| AWAKE clip, R3D-18 dual-stream | Predicted SPONTANEOUS ✗ (consistent with its 28% test accuracy); no error |
| Corrupted `.mp4` (text bytes) | Message: "The file could not be opened as a video…"; no crash |

## 6. Known limitations

1. **Bit-exact preprocessing is impossible** because of MediaPipe's tracking state in the training cache (see §3).
2. **Domain shift:** ASL Citizen clips are webcam recordings of a single signer. Phone videos, side angles, poor lighting or several people will reduce accuracy.
3. **Closed vocabulary:** an unknown sign is still mapped to the nearest known sign, often with moderate confidence.
4. **Swin3D-T may be under-trained:** its best epoch was its last (25/25).
5. **Different selection rule for R3D-18:** the R3D-18 checkpoint was selected on validation loss, while all other models were selected on validation top-1.
6. **Cold starts on free hosting:** the first prediction after the app wakes up includes downloading the model weights from Hugging Face. Expect about 10–30 s for Swin and R3D.
7. **Isolated signs only:** the system classifies one sign per clip and does not translate continuous signing.
