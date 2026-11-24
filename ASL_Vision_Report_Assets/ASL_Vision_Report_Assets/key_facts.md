# Key facts and numbers

## Project
- Thesis: *Building a Word-Level American Sign Language Recognition System: A Multi-Architecture Design and Evaluation Study*
- Task: word-level (isolated sign) recognition -- one sign per video clip, 16 sampled frames per clip.
- Dataset: ASL Citizen (Microsoft Research); 63,550 videos / 2,731 glosses preprocessed.
- 1,000-sign subset: 30,909 clips (14,571 train / 3,778 val / 12,560 test), 30-32 clips per sign, signer-independent splits.

## Results, 1,000 signs (test, 12,560 clips)
| Model | Top-1 | Top-5 | Macro F1 | Params | CPU latency |
|---|---|---|---|---|---|
| Video Swin-T | 66.8% | 88.9% | 0.665 | 28.6 M | 1220 ms |
| Geometry Transformer | 57.5% | 81.8% | 0.571 | 1.91 M | 3 ms |
| ST-GCN | 48.8% | 74.0% | 0.484 | 1.37 M | 10 ms |
| R3D-18 Dual-Stream (v9) | 28.2% | 55.1% | 0.271 | 35.3 M | 457 ms |

## Vocabulary scaling (Geometry Transformer, test top-1 / top-5)
- 300 signs: 63.2% / 85.0%
- 500 signs: 61.3% / 84.1%
- 1,000 signs: 57.5% / 81.8%
- 1,200 signs: 58.1% / 81.2%

## Deployment
- Free hosting: Streamlit Community Cloud (CPU); weights on Hugging Face, downloaded on demand; one model in memory at a time.
- Peak process memory (local): about 827 MB with Video Swin-T loaded.
- MediaPipe feature extraction per clip (local CPU): 1.1 s; model inference: Swin 2,551 ms, Geometry Transformer 3 ms.
- Parity test (30 raw test videos): the app's predictions match predictions on the training-cache features for
  83% (Swin) / 87% (Geometry Transformer) of clips;
  repeated runs give identical predictions (100%). Exact equality is impossible because MediaPipe's tracking state carried over between videos when the cache was built.
- R3D-18 re-evaluation with the app's model code reproduced the recorded test top-1 exactly (0.282).
