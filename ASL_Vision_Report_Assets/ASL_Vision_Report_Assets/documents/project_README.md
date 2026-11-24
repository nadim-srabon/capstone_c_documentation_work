# ASL Vision

**Building a Word-Level American Sign Language Recognition System: A Multi-Architecture Design and Evaluation Study**

A Streamlit web app that presents the capstone research of a team from East West University (Dhaka) and lets visitors test the trained models on their own videos of **isolated ASL signs**. All inference runs on CPU and costs nothing to host.

| Model | Input | Signs | Test top-1 | Test top-5 | Macro F1 |
|---|---|---|---|---|---|
| **Video Swin-T** | RGB video | 1000 | **66.8%** | **88.9%** | 0.665 |
| Geometry Transformer (v10) | MediaPipe landmarks | 1000 | 57.5% | 81.8% | 0.571 |
| ST-GCN | MediaPipe landmarks | 1000 | 48.8% | 74.0% | 0.484 |
| R3D-18 Dual-Stream (v9) | RGB + landmarks | 1000 | 28.2% | 55.1% | 0.271 |
| Geometry Transformer | landmarks | 300 / 500 / 1200 | 63.2% / 61.3% / 58.1% | 85.0% / 84.1% / 81.2% | 0.630 / 0.610 / 0.576 |

The scores come from the official signer-independent ASL Citizen test split (12,560 clips for the 1000-sign models).

---

## Project structure

```
app.py                     Streamlit entry point (navigation, theme)
pages_impl/                overview, recognition (demo), comparison, methodology, about
ui/                        theme/CSS, Plotly charts, results loaders
inference/
  architectures.py         exact model definitions (state_dict-compatible with the training notebooks)
  config.py                model registry, paths, Hugging Face repo setting
  video_processor.py       raw video -> 16 frames -> MediaPipe landmarks + 112x112 signer crop
  model_loader.py          download (if needed), strict load, validation; one model at a time
  predictor.py             per-model tensor layout, softmax, top-k
assets/mediapipe/          MediaPipe .task models used to build the training cache (Apache-2.0)
results/                   metrics, per-class metrics, confusion matrices, training histories, class lists
models/                    weights (git-ignored; downloaded from Hugging Face in the cloud)
scripts/
  prepare_artifacts.py     builds models/ + results/ from the Kaggle outputs
  parity_test.py           compares the app's pipeline with the training cache
  upload_to_hf.py          one-time upload of the weights
training_notebooks/        Kaggle notebooks for the new 1000-class models (+ generator script)
docs/testing_report.md     test results, parity analysis, limitations
.streamlit/config.toml     theme, 50 MB upload limit
requirements.txt           CPU-only Python dependencies
packages.txt               system libraries MediaPipe/OpenCV need on Linux
```

## Run locally

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/prepare_artifacts.py # only if kaggle_outputs/ is present; otherwise set ASL_HF_REPO (below)
streamlit run app.py
```

Then open http://localhost:8501. Python 3.11–3.13 works. No GPU is needed.

## Model storage (free)

- **Why Hugging Face:** the Swin (129 MB) and R3D (141 MB) weights are over GitHub's 100 MB limit, so the weights go in a **Hugging Face model repository**, which is free.
- **One file at a time:** the app downloads only the model a visitor selects, the first time it is used.

```bash
pip install huggingface_hub
huggingface-cli login                                         # WRITE token, huggingface.co/settings/tokens
python scripts/upload_to_hf.py <hf-username>/asl-vision-models  # creates a PRIVATE repo, uploads 7 files
```

**Licence note.** ASL Citizen is distributed under Microsoft Research License Terms: non-commercial research use, and the data may not be redistributed. This repository therefore contains **no dataset videos, cached features, or executed notebooks with dataset frames**.

**Which repo type to use:** the upload script creates a **private** model repo by default. Whether the trained weights may be published openly is for the team and supervisor to decide. If they decide it's fine, rerun the script with `--public`; then no token is needed.

## Deploy to Streamlit Community Cloud (free)

1. **Push to GitHub.** Push this folder to a public GitHub repository. The `.gitignore` already excludes the weights, `kaggle_outputs/`, notebooks with outputs, and the zips. Check with `git status` before the first commit.
2. **Create the app.** Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, then **Create app**:
   - repository and branch: your new repo, `main`
   - main file: `app.py`
   - **Advanced settings:** Python **3.12**
3. **Add secrets.** In **Secrets**, add:
   ```toml
   ASL_HF_REPO = "<hf-username>/asl-vision-models"
   HF_TOKEN = "hf_..."        # a READ token; only needed for a private repo
   ```
4. **Deploy.** The first build installs CPU PyTorch and MediaPipe, which takes several minutes. `packages.txt` installs `libgl1`, `libglib2.0-0` and `libportaudio2`.
5. **Smoke test.** Open the app, upload a short clip on **Live Recognition**, and try each model once.

Free apps go to sleep after a period without visitors. Open the app a few minutes before a presentation and run one prediction per large model, so their weights are already downloaded.

Memory: peak process memory measured locally is about 830 MB, with only one model in memory at a time (`docs/testing_report.md` §4).

## How the demo reproduces training preprocessing

`inference/video_processor.py` is a direct port of the team's cache builder:
- **Frames:** decode the video with OpenCV (BGR → RGB) and take 16 evenly spaced frames.
- **Landmarks:** run MediaPipe Tasks 0.10.35 in VIDEO mode (2 hands, pose-lite, face). Assign the hands by screen position, normalise to shoulder centre and shoulder width → 58 points × (x, y).
- **Signer crop:** cut out the pose bounding box with a 25% margin and resize to 112×112.
- **Per model:**
  - **Swin:** /255 → resize to 224 → Kinetics mean/std
  - **R3D:** /255 → Kinetics mean/std
  - **Landmark models:** the raw 16×116 vector

The R3D-18 re-evaluation with this code reproduced its recorded test top-1 (0.282). The parity test on 30 held-out test videos is in `docs/testing_report.md`.

## Known limitations

- **Isolated signs only.** The app classifies one sign per clip; it does not translate continuous signing in real time.
- **Closed vocabulary.** Signs outside the vocabulary are mapped to the closest known sign.
- **Domain shift.** The models were trained on ASL Citizen webcam recordings. Other camera setups reduce accuracy.
- **MediaPipe tracking state.** Because of it, the app's features match the training cache closely but not bit-for-bit (`docs/testing_report.md` §3).
- **Cold starts.** Free hosting sleeps when idle; the first Swin/R3D prediction after a cold start also downloads about 130–140 MB.

## Team

Nadim Srabon, Pranay Saha, Redoanul Alam Mozumder, Aniruddha Manab Deb.
Supervisor: Dr. Anup Kumar Paul.
Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh.
