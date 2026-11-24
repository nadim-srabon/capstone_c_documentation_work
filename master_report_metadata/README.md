# ASL Vision -- Report Assets

**Thesis:** Building a Word-Level American Sign Language Recognition System: A Multi-Architecture Design and
Evaluation Study

Everything here is generated from the project's own evaluation outputs (Kaggle runs) and local measurements.
No number was estimated. `figure_index.csv` lists **every image** with a ready-to-use title, caption, source
file and whether it shows people (96 images in total).

## Folder guide

| Folder | What it contains |
|---|---|
| `A_generated_figures/` | Report-quality figures in one consistent style: **PNG (300 dpi) + PDF (vector)**. Use these first. |
| `A_.../01_dataset` | Split sizes, test clips per vocabulary size |
| `A_.../02_methodology` | Recognition pipeline diagram, architecture overview diagram |
| `A_.../03_training_curves` | Loss/accuracy curves for all 7 models + validation comparison of the four 1,000-sign models |
| `A_.../04_model_comparison_1000` | Top-1/top-5, macro P/R/F1, accuracy vs CPU latency, per-class F1 distribution |
| `A_.../05_vocabulary_scaling` | Geometry Transformer at 300/500/1,000/1,200 signs |
| `A_.../06_error_analysis` | Per model: most-confused pairs, confusion sub-matrix, easiest/hardest signs |
| `A_.../07_deployment_testing` | Preprocessing parity test, CPU inference time, memory footprint |
| `B_original_notebook_figures/` | The figures produced inside the training notebooks, renamed clearly (duplicates removed) |
| `C_app_screenshots/` | Screenshots of every page of the web application (signer blurred) |
| `D_RESTRICTED_contains_dataset_signers/` | 12 figures showing **real ASL Citizen participants** -- read the licence note below |
| `tables/` | Result tables as **CSV, Markdown and LaTeX** (`table00`...`table06`) |
| `data/` | Raw evaluation data: metrics.json, per-class metrics, confusion matrices (`.npz`, key `cm`), training histories, class lists, deployment measurements |
| `metadata/project_metadata.json` | Title, team, supervisor, dataset facts, model registry, software versions |
| `documents/` | Testing report and project README |
| `references.bib` | BibTeX for the dataset, architectures and tools used |
| `key_facts.md` | All headline numbers in one place, for writing the text |
| `scripts/` | The scripts that regenerated this folder (reproducibility) |

## Suggested placement in the report

| Chapter | Figures / tables |
|---|---|
| Dataset | fig01_01, fig01_02, table04, B/r3d18_v9__dataset_* |
| Methodology | fig02_01 (pipeline), fig02_02 (architectures), B/r3d18_v9__mediapipe_hand_landmark_indices_schematic, table03 |
| Training | fig03_* (one per model), fig03_08 (all four on one plot) |
| Results | table01, fig04_01 - fig04_04 |
| Vocabulary scaling | table02, fig05_01, fig05_02 |
| Error analysis | fig06_* (a = confused pairs, b = confusion sub-matrix, c = best/worst signs) |
| Implementation / deployment | fig07_*, table05, table06, C_app_screenshots |
| Appendix | B_original_notebook_figures, data/ |

## Important notes before using a figure

1. **Licence (ASL Citizen, Microsoft Research License Terms).** The dataset may not be redistributed, and its
   videos show identifiable people. Folder `D_RESTRICTED...` contains raw dataset frames. Check with your
   supervisor before putting any of them in a document that will be published or shared outside the team;
   if you use one, blur the faces. All other folders contain no dataset frames (app screenshots are blurred).
2. **Do not cite the v9 bars in `B/.../geomtransformer_v10_500cls__precision_recall_f1_vs_v9.png` or the
   1200cls version.** The notebook reused the 300-class v9 values for every vocabulary size. The 300-class
   version is correct. Correct v9 (R3D-18 dual-stream) 1,000-class P/R/F1 are in `table01`.
3. **Checkpoint selection differs:** R3D-18 dual-stream (Capstone B) was selected on lowest validation loss;
   all other models on highest validation top-1.
4. **Video Swin-T** reached its best validation accuracy in its last epoch (25/25) -- longer training might
   improve it (a limitation worth stating).
5. CPU latency in `fig04_03` / `table01` was measured on a Kaggle CPU (2 threads, model forward pass only);
   `fig07_02` / `table06` were measured on the development laptop. Do not mix them in one comparison.
6. The 1,000-sign Geometry Transformer (57.5 %) scores slightly below the 1,200-sign one (58.1 %): they come
   from different training runs (the 1,000 run used the shared 1,000-class notebook). Mention this if you
   discuss the scaling curve.
