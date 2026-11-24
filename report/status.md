# Report Status Tracker

**Thesis:** *Building a Word-Level American Sign Language Recognition System: A Multi-Architecture Design and Evaluation Study*
**Team:** Nadim Srabon, Pranay Saha, Redoanul Alam Mozumder, Aniruddha Manab Deb. **Supervisor:** Dr. Anup Kumar Paul (EWU CSE)
**Last updated:** 2026-11-24 (Chapter 1 drafted, full LaTeX scaffold built)

> Update this file every time you finish or change something. Keep the newest entry at the top of the *Change log* (§9).

---

## 1. Overall progress

| Part | File | Status | Pages now | Target pages* |
|---|---|---|---|---|
| Title page | `frontmatter/titlepage.tex` | ✅ Done (update `\SubmissionDate` in `macros.tex` before printing) | 1 | 1 |
| Declaration | `frontmatter/declaration.tex` | ✅ Done (needs signatures) | 1 | 1 |
| Letter of Approval | `frontmatter/approval.tex` | 🟡 Chairperson's name missing | 1 | 1 |
| Acknowledgement | `frontmatter/acknowledgement.tex` | ⬜ Placeholder (write last) | 1 | 1 |
| Abstract | `frontmatter/abstract.tex` | ⬜ Placeholder (write last) | 1 | 1 |
| TOC / LoF / LoT | automatic | ✅ Working | 4 | 4–5 |
| List of Abbreviations | `frontmatter/abbreviations.tex` | 🟡 Started (extend with each chapter) | 1 | 1–2 |
| **Ch 1 Introduction** | `chapters/ch1_introduction.tex` | ✅ **First full draft** | 8 | 6–8 |
| Ch 2 Related Works | `chapters/ch2_related_works.tex` | ⬜ Skeleton + per-section plan | 1 | 20–23 |
| Ch 3 Materials and Method | `chapters/ch3_materials_and_method.tex` | ⬜ Skeleton + per-section plan | 2 | 22–25 |
| Ch 4 Data Description | `chapters/ch4_data_description.tex` | ⬜ Skeleton | 1 | 3–4 |
| Ch 5 Results and Discussion | `chapters/ch5_results_and_discussion.tex` | ⬜ Skeleton | 1 | 12–15 |
| Ch 6 Conclusion | `chapters/ch6_conclusion.tex` | ⬜ Skeleton | 1 | 2–3 |
| Bibliography | `references.bib` | 🟡 39 entries; 5 need verification (§6) | 2 | 3–4 |
| Appendix A | `backmatter/appendix_a.tex` | ⬜ Skeleton | 1 | 5–8 |
| CO/PO mapping (400-A/B/C) | `backmatter/co_po_mapping.tex` | ⬜ Table frames only; needs official CO/PO text | 1 | 6–10 |
| **Total** | | | **29** | **≈ 90–100** |

\*Page targets come from the department's sample TOC (Ch 1 ≈ p.12–17, Ch 2 ≈ 18–40, Ch 3 ≈ 41–63, Ch 4 ≈ 64–65, Ch 5 ≈ 66–75, Ch 6 ≈ 76–77, Bibliography ≈ 78–80, appendix + CO/PO up to ≈ 94).

**Next task:** Chapter 2 (Related Works).

---

## 2. The storyline (every chapter must stay consistent with this)

1. **Capstone A (survey).** We reviewed 25 papers and set three research questions (RQ1 accuracy vs cost vs real time; RQ2 landmarks vs pixels for generalisation; RQ3 Transformer vs RNN, with an accuracy-gated word→sentence path). We **planned** a dual-stream design: a CNN on hand crops plus normalised landmarks, per-frame fusion, a 2–4-layer Transformer, and an accuracy gate of val top-1 ≥ 70 % to unlock a CTC sentence decoder. The plan targeted WLASL, a FastAPI/WebSocket webcam deployment and a 250 ms latency budget. It also planned an **ablation with the landmark stream alone as the "accuracy floor"**.
2. **Capstone B (training).** There were nine development iterations (internally v1–v9, but **never write version numbers in the report**; describe each step by what changed). Problems along the way: MediaPipe/protobuf conflicts (fixed by moving to the Tasks API), a fork-unsafe DataLoader (fixed by an offline cache), a GradScaler bug, severe overfitting on WLASL (median 6 videos/class), and a 38.1 % hand-miss rate (reduced to 30.7 %). Unreliable handedness was fixed by x-position assignment. The R3D-18 swap made results worse (a negative result). The hand crops were replaced by a signer bounding box. We moved to ASL Citizen and ran a controlled scaling study (300/500/1000/1200 signs, median ≈ 31 videos/class). **Results:** test top-1 31.0 / 27.8 / 28.2 / 28.5 %. TTA hurt at every K (by up to −3.1 points). The 0.70 gate was never reached (best val 0.423).
3. **The redesign after B, the "important changes" (reported in this final report).**
   - **Geometry Transformer (geometry-only model):** the landmark stream alone, same 116-d input (58 points × (x, y)) plus velocity, D_MODEL 256, 4 heads, 3 layers. Test top-1 **63.2 / 61.3 / 57.5 / 58.1 %** at 300 / 500 / 1000 / 1200 signs. This is about **double** the dual-stream result, so **the 3D-CNN appearance stream was the bottleneck**.
   - **ST-GCN** (adaptive graph over the 58 landmarks): **48.8 %** at 1000 signs.
   - **Video Swin-T** (Kinetics-400, end-to-end, 224×224): **66.8 % top-1 / 88.9 % top-5** at 1000 signs. It is the best model.
   - All 1000-sign models use the **same class list** (fingerprint `72d3ef78291c`) and the **same signer-independent split**.
   - Validation top-1 now **exceeds the 0.70 gate from Capstone A** (Swin 74.6 %, Geometry Transformer 74.0 % at 300 signs and 71.6 % at 500 signs). The sentence-level path was **not** built: continuous signing is out of scope (Ch 6 future work).
4. **Capstone C (product).** A Streamlit web app with 5 pages (Home, Recognize, Results, Methodology, About), deployed on Streamlit Community Cloud (CPU, free). Weights live in a private Hugging Face repo and are downloaded on demand. One model is kept in memory, and the 7 models are selectable. Preprocessing is a direct port of the cache builder. It has been tested: parity, latency, memory, invalid inputs, determinism.
   - Live app: https://aslr-system.streamlit.app/ · Repo: https://github.com/nadim-srabon/asl-vision
   - **Planned in A but changed in C:** FastAPI + live webcam + TTS became upload-based Streamlit, because it is free CPU hosting and the models are word-level. Explain this honestly in Ch 3.3 and Ch 6.2.

---

## 3. Key facts registry (copy numbers from here or from `master_report_metadata/`)

Source of truth: `master_report_metadata/key_facts.md`, `tables/table00_all_models_all_metrics.csv`, `documents/testing_report.md`.

| Fact | Value |
|---|---|
| ASL Citizen (full) | 83,399 videos, 2,731 signs, 52 signers |
| Our feature cache | 63,550 videos, 2,731 glosses, 3 Kaggle shards |
| 1,000-sign subset | 30,909 clips: 14,571 train / 3,778 val / 12,560 test; 30–32 clips per sign |
| Clip input | 16 frames; landmarks 16 × 116; signer crop 112×112 (Swin resized to 224) |
| Typical source video | 640×480, ~30 fps, median 73 frames |
| Swin-T (1000) | val 74.6, test top-1 66.8 / top-5 88.9, macro F1 0.665, 28.6 M params, 1,220 ms Kaggle CPU, best epoch 25/25, 2× T4, 6.1 h |
| Geometry Transformer (1000) | val 67.5, test 57.5 / 81.8, F1 0.571, 1.91 M, 3.2 ms |
| ST-GCN (1000) | val 58.4, test 48.8 / 74.0, F1 0.484, 1.37 M, 10.1 ms |
| R3D-18 Dual-Stream (1000, final Capstone B model) | val 36.2, test 28.2 / 55.1, F1 0.271, 35.3 M, 457 ms |
| Geometry Transformer scaling | 300: 63.2 / 85.0; 500: 61.3 / 84.1; 1000: 57.5 / 81.8; 1200: 58.1 / 81.2 |
| Capstone B scaling (plain / TTA top-1) | 300: 31.0 / 29.7; 500: 27.8 / 27.2; 1000: 28.2 / 27.1; 1200: 28.5 / 25.4 |
| App memory | peak ≈ 810–830 MB (Swin loaded) |
| App local timing | features ≈ 2.7 s (fresh landmarkers, default); Swin inference 2,551 ms; Geometry Transformer 3 ms; end-to-end Swin ≈ 5–6 s |
| Parity (30 test videos) | top-1 agreement with cache: Swin 83 %, GeomTr 87 %, ST-GCN 73 %, R3D 73 %; determinism 100 % |
| R3D re-evaluation with app code | 0.2815 (matches the recorded 0.282) |

**Caveats that must be stated when the numbers are used:**

- CPU latency in table01 (Kaggle, 2 threads) and table06 (laptop) are different machines. Never mix them in one comparison.
- R3D-18 checkpoint selection used the lowest val loss; all the other models used the highest val top-1.
- The 1000-sign Geometry Transformer (57.5 %) is below the 1200-sign one (58.1 %) because they are different runs. Mention this when discussing the scaling curve.
- Swin's best epoch was its last one (25/25), so longer training may help (a limitation).
- **Do not use** `B/.../geomtransformer_v10_500cls__precision_recall_f1_vs_v9.png` or the 1200cls version (they show wrong v9 bars).

---

## 4. Figure / table / label registry (avoid duplicates)

### Used so far

| Label | Type | File / content | Chapter |
|---|---|---|---|
| `fig:project-phases` | Figure 1.1 | TikZ: Capstone A → B → Redesign → C | 1 |
| `fig:app-preview` | Figure 1.2 | `figures/ch1/app06_recognize_result.png` | 1 |
| `tab:challenges-responses` | Table 1.1 | Challenges C1–C5 of word-level SLR and how the project addresses them | 1 |
| `tab:headline` | Table 1.2 | 1000-sign headline results | 1 |
| `eq:task` | Eq. 1.1 | classification task definition | 1 |

### Reserved labels (already referenced or planned; keep these names)

`ch:introduction, ch:related, ch:method, ch:data, ch:results, ch:conclusion, app:a, app:copo, tab:lit-summary, tab:outcome-400a, tab:outcome-400b, tab:outcome-400c`, and all `sec:` labels in the skeleton files.

### Planned placement of the remaining assets

| Chapter | Assets (from `master_report_metadata/`) |
|---|---|
| 3 | `fig02_01_recognition_pipeline` (copied to `figures/ch1/`, not yet used; use it in 3.2.6), `fig02_02_architecture_overview`, B hand-landmark schematic, table03 (hyperparameters), app screenshots app01–app12 (3.3/3.4) |
| 4 | `fig01_01`, `fig01_02`, table04, B report EDA figures (frame-rate/resolution) |
| 5 | table00/01/02/05/06, `fig03_*` training curves, `fig04_*` comparison, `fig05_*` scaling, `fig06_*` error analysis, `fig07_*` deployment tests |
| Appendix A | `B_original_notebook_figures/*`, extra curves, code listings |

---

## 5. Notes for writing the remaining chapters

- **Ch 2:** rewrite the 25 summaries from Capstone A in the report's own voice (A's text is weak in places and repeats itself; B's Ch 2 is better and can be adapted). Keep the three groups (A)/(B)/(C) as the department TOC requires. End with the **summary longtable** (`tab:lit-summary`: Ref | Method | Dataset | Result | Limitation) and the research gap. Add Video Swin and ST-GCN as background, because they are now our models.
- **Ch 3:** this is where the **Capstone B development history** goes (described by what changed and why, with no version numbers), condensed, in 3.2.2 *Proposed Models*, followed by the **post-B redesign**. Include the equations from B (landmark normalisation, velocity, attention, label smoothing). Section 3.2.12 *Computation of Parameters* should derive the parameter counts (e.g. Geometry Transformer ≈ 1.91 M). Sections 3.3/3.4 cover the app.
- **Ch 4:** a short chapter. Cover ASL Citizen, the licence, the split policy, B's Table 6.1 (per-K split sizes) and the EDA figures.
- **Ch 5:** combine the B results (scaling, TTA, overfitting, B's confusion analysis) with the new models. Compare at 1000 signs, cover scaling and the accuracy–latency trade-off, and include the app tests. Answer RQ1–RQ3 explicitly in 5.4 *Discussion*.
- **Ch 6:** contributions; limitations (single seed, isolated signs only, closed vocabulary, domain shift, Swin under-trained, no bit-exact parity, cold starts, licence); future work (continuous recognition and the sentence path, live webcam, multi-seed, a Swin+landmark ensemble, self-supervised pre-training, Bangla SL).

---

## 6. Open questions / items to verify

| # | Item | Where | Owner |
|---|---|---|---|
| V1 | Year of the Capstone A and B reports (currently 2026) | `references.bib` (`srabon2026capstoneA/B`) | team |
| V2 | Author lists for 3 papers (`jimaging2024resnetlstm`, `vivit2025breaking`, `mediapipelstm2024continuous`) | `references.bib` | team |
| V3 | Chairperson's name for the approval page | `frontmatter/approval.tex` | team |
| V4 | Official CO/PO statements for CSE400 A/B/C | `backmatter/co_po_mapping.tex` | team / department |
| V5 | EWU logo file (`figures/common/ewu_logo.png`); the title page includes it automatically if the file exists | title page | team |
| V6 | Submission month (`\SubmissionDate` in `setup/macros.tex`, currently "December 2026") | macros | team |
| V7 | Confirm the cloud deployment test (testing report T13 was "pending" on 2026-11-21) before claiming cloud test results in Ch 5 | Ch 5 | team |
| V8 | Supervisor approval needed before using any `D_RESTRICTED` image | Ch 5 / App. A | supervisor |

---

## 7. Build environment notes

- This machine uses MiKTeX 25.12. The default MiKTeX package source was an incomplete local repository, so on 2026-11-24 it was switched to a remote CTAN mirror (`mpm --set-repository`) and *AutoInstall* was enabled. These packages were installed: xpatch, setspace, siunitx, multirow, makecell, tocloft, cite, cleveref, ieeetran. The font maps were refreshed with `initexmf --mkmaps`.
- MiKTeX's `latexmk` needs Perl, so build with `pdflatex → bibtex → pdflatex → pdflatex`.
- The report also builds on Overleaf (pdfLaTeX) without changes.

---

## 8. Chapter 1 summary (what it already claims, so later chapters must back it up)

- 1.1 Background: WFD ≈ 70 M Deaf people and 300+ sign languages; WHO 430 M with disabling hearing loss; isolated vs continuous SLR; R3D / Video Swin / Kinetics / MediaPipe / Transformer / ST-GCN; WLASL vs ASL Citizen; the A→B→Redesign→C story (Figure 1.1); app screenshot (Figure 1.2).
- 1.2 Problem statement: the **real-world problem** (no practical, accurate, signer-independent, large-vocabulary ASL recogniser that runs on ordinary hardware and reaches users), plus the formal task (Eq. 1.1). Analysis: inherent challenges C1–C5 of the task (similar-looking signs in large vocabularies, signer variation, few examples per sign, spatio-temporal information, accuracy vs deployment cost) in Table 1.1.
  - **Rule:** the problems *we* ran into during development (Capstone B overfitting, test-time augmentation hurting, the weak 3D-CNN stream, no cost measurement, no product) are **not** part of the problem statement. They belong in Ch 3.2.2 (development path) and Ch 5 (results/discussion).
- 1.3 Objectives O1–O5.
- 1.4 Contributions 1–5 (Table 1.2 headline results). **Claims that later chapters must back up with evidence:** the stream ablation doubles accuracy (Ch 3.2.2 + Ch 5); 15× fewer params and ≈ 400× faster (Ch 5.3); val > 70 % gate (Ch 5); parity 83–87 % and determinism (Ch 5.3); peak ≈ 830 MB (Ch 5.3).
- 1.5 Outline of Chapters 2–6 and the back matter.

---

## 9. Change log

| Date | Who | Change |
|---|---|---|
| 2026-11-24 | Claude (for Redoanul) | Rewrote Section 1.2: the problem statement now describes the real-world problem the system solves (not our project difficulties); the analysis lists the inherent challenges C1–C5. Table 1.1 was replaced (`tab:problems-responses` → `tab:challenges-responses`). Build: 30 pages, 0 errors, 0 warnings. |
| 2026-11-24 | Claude (for Redoanul) | Switched the font to the LaTeX default (Computer Modern via `lmodern`); removed `newtx`; `amssymb` loaded again. Removed all "notebook version" wording from Ch 1 and the skeleton notes; added the "no internal development labels" rule (formatting instructions §12). Table 1.2: the Phase column was dropped (the information moved to the caption) to fit the wider font. Build: 29 pages, 0 errors, 0 warnings. |
| 2026-11-24 | Claude (for Redoanul) | Created the LaTeX project, preamble, macros, front matter, bibliography (39 entries), skeletons for Ch 2–6 and back matter. Wrote the full first draft of Chapter 1. Build: 29 pages, 0 errors, 0 warnings. Created `formatting_instructions.md` and this file. |
