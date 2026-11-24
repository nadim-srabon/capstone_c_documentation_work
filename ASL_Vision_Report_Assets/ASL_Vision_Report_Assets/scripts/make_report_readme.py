"""Write README.md, key_facts.md and references.bib into the report-asset folder (numbers read from results/)."""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1])
RES = ROOT / "results"


def m(mid):
    return json.loads((RES / mid / "metrics.json").read_text())


rows = list(csv.DictReader(open(OUT / "figure_index.csv", encoding="utf-8")))
n_people = sum(r["contains_people"].startswith("YES") for r in rows)
ds = json.loads((RES / "dataset.json").read_text())
fresh = json.loads((ROOT / "docs/parity_results_fresh.json").read_text())
ml = json.loads((ROOT / "docs/memory_latency_local.json").read_text())
sw, gt, st, r3 = (m(k) for k in ["swin3d_t_1000", "geometry_transformer_1000", "stgcn_1000", "r3d18_dualstream_1000"])
g300, g500, g1200 = (m(f"geometry_transformer_{k}") for k in (300, 500, 1200))
s = ds["subset_1000"]

readme = f"""# ASL Vision -- Report Assets

**Thesis:** Building a Word-Level American Sign Language Recognition System: A Multi-Architecture Design and
Evaluation Study

Everything here is generated from the project's own evaluation outputs (Kaggle runs) and local measurements.
No number was estimated. `figure_index.csv` lists **every image** with a ready-to-use title, caption, source
file and whether it shows people ({len(rows)} images in total).

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
| `D_RESTRICTED_contains_dataset_signers/` | {n_people} figures showing **real ASL Citizen participants** -- read the licence note below |
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
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

facts = f"""# Key facts and numbers

## Project
- Thesis: *Building a Word-Level American Sign Language Recognition System: A Multi-Architecture Design and Evaluation Study*
- Task: word-level (isolated sign) recognition -- one sign per video clip, 16 sampled frames per clip.
- Dataset: {ds['name']} ({ds['publisher']}); {ds['cached_videos']:,} videos / {ds['glosses_in_cache']:,} glosses preprocessed.
- 1,000-sign subset: {s['videos']:,} clips ({s['train']:,} train / {s['val']:,} val / {s['test']:,} test), {s['videos_per_class_min']}-{s['videos_per_class_max']} clips per sign, signer-independent splits.

## Results, 1,000 signs (test, {s['test']:,} clips)
| Model | Top-1 | Top-5 | Macro F1 | Params | CPU latency |
|---|---|---|---|---|---|
| Video Swin-T | {sw['test']['top1']:.1%} | {sw['test']['top5']:.1%} | {sw['test']['macro_f1']:.3f} | {sw['total_params'] / 1e6:.1f} M | {sw['cpu_latency']['mean_s'] * 1000:.0f} ms |
| Geometry Transformer | {gt['test']['top1']:.1%} | {gt['test']['top5']:.1%} | {gt['test']['macro_f1']:.3f} | {gt['total_params'] / 1e6:.2f} M | {gt['cpu_latency']['mean_s'] * 1000:.0f} ms |
| ST-GCN | {st['test']['top1']:.1%} | {st['test']['top5']:.1%} | {st['test']['macro_f1']:.3f} | {st['total_params'] / 1e6:.2f} M | {st['cpu_latency']['mean_s'] * 1000:.0f} ms |
| R3D-18 Dual-Stream (v9) | {r3['test']['top1']:.1%} | {r3['test']['top5']:.1%} | {r3['test']['macro_f1']:.3f} | {r3['total_params'] / 1e6:.1f} M | {r3['cpu_latency']['mean_s'] * 1000:.0f} ms |

## Vocabulary scaling (Geometry Transformer, test top-1 / top-5)
- 300 signs: {g300['test']['top1']:.1%} / {g300['test']['top5']:.1%}
- 500 signs: {g500['test']['top1']:.1%} / {g500['test']['top5']:.1%}
- 1,000 signs: {gt['test']['top1']:.1%} / {gt['test']['top5']:.1%}
- 1,200 signs: {g1200['test']['top1']:.1%} / {g1200['test']['top5']:.1%}

## Deployment
- Free hosting: Streamlit Community Cloud (CPU); weights on Hugging Face, downloaded on demand; one model in memory at a time.
- Peak process memory (local): about {ml['models']['swin3d_t_1000']['rss_peak_mb']:,} MB with Video Swin-T loaded.
- MediaPipe feature extraction per clip (local CPU): {ml['feature_extraction_s']:.1f} s; model inference: Swin {ml['models']['swin3d_t_1000']['infer_ms_2threads']:,} ms, Geometry Transformer {ml['models']['geometry_transformer_1000']['infer_ms_2threads']} ms.
- Parity test ({fresh['n_videos']} raw test videos): the app's predictions match predictions on the training-cache features for
  {fresh['models']['swin3d_t_1000']['top1_agreement_cache_vs_app']:.0%} (Swin) / {fresh['models']['geometry_transformer_1000']['top1_agreement_cache_vs_app']:.0%} (Geometry Transformer) of clips;
  repeated runs give identical predictions (100%). Exact equality is impossible because MediaPipe's tracking state carried over between videos when the cache was built.
- R3D-18 re-evaluation with the app's model code reproduced the recorded test top-1 exactly ({r3['test']['top1']:.3f}).
"""
(OUT / "key_facts.md").write_text(facts, encoding="utf-8")

bib = r"""% Verify every entry against the publisher's page before submission.
@inproceedings{desai2023aslcitizen,
  title     = {{ASL Citizen}: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author    = {Desai, Aashaka and Berger, Lauren and Minakov, Fyodor O. and Milan, Vanessa and Singh, Chinmay and Pumphrey, Kriston and Ladner, Richard E. and Daum{\'e} III, Hal and Lu, Alex X. and Caselli, Naomi and Bragg, Danielle},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2023}
}
@inproceedings{liu2022videoswin,
  title     = {Video Swin Transformer},
  author    = {Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
@inproceedings{tran2018closer,
  title     = {A Closer Look at Spatiotemporal Convolutions for Action Recognition},
  author    = {Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2018}
}
@inproceedings{yan2018stgcn,
  title     = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author    = {Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2018}
}
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}
@article{lugaresi2019mediapipe,
  title   = {{MediaPipe}: A Framework for Building Perception Pipelines},
  author  = {Lugaresi, Camillo and Tang, Jiuqiang and Nash, Hadon and McClanahan, Chris and Uboweja, Esha and Hays, Michael and Zhang, Fan and Chang, Chuo-Ling and Yong, Ming Guang and Lee, Juhyun and Chang, Wan-Teh and Hua, Wei and Georg, Manfred and Grundmann, Matthias},
  journal = {arXiv preprint arXiv:1906.08172},
  year    = {2019}
}
@article{kay2017kinetics,
  title   = {The {Kinetics} Human Action Video Dataset},
  author  = {Kay, Will and Carreira, Jo{\~a}o and Simonyan, Karen and Zhang, Brian and Hillier, Chloe and Vijayanarasimhan, Sudheendra and Viola, Fabio and Green, Tim and Back, Trevor and Natsev, Paul and Suleyman, Mustafa and Zisserman, Andrew},
  journal = {arXiv preprint arXiv:1705.06950},
  year    = {2017}
}
@inproceedings{szegedy2016rethinking,
  title     = {Rethinking the Inception Architecture for Computer Vision},
  author    = {Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  note      = {Introduces label smoothing},
  year      = {2016}
}
@inproceedings{paszke2019pytorch,
  title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019}
}
"""
(OUT / "references.bib").write_text(bib, encoding="utf-8")

(OUT / "scripts").mkdir(exist_ok=True)
for f in ["make_report_assets.py", "make_app_screenshots.py", "make_report_readme.py", "parity_test.py"]:
    shutil.copy2(ROOT / "scripts" / f, OUT / "scripts" / f)
print("README, key_facts, references written")
