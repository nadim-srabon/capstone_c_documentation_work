# Formatting Instructions: Capstone Final Report (LaTeX)

**Thesis:** *Building a Word-Level American Sign Language Recognition System: A Multi-Architecture Design and Evaluation Study*

This document is the style guide for the final report. **Anyone who writes or edits a chapter must follow it**, so that every part of the report looks as if one person wrote it. All the formatting is already set up in the preamble. Normally you only write text, figures and tables using the patterns below. **Never change the layout inside a chapter file.**

> Chapter 1 ([chapters/ch1_introduction.tex](chapters/ch1_introduction.tex)) is the reference implementation. When unsure, copy what it does.

---

## 1. Project structure

```
report/
├── main.tex                     ← master file: switches, order of parts (do not add content here)
├── setup/
│   ├── preamble.tex             ← ALL packages + page/heading/caption/TOC formatting
│   └── macros.tex               ← names, model names, helper commands, \todo / \verify
├── frontmatter/                 ← title page, declaration, approval, acknowledgement,
│                                   abstract, list of abbreviations
├── chapters/
│   ├── ch1_introduction.tex
│   ├── ch2_related_works.tex
│   ├── ch3_materials_and_method.tex
│   ├── ch4_data_description.tex
│   ├── ch5_results_and_discussion.tex
│   └── ch6_conclusion.tex
├── backmatter/
│   ├── appendix_a.tex           ← supplementary figures / code
│   └── co_po_mapping.tex        ← "Mapping of Course and Program Outcomes" (400-A/B/C tables)
├── figures/
│   ├── ch1/  ch3/  ch4/  ch5/   ← one folder per chapter
│   └── common/                  ← logo, figures used in several places
├── references.bib               ← THE ONLY bibliography file
├── formatting_instructions.md   ← this file
└── status.md                    ← progress tracker (update it when you finish something)
```

**Rules**

- One chapter = one file in `chapters/`. Each file starts with the header comment block (title + status) and `\chapter{...}\label{ch:...}`.
- Do **not** add `\usepackage` lines inside chapter files. If you truly need a new package, add it to `setup/preamble.tex` **and** record it in §14 of this file and in `status.md`.
- Do **not** use `\newpage`, `\vspace`, `\\` for spacing, or font-size commands in body text. The only exceptions are tables (`\small`) and the title/declaration pages.

---

## 2. Building the PDF

The engine is **pdfLaTeX + BibTeX** (not XeLaTeX, not biber).

```bash
cd report
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

or `latexmk -pdf main.tex` (works on Overleaf and TeX Live; on this machine MiKTeX's latexmk wrapper needs Perl, so use the four commands above).

**Overleaf:** upload the whole `report/` folder, set *Compiler = pdfLaTeX*, *Main document = main.tex*. It compiles as is.

**Before you hand over a chapter,** the build log (`main.log`) must contain **no** `!` errors, **no** `undefined` references/citations and **no** `Overfull \hbox` warnings. Check with:

```bash
grep -nE "^!|undefined|Overfull" main.log
```

### Switches in `main.tex`

| Switch | Default | Effect |
|---|---|---|
| `\printversionfalse` / `\printversiontrue` | false | `true` makes all links black for the printed (black-book) copy |
| `\finalfalse` / `\finaltrue` | false | `true` hides all red `\todo{}` notes and orange `\verify{}` marks. Set it to `true` only for the final submission |

---

## 3. Page layout (set in `setup/preamble.tex`, section 2)

| Property | Value |
|---|---|
| Document class | `report`, 12 pt, A4, one-sided |
| Margins | **left 1.5 in** (binding), right 1 in, top 1 in, bottom 1 in |
| Header | left: `Chapter N. Title` (grey, small); right: short title *Word-Level ASL Recognition*; thin rule underneath |
| Footer | page number, centred |
| First page of a chapter | page number only (no header) |
| Page numbers | front matter in roman (i, ii, …); from Chapter 1 onward in arabic (1, 2, …) |

## 4. Fonts, spacing, paragraphs

| Property | Value |
|---|---|
| Body font | **LaTeX default font: Computer Modern**, loaded as `lmodern` (Latin Modern, the scalable version) for text and maths. Do not load any other font package (no `times`, `newtx`, `mathptmx`, `fontspec`) |
| Body size | 12 pt |
| Line spacing | **1.5** (`\onehalfspacing`) |
| Paragraph | first-line indent 1.5 em **and** 6 pt space between paragraphs |
| Alignment | fully justified (default). Do not use `\raggedright` in body text |
| Widows/orphans | suppressed globally |
| Monospace | `\code{file_name.py}` for files, functions and variables |

`amsmath` and `amssymb` are already loaded for maths symbols.

---

## 5. Headings and numbering

| Level | Command | Look | In TOC? |
|---|---|---|---|
| Chapter | `\chapter{Title}` | centred, blue, "CHAPTER N" on a small line, large title, blue rule underneath | yes, as **Chapter N  Title** |
| Section | `\section{Title}` | 14 pt bold, numbered `1.2` | yes |
| Subsection | `\subsection{Title}` | 12 pt bold, numbered `1.2.1` | yes |
| Subsubsection | `\subsubsection{Title}` | 12 pt bold italic, numbered `1.2.1.1` | no |
| Paragraph heading | `\paragraph{Title.}` | inline bold run-in | no |

- Use **Title Case** for chapter, section and subsection titles (*Dataset Preprocessing*, *Why Transformer Models instead of RNN or Encoder–Decoder*).
- The section titles must stay as defined in the skeleton files, because they follow the table of contents prescribed by the department. Do not rename them without updating `status.md`.
- If a section title is too long for the TOC, give a short version: `\section[Short title]{Full long title}`.
- **Never skip a level** (no `\subsubsection` directly under a `\section`).
- Unnumbered front/back-matter chapters use `\chapter*{...}` followed by `\addcontentsline{toc}{chapter}{...}`.

### Labels: mandatory, with fixed prefixes

| Object | Prefix | Example |
|---|---|---|
| Chapter | `ch:` | `\label{ch:method}` |
| Section / subsection | `sec:` | `\label{sec:feature-extraction}` |
| Figure | `fig:` | `\label{fig:architecture-overview}` |
| Table | `tab:` | `\label{tab:model-comparison}` |
| Equation | `eq:` | `\label{eq:attention}` |
| Appendix | `app:` | `\label{app:a}` |
| Listing | `lst:` | `\label{lst:video-processor}` |

Labels are lowercase words joined with hyphens. **Existing labels used by other chapters must not be renamed** (see the label registry in `status.md`).

---

## 6. Cross-references

Always use **`\cref{}`** (cleveref). It writes the word automatically ("Figure 1.1", "Table 3.2", "Chapter 5", "Section 3.1.2", "Equation (1.1)"). Use `\Cref{}` at the start of a sentence.

```latex
As shown in \cref{fig:project-phases}, ...
\Cref{tab:headline} summarises ...
The details are in \cref{ch:method,ch:results}.     % -> "Chapters 3 and 5"
```

Do **not** type "Figure~\ref{...}" by hand. Never write a number such as "Table 5" directly in the text.

---

## 7. Figures

**Template**

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\linewidth]{ch3/fig02_02_architecture_overview.pdf}
\caption{Overview of the four 1,000-sign architectures: input representation,
backbone and classification head.}
\label{fig:architecture-overview}
\end{figure}
```

**Rules**

1. Placement is always `[htbp]`. Never use `[H]` unless a figure *must* stay exactly in place.
2. The file path is relative to `figures/` (set by `\graphicspath`), so write `ch3/name.pdf`, not `figures/ch3/name.pdf`.
3. Copy the figure into the **chapter's folder** in `figures/`. Keep the original file name from `master_report_metadata/` so its source can be traced.
4. Prefer the **vector PDF** version of each figure in `master_report_metadata/A_generated_figures/`, and use the PNG only if no PDF exists. Screenshots are PNG.
5. Width: `0.9\linewidth` for wide diagrams and plots, `0.6–0.75\linewidth` for square plots, `\linewidth` maximum. Never set a height that distorts the image.
6. The **caption goes below** the figure (automatic). It is one or more full sentences ending with a full stop. The first sentence says *what* is shown, and the second sentence (optional) gives the takeaway. Mention the data source if it is not obvious.
7. Every figure **must be referenced in the text before or near where it appears** (`\cref{fig:...}`).
8. Side-by-side panels use `subcaption`:

```latex
\begin{figure}[htbp]
\centering
\begin{subfigure}[b]{0.48\linewidth}
  \centering\includegraphics[width=\linewidth]{ch5/a.pdf}
  \caption{Video Swin-T}\label{fig:cm-swin}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.48\linewidth}
  \centering\includegraphics[width=\linewidth]{ch5/b.pdf}
  \caption{Geometry Transformer}\label{fig:cm-geo}
\end{subfigure}
\caption{Confusion sub-matrices of the two best 1,000-sign models.}
\label{fig:cm-both}
\end{figure}
```

9. **Diagrams drawn in TikZ** use the same visual language as Figure 1.1: boxes with `draw=thesisblue, very thick, rounded corners=3pt, fill=boxfill`, grey `-{Stealth}` arrows (`color=thesisgray`), `\footnotesize` text. Copy the style block from `ch1_introduction.tex`.

### Privacy and licence rule (important)

The ASL Citizen licence forbids redistributing the data, and its videos show identifiable people.

- **Never** use images from `master_report_metadata/D_RESTRICTED_contains_dataset_signers/` without written approval from the supervisor. If one is approved, the faces must be blurred.
- App screenshots in `C_app_screenshots/` are already blurred and may be used.

---

## 8. Tables

**Template**

```latex
\begin{table}[htbp]
\centering
\caption{Test results of the four 1,000-sign models (12,560 test clips).}
\label{tab:model-comparison}
\small
\begin{tabular}{@{}l l c c r@{}}
\toprule
\rowcolor{tablehead}
\textbf{Model} & \textbf{Input} & \textbf{Top-1} & \textbf{Top-5} & \textbf{Params} \\
\midrule
Video Swin-T         & RGB video  & 66.8\,\% & 88.9\,\% & 28.6\,M \\
Geometry Transformer & Landmarks  & 57.5\,\% & 81.8\,\% & 1.91\,M \\
\bottomrule
\end{tabular}
\end{table}
```

**Rules**

1. The **caption goes above** the table (automatic), followed by `\label`, then `\small`.
2. Use **booktabs** only: `\toprule`, `\midrule`, `\bottomrule`. **No vertical lines** and no `\hline`.
3. The header row is shaded with `\rowcolor{tablehead}` and every header cell is `\textbf{}`.
4. Text columns are left-aligned (`l`, or `L{3cm}` for wrapped text). Numeric columns are centred (`c`) or right-aligned (`r`). Put the **best value in bold** in comparison tables.
5. For tables of full text width with wrapping text, use `tabularx` with the `Y` column type (ragged-right `X`). Example: `\begin{tabularx}{\linewidth}{@{}c L{4.6cm} Y@{}}` (see `tab:problems-responses` in Chapter 1).
6. Tables longer than one page use `longtable`. The literature summary table in Chapter 2 must be a `longtable`.
7. If a table is slightly too wide, first add `\setlength{\tabcolsep}{4pt}` inside the table environment. If that is not enough, shorten the headers. Never use `\resizebox` on tables with text.
8. Ready-made LaTeX versions of the result tables exist in `master_report_metadata/tables/*.tex`. **Reformat them to this template** (booktabs, shaded header, `\small`) instead of pasting them unchanged.
9. Every table **must be referenced in the text** with `\cref{tab:...}`.

---

## 9. Equations

- Use `equation` for numbered equations, and `align` for multi-line equations. Every equation that is referred to gets a `\label{eq:...}`.
- Numbering is per chapter, e.g. (1.1) and (3.4), and is automatic.
- An equation is part of the sentence. Put punctuation after it, and define every symbol right after it ("where $Q$, $K$ and $V$ are …").
- Vectors and matrices are bold upright: `\mathbf{x}_t`, `\mathbf{W}`. Sets are calligraphic: `\mathcal{V}`. Use `\operatorname{softmax}`, `\arg\max`.
- Inline maths only for symbols and short expressions: `$T=16$`, `$K$`.

```latex
\begin{equation}
  \operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
  = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V},
  \label{eq:attention}
\end{equation}
where $d_k$ is the key dimension.
```

---

## 10. Citations and bibliography

- **Style:** IEEE numeric (`IEEEtran.bst`), numbered in order of first citation. The heading is "Bibliography".
- Cite with `\cite{key}`. Several sources go in one command: `\cite{li2020wlasl,desai2023aslcitizen}` gives "[9], [10]".
- Put a non-breaking space before the citation when it follows a word: `MediaPipe~\cite{lugaresi2019mediapipe}`. A normal space (as in Chapter 1) is also acceptable, but use one form consistently within a chapter.
- **All entries live in `references.bib`.** Before adding a key, search the file, because most papers from Capstone A/B are already there.
- **Key format:** `firstauthorYEARkeyword`, all lowercase (`liu2022videoswin`, `hu2023signbertplus`).
- Our own earlier reports are cited as `srabon2026capstoneA` and `srabon2026capstoneB`.
- Entries with unknown details carry a `% VERIFY` comment **above** the entry. Comments are not allowed *inside* an entry, because they break BibTeX.
- Do not cite Wikipedia or blogs. For software, cite the paper if one exists (PyTorch, MediaPipe), otherwise a `@misc` with the URL and an access date.

---

## 11. Project-wide macros (`setup/macros.tex`)

Always use these instead of typing names by hand, so that spelling stays identical:

| Macro | Prints |
|---|---|
| `\aslc{}` | ASL Citizen |
| `\wlasl{}` | WLASL |
| `\swin{}` | Video Swin-T |
| `\geotr{}` | Geometry Transformer |
| `\stgcn{}` | ST-GCN |
| `\rthreed{}` | R3D-18 Dual-Stream |
| `\mediapipe{}` | MediaPipe |
| `\capA{}`, `\capB{}`, `\capC{}` | Capstone A / B / C |
| `\ProjectName` | ASL Vision |
| `\AppURL`, `\RepoURL` | live app / GitHub links |
| `\topone{}`, `\topfive{}` | top-1, top-5 |
| `\gloss{DONUT}` | sign gloss in small caps (DONUT) |
| `\code{video_processor.py}` | monospace code/file name |
| `\etal` | *et al.* |
| `\todo{...}` | red draft note (hidden when `\finaltrue`) |
| `\verify{...}` | orange text that still needs fact-checking |

Remember the `{}` after text macros that are followed by a space (`\swin{} reaches`), otherwise the space is swallowed.

---

## 12. Writing style and terminology

**Voice and tense**

- Formal academic English, **first person plural** ("we trained", "our model"). Do not write "I" or "the author".
- Use the **past tense** for what we did and measured ("The model reached 57.5 % top-1"). Use the **present tense** for facts and for what the report contains ("Table 5.1 shows …").
- Use **British spelling** throughout: *normalise, analyse, modelling, colour, behaviour, generalise, summarise*. Exception: proper names and quoted titles (*Normalization* in a paper title stays as is).
- Write short, clear sentences. No marketing language (avoid "revolutionary", "state-of-the-art" unless it is cited, and "seamless"). No contractions (*don't* becomes *do not*).
- Be honest about limitations and negative results. This is a strength of the B report that we keep.

**No internal development labels (strict rule).** The report is documentation, not a lab diary. Never write:

- version numbers of our code or models: *v9, v10, version 7, "the v10 model"*;
- notebook or file names in prose: *"the 1000-class notebook"*, `train_1000_swin3d_t.ipynb`, *"Section 3 of the notebook"*;
- internal jargon: *pilot run, smoke test, cache shard id00, fingerprint, changelog, audit guide*.

Instead, name models by **what they are** (the Geometry Transformer, the final Capstone B dual-stream model) and describe development steps by **what changed and why** ("we then replaced the two hand crops with a single signer crop because ..."). Use *development iteration* or *refinement* if a sequence must be described. A file name may appear only in Chapter 3.3–3.4 or Appendix A, when it describes the delivered application's code structure, written with `\code{}`.

**Numbers and units**

| Case | Write | LaTeX |
|---|---|---|
| Percentages | 66.8 % | `66.8\,\%` (thin space). One decimal place for accuracies |
| Thousands | 1,000 / 12,560 | comma separator in running text |
| Vocabulary size | 1,000 signs, the 1,000-sign model | hyphenate when used as an adjective |
| Parameters | 28.6 M | `28.6\,M` |
| Time | 3 ms, 1,220 ms, 6.1 h | `3~ms` (non-breaking space) |
| Memory | 830 MB | `830~MB` |
| Ranges | 300–1,200 | `300--1,200` (en dash) |
| Small counts | spell out one to nine in prose ("nine notebook versions"), digits for 10 and above and for all measurements |

**Fixed terminology** (use exactly these forms)

| Use | Do not use |
|---|---|
| sign language recognition (SLR) | sign recognition, SL recognition |
| word-level / isolated sign recognition | word level, single-sign |
| gloss / glosses (the label of a sign) | class word, label word |
| signer-independent split | user-independent split |
| landmarks | keypoints *(only acceptable when quoting a paper)* |
| geometry-only model / landmark stream | pose-only, skeleton-only |
| appearance stream (RGB) | image stream, CNN stream |
| top-1 / top-5 accuracy | Top1, top 1 |
| validation, test | val, dev *(in prose)* |
| web application / the application | site, website, webapp |
| Capstone A/B/C | 400A, part A *(except in the CO/PO tables)* |

**Abbreviations:** spell out at the first use in **each chapter**, as in "spatial-temporal graph convolutional network (ST-GCN)", and add every new abbreviation to `frontmatter/abbreviations.tex` (alphabetical).

---

## 13. Lists

- Numbered lists for sequences, objectives and contributions. Bulleted lists for unordered items.
- Labelled lists like those in Chapter 1: `\begin{enumerate}[label=\textbf{O\arabic*.}, leftmargin=2.6em]`.
- Each item is either a full sentence (ending with a full stop) or a short phrase (no full stop). Do not mix the two in one list.
- Bold lead-in for long items: `\item \textbf{Short title.} Explanation ...`.
- Do not turn whole sections into bullet lists. Chapters are mainly prose.

---

## 14. Packages already loaded (do not load them again)

`fontenc, inputenc, lmodern, babel, microtype, geometry, setspace, xcolor[table], amsmath, amssymb, bm, siunitx, graphicx, float, booktabs, tabularx, multirow, longtable, array, makecell, subcaption, tikz (positioning, arrows.meta, shapes.geometric, fit, calc, backgrounds), caption, enumitem, titlesec, tocloft, fancyhdr, cite, listings, url, hyperref, cleveref`

**Colours:** `thesisblue` (headings, links, diagram borders), `thesisgray` (header text, arrows), `tablehead` (table header row), `boxfill` (diagram boxes, code background).

**Code listings:**

```latex
\begin{lstlisting}[language=Python, caption={Sampling 16 frames.}, label={lst:sampling}]
idx = np.linspace(0, n - 1, 16).round().astype(int)
\end{lstlisting}
```

---

## 15. Checklist before handing over a chapter

- [ ] The chapter compiles with **no errors, no undefined references, no overfull boxes**.
- [ ] Every figure and table has a caption, a label, and a `\cref` in the text.
- [ ] Figure files are inside `figures/chN/` and come from `master_report_metadata/` (no restricted signer images).
- [ ] Every number matches `master_report_metadata/key_facts.md` or `tables/*.csv`. Copy numbers, never retype them from memory.
- [ ] Every citation key exists in `references.bib`.
- [ ] Model and dataset names use the macros. Spelling is British.
- [ ] New abbreviations are added to the list of abbreviations.
- [ ] `\todo{}` notes are either resolved or recorded in `status.md`.
- [ ] `status.md` is updated (chapter status, figures/tables used, labels added).
