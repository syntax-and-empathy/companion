# Authenticity Analysis Workflow (11##)

*A reproducible Jupyter workflow for lexical-semantic analysis of text authenticity.*

## Overview
This directory contains Jupyter notebooks and supporting files for a multi-stage analysis of text authenticity.  The workflow is modular, allowing for independent execution and review of each stage.  Notebooks are designed for reproducibility and are optimized for Google Colab.  Each module focuses on a specific analytical task, producing artifacts that inform subsequent stages. The process involves lexical analysis, burstiness detection, syntactic parsing, and semantic embedding comparisons.

## Contents
* `ChatGPT-notebook-engineer-1.md` — Notebook engineering notes for the main attribution workflow.
* `ChatGPT-notebook-fixes-1.md` — Notes on troubleshooting notebook errors.
* `module-0.ipynb` — Foundation notebook with helper functions and environment setup.
* `module-1.ipynb` — Notebook performing lexical analysis using `textstat` and `wordfreq`.
* `module-2.ipynb` — Notebook for NLTK-based tokenization and burstiness analysis.
* `module-3.ipynb` — Notebook incorporating spaCy for syntactic parsing and discourse analysis.
* `module-4.ipynb` — Notebook using `distilgpt2` for perplexity calculations.
* `module-5.ipynb` — Notebook computing semantic coherence and drift using sentence embeddings.
* `module-6.ipynb` — Notebook applying BERTopic for topic modeling and analysis.
* `module-0-notes.md` — Feedback and notes for Module 0.
* `module-1-notes.md` — Feedback and notes for Module 1.
* `module-2-notes.md` — Feedback and notes for Module 2.
* `module-3-notes.md` — Feedback and notes for Module 3.
* `module-4-notes.md` — Feedback and notes for Module 4.
* `module-5-notes.md` — Feedback and notes for Module 5.
* `module-6-notes.md` — Feedback and notes for Module 6.
* `roadmap.md` — Detailed roadmap outlining the workflow's structure and dependencies.


## Quick Start
Each module is a self-contained Jupyter notebook.  Execute the notebooks sequentially, starting with `module-0.ipynb`.  Ensure necessary Python libraries are installed as indicated within each notebook.


## Conventions
Each module follows a consistent naming scheme and structure, beginning with its own installation and import cells to maintain modularity.  Artifacts are written to dedicated output directories, and plots use Matplotlib.  Deterministic configurations are maintained using fixed seeds where appropriate.

---
Last updated: 2025-09-12
