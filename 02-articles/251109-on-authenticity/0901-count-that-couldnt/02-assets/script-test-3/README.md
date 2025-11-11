# script-test-3

*A collection of scripts and data for analyzing article revisions.*

## Overview
This directory contains Python scripts and CSV files used to analyze the evolution of an article across multiple revisions (draft, refined, edited, final).  The analysis focuses on metrics such as word count, sentence count, and token retention between revisions.  The data is presented in various formats including CSV tables and charts (bundled separately). This repository allows for a quantitative assessment of the editorial process. The included Python script facilitates the analysis, while the CSVs provide a summary of the key findings.

## Contents
- `0510-generate-data.py` — Python script to process and analyze article revisions.
- `Article_file_basic_counts.csv` — Basic word, character, and sentence counts per article version.
- `Pairwise_TF-IDF_cosine_similarity.csv` — Cosine similarity between article versions using TF-IDF.
- `Pairwise_difflib_ratio.csv` — Similarity ratio between article versions using difflib.
- `Per-stage_metrics.csv` — Per-stage metrics (draft, refined, edited, final).
- `Token_retention_transitions.csv` — Token retention percentage between different stages.
- `Uploaded_files__quick_preview.csv` — Quick preview of uploaded files.
- `base_agg.csv` — Aggregated metrics across article versions.
- `charts_bundle.zip` — (Contains charts summarizing the analysis; see individual files within the zip).


## Quick Start
To run the analysis, ensure you have the necessary Python libraries installed (`pandas`, `scikit-learn`). Then, execute the script: `python 0510-generate-data.py`


## Conventions
CSV files use comma as the delimiter.  The script is designed to handle multiple article files; modify file paths within the script as needed.


---
Last updated: 2025-09-13
