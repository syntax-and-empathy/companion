# script-test-2

*A collection of scripts and data for analyzing text transformations during the writing process.*

## Overview
This directory contains data and scripts used to analyze the evolution of a text document through several editing stages (draft, refined, edited, final).  The analysis focuses on metrics such as word retention, sentence structure changes, and cosine similarity between versions using TF-IDF.  The results are presented in both CSV and visual formats. This collection serves as a record of the experiment and its findings. The analysis highlights the impact of the editing process on the document's structure and content.  This repository includes data files, analysis scripts, and generated charts.


## Contents
* `Article_metrics__by_stage_.csv` — Metrics (words, sentences, etc.) for each stage of article development.
* `ChatGPT-Python analysis results.md` — ChatGPT analysis of the Python script and results.
* `Consecutive_version_retention___similarity.csv` — Word and sequence similarity between consecutive versions.
* `Pairwise_cosine_similarity__TF-IDF_.csv` — Pairwise cosine similarity between all versions using TF-IDF.
* `base_agg.csv` — Aggregated metrics across all article versions.
* `charts_bundle.zip` — Zip archive containing visualizations of the analysis results (png format).


## Quick Start
The Python script (not included here) processes the four article versions (.md files, not included here) to generate the CSV files.  The charts were generated using a separate process, not included here.


## Conventions
CSV files use consistent naming and column headers.  Data represents different perspectives on the text transformation process.


---
Last updated: 2025-09-13
