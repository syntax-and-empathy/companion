# script-test-1

*A collection of scripts and data for analyzing article revisions.*

## Overview
This directory contains Python scripts and data related to the analysis of multiple revisions of an article.  The analysis focuses on word count, sentence structure, and pairwise similarity between revisions.  The results are presented in CSV and markdown files, along with visualizations (located in the `charts_export` zip archive). The data can be used to understand the impact of different editorial stages on the final version of the article.  This project uses a bag-of-words approach for text similarity calculations.

## Contents
* `Article_Stage_Metrics.csv` — Metrics (characters, words, sentences, unique words) for each stage of the article.
* `ChatGPT-Python script execution results.md` — ChatGPT session detailing the execution of Python scripts and analysis.
* `Pairwise_Similarity__Cosine__bag-of-words_.csv` — Pairwise cosine similarity scores between article revisions.
* `Successive_Stage_Word_Retention____.csv` — Word retention percentage between successive article revisions.
* `base_agg.csv` — Aggregated metrics (word, sentence, and paragraph retention) across revisions.
* `charts_export.zip` —  Archive containing visualizations of the analysis results.


## Quick Start
The Python scripts used for this analysis are not directly included. The `ChatGPT-Python script execution results.md` file details the analysis process and results obtained using these scripts.  The data files, such as `Article_Stage_Metrics.csv`, can be directly imported and analyzed using various data analysis tools.

## Conventions
CSV files use a comma as a separator.  File names are descriptive and use underscores for clarity.


---
Last updated: 2025-09-13
