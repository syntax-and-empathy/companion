# AI vs Human

Generated: 2025-11-11T19:11:30.076538Z  
Pipeline Version: 3.4

## CONTENTS

This bundle contains all outputs from a complete pipeline run analyzing
12 modules:

- nltk/ (9 files)
- transformers/ (7 files)
- nli/ (2 files)
- textstat_lex/ (8 files)
- spacy/ (9 files)
- semantic/ (10 files)
- ruptures/ (3 files)
- bertopic/ (13 files)
- final/ (4 files)
- calibration/ (3 files)
- lexicons/ (2 files)
- rapidfuzz/ (10 files)

## KEY FILES

outputs/final/
- content_complete_summary.json.   : Complete analysis schema
- report.html                      : Interactive HTML report
- [timeline_heatmap.png](#timeline-heatmap)             : Attribution timeline visualization
- [hybrid_map.png](#hybrid-map)                   : Change-point detection map

outputs/calibration/
- labels.parquet                   : Segment labels (human/synthetic/hybrid/uncertain)
- segments.parquet                 : Segment boundaries and features

outputs/ruptures/
- hybrid_seams.parquet             : Detected change-points
- feature_fusion.parquet           : Normalized feature matrix for detection

outputs/nltk/
- fw_burstiness_windows.parquet    : Window-level features (basis for all modules)

outputs/spacy/
- syntax_discourse_windows.parquet : Syntax & discourse features

outputs/lexicons/
- style_signals.parquet            : Hedge, idiom, intensifier densities

outputs/nli/
- nli_consistency.parquet          : Contradiction detection results

## QUICK START

1. View the HTML report: Open outputs/final/report.html in a browser
2. Access the schema: Load outputs/final/content_complete_summary.json
3. Analyze segments: Read outputs/calibration/labels.parquet with pandas

## MODULES

Module 0:  Foundations (paths, determinism, helpers)
Module 1:  Lexical features (textstat, wordfreq)
Module 2:  NLTK (stopwords, burstiness, windows)
Module 3:  spaCy (syntax, discourse markers)
Module 7:  Rapidfuzz (paraphrase entropy)
Module 8:  Custom lexicons (hedges, idioms, intensifiers)
Module 9:  NLI (contradiction detection)
Module 10: Ruptures (change-point ensemble)
Module 11: Calibration & labeling
Module 12: Schema & final report

## VISUALIZATIONS

### Index

**NLTK**
- [Stopword Radar](#stopword-radar)
- [Burstiness CV Trend](#burstiness-cv-trend)
- [Stopword Ratio Trend](#stopword-ratio-trend)

**Transformers**
- [Perplexity Histogram (Global)](#perplexity-histogram-global)
- [Perplexity Trend (Version)](#perplexity-trend-version)
- [Perplexity Trend (Windows)](#perplexity-trend-windows)

**NLI**
- [NLI Visuals](#nli-visuals)

**Textstat/Lexical**
- [Sentence Length Histogram](#sentence-length-histogram)
- [Flesch Reading Ease Trend](#flesch-reading-ease-trend)
- [Zipf Bins Stacked](#zipf-bins-stacked)
- [Zipf Histogram](#zipf-histogram)

**spaCy**
- [Coordination/Subordination Stack](#coordination-subordination-stack)
- [Length vs Depth](#length-vs-depth)

**Semantic**
- [Coherence Document Violin](#coherence-document-violin)
- [Cosine Heatmap](#cosine-heatmap)
- [Coherence Trend](#coherence-trend)
- [Window Coherence](#window-coherence)

**Ruptures**
- [Change Point Visuals](#change-point-visuals)

**BERTopic**
- [Topic Coherence](#topic-coherence)
- [Topic Counts Histogram](#topic-counts-histogram)
- [Topic Timeline](#topic-timeline)

**Final**
- [Timeline Heatmap](#timeline-heatmap)
- [Hybrid Map](#hybrid-map)

**Calibration**
- [Label Visuals](#label-visuals)

**Lexicons**
- [Style Signals Visuals](#style-signals-visuals)

**Rapidfuzz**
- [Entropy Ridge](#entropy-ridge)
- [Repetition Heatmap](#repetition-heatmap)

---

### Stopword Radar

![Stopword Radar](nltk/plots/stopword_radar_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Burstiness CV Trend

![Burstiness CV Trend](nltk/plots/trend_burstiness_cv_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Stopword Ratio Trend

![Stopword Ratio Trend](nltk/plots/trend_stopword_ratio_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Perplexity Histogram Global

![Perplexity Histogram (Global)](transformers/plots/ppl_hist_global.png)

[↑ Back to Index](#index)

---

### Perplexity Trend Version

![Perplexity Trend (Version)](transformers/plots/ppl_trend_version_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Perplexity Trend Windows

![Perplexity Trend (Windows)](transformers/plots/ppl_trend_windows_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### NLI Visuals

![NLI Visuals](nli/nli_visuals.png)

[↑ Back to Index](#index)

---

### Sentence Length Histogram

![Sentence Length Histogram](textstat_lex/plots/sentence_length_hist.png)

[↑ Back to Index](#index)

---

### Flesch Reading Ease Trend

![Flesch Reading Ease Trend](textstat_lex/plots/trend_flesch_reading_ease_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Zipf Bins Stacked

![Zipf Bins Stacked](textstat_lex/plots/zipf_bins_stacked_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Zipf Histogram

![Zipf Histogram](textstat_lex/plots/zipf_hist.png)

[↑ Back to Index](#index)

---

### Coordination Subordination Stack

![Coordination/Subordination Stack](spacy/plots/coord_subord_stack_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Length vs Depth

![Length vs Depth](spacy/plots/len_vs_depth_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Coherence Document Violin

![Coherence Document Violin](semantic/plots/coherence_doc_violin.png)

[↑ Back to Index](#index)

---

### Cosine Heatmap

![Cosine Heatmap](semantic/plots/cosine_heatmap_not-magic-measurement_v1.png)

[↑ Back to Index](#index)

---

### Coherence Trend

![Coherence Trend](semantic/plots/trend_coherence_not-magic-measurement_.png)

[↑ Back to Index](#index)

---

### Window Coherence

![Window Coherence](semantic/plots/win_coherence_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Change Point Visuals

![Change Point Visuals](ruptures/change_point_visuals.png)

[↑ Back to Index](#index)

---

### Topic Coherence

![Topic Coherence](bertopic/plots/topic_coherence_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Topic Counts Histogram

![Topic Counts Histogram](bertopic/plots/topic_counts_hist.png)

[↑ Back to Index](#index)

---

### Topic Timeline

![Topic Timeline](bertopic/plots/topic_timeline_not-magic-measurement.png)

[↑ Back to Index](#index)

---

### Timeline Heatmap

![Timeline Heatmap](final/timeline_heatmap.png)

[↑ Back to Index](#index)

---

### Hybrid Map

![Hybrid Map](final/hybrid_map.png)

[↑ Back to Index](#index)

---

### Label Visuals

![Label Visuals](calibration/label_visuals.png)

[↑ Back to Index](#index)

---

### Style Signals Visuals

![Style Signals Visuals](lexicons/style_signals_visuals.png)

[↑ Back to Index](#index)

---

### Entropy Ridge

![Entropy Ridge](rapidfuzz/plots/entropy_ridge.png)

[↑ Back to Index](#index)

---

### Repetition Heatmap

![Repetition Heatmap](rapidfuzz/plots/repetition_heatmap_not-magic-measurement.png)

[↑ Back to Index](#index)