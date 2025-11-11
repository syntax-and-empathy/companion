# Roadmap (Colab-Optimized, Technology Modules Only)

**Global conventions (apply to every module)**

* **Cell ID:** `cell <major.minor>: <technology>: <purpose>`
* **Colab setup:** Every module begins with its *own* `pip` cell & imports (so modules can be moved/omitted without breakage).
* **Artifacts:** Each module writes to `outputs/<tech>/…` and never reads from future modules.
* **Plots:** Use **matplotlib** only (reliable on Colab). If a plot needs later tech, move the entire visualization to that later module.
* **Runtime discipline:** Seed set in each module; batch sizes capped; small models selected; `.cache` used by 🤗 to reduce re-downloads.

---

## Module 1 — `textstat` + `wordfreq` (Lexical baselines)

**Why first:** Fast, no heavy deps. Establishes readability, frequency, and basic lexical dispersion.

* **Installs:**
  `cell 1.1: textstat|wordfreq: install`

  ```bash
  %pip install -q textstat wordfreq regex
  ```
* **Imports & checks:**
  `cell 1.2: textstat|wordfreq: imports & sanity checks`
* **Features:** Readability indices, type–token ratio (TTR), Zipf frequency stats, punctuation/length distributions.
  `cell 1.3: textstat: readability metrics`
  `cell 1.4: wordfreq: Zipf frequency features`
  `cell 1.5: lexicons: function-word profile & TTR`
* **Visuals (local):** Histograms/boxplots for sentence length, Zipf freq; radar for function words (matplotlib).
  `cell 1.6: textstat|wordfreq: visuals — baseline distributions`
* **Output:** `outputs/textstat_lex/lexical_features.parquet`

---

## Module 2 — `nltk` (Stopwords, tokenization aids, burstiness)

**Why now:** Light dependency; supports function-word baselines and burstiness.

* **Installs:**
  `cell 2.1: NLTK: install & corpora download`

  ```bash
  %pip install -q nltk pandas
  python - <<'PY'
  import nltk
  nltk.download('punkt', quiet=True)
  nltk.download('stopwords', quiet=True)
  PY
  ```
* **Imports:**
  `cell 2.2: NLTK: imports & tokenizer init`
* **Features:** Function-word density, stopword profile, **burstiness** (token inter-arrival variance).
  `cell 2.3: NLTK: function-word & burstiness features`
* **Visuals (local):** Burstiness violin; function-word radar (matplotlib).
  `cell 2.4: NLTK: visuals — burstiness & radar`
* **Output:** `outputs/nltk/fw_burstiness.parquet`

---

## Module 3 — `spaCy` (Syntax metrics & discourse markers)

**Why next:** Enables clause chaining, parse-depth variance, coordination/subordination.

* **Installs:**
  `cell 3.1: spaCy: install & model download`

  ```bash
  %pip install -q spacy spacy-lookups-data
  python -m spacy download en_core_web_sm
  ```
* **Imports:**
  `cell 3.2: spaCy: imports & pipeline init`
* **Features:** Parse depth variance, clause chaining length, coord/subord ratios, basic discourse marker density (lexicon-based).
  `cell 3.3: spaCy: dependency parses → clause metrics`
  `cell 3.4: spaCy: discourse markers (lexicon-based)`
* **Visuals (local):** Length vs. parse-depth scatter; coord/subord stacked bars.
  `cell 3.5: spaCy: visuals — syntax & discourse summaries`
* **Output:** `outputs/spacy/syntax_discourse.parquet`

---

## Module 4 — `transformers` + `torch` (Pseudo-Perplexity / fluency)

**Colab-free constraints:** Prefer **CPU**; if T4 GPU appears, great. Use **`distilgpt2`** to keep VRAM/RAM small.

* **Installs:**
  `cell 4.1: transformers|torch: install`

  ```bash
  %pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  %pip install -q transformers==4.* accelerate
  ```
* **Model load:**
  `cell 4.2: transformers: load tokenizer/model (distilgpt2) [CPU/GPU auto]`
* **Features:** Sentence-level pseudo-PPL; optional SLOR if token log-probs accessible. Batch size ≤ 8; truncate long sentences.
  `cell 4.3: transformers: sentence pseudo-perplexity`
* **Visuals (local):** PPL distribution & trendline over windows.
  `cell 4.4: transformers: visuals — PPL distributions & trends`
* **Output:** `outputs/transformers/perplexity.parquet`

---

## Module 5 — `sentence-transformers` (Semantic drift & paraphrase mining)

**Why after 4:** Shares torch backend. Use **`all-MiniLM-L6-v2`** (light, accurate, Colab-friendly).

* **Installs:**
  `cell 5.1: sentence-transformers: install`

  ```bash
  %pip install -q sentence-transformers scikit-learn
  ```
* **Imports/Model:**
  `cell 5.2: sentence-transformers: load model (all-MiniLM-L6-v2)`
* **Features:** 3–5 sentence **window embeddings**, cosine drift between adjacent windows, simple paraphrase mining via nearest neighbors (k=3).
  `cell 5.3: sentence-transformers: windowed embeddings & drift`
  `cell 5.4: sentence-transformers: paraphrase mining & entropy proxy`
* **Visuals (local):** **Semantic drift heatmap** (matplotlib imshow).
  `cell 5.5: sentence-transformers: visuals — drift heatmap`
* **Output:** `outputs/sbert/semantic_windows.parquet`

---

## Module 6 — `BERTopic` (+ `umap-learn`, `hdbscan`) (Topic stability)

**Colab-free note:** Heavier; cap docs/windows (e.g., ≤ 2k). Use low-dim UMAP (n\_components=5) for speed.

* **Installs:**
  `cell 6.1: BERTopic|UMAP|HDBSCAN: install`

  ```bash
  %pip install -q bertopic umap-learn hdbscan
  ```
* **Fit & metrics:**
  `cell 6.2: BERTopic: init & fit on windows`
  `cell 6.3: BERTopic: topic stability & churn metrics`
* **Visuals (local):** Topic timeline; intra-topic coherence bars (matplotlib fallback).
  `cell 6.4: BERTopic: visuals — topic timeline & coherence`
* **Output:** `outputs/bertopic/topics.parquet`

---

## Module 7 — `rapidfuzz` (Paraphrase entropy & repetition)

* **Installs:**
  `cell 7.1: rapidfuzz: install`

  ```bash
  %pip install -q rapidfuzz
  ```
* **Features:** Local paraphrase entropy (edit distance across near-neighbors from Module 5), repetition/boilerplate spans.
  `cell 7.2: rapidfuzz: paraphrase entropy features`
* **Visuals (local):** Entropy ridge plot; repetition heatmap.
  `cell 7.3: rapidfuzz: visuals — entropy & repetition`
* **Output:** `outputs/rapidfuzz/paraphrase_entropy.parquet`

---

## Module 8 — Custom lexicons (Hedges/Idioms/Intensifiers)

**No pip dependency; ship plain text lists in repo** under `lexicons/`.

* **Resources:**
  `cell 8.1: lexicons: load hedges/idioms/intensifiers`
* **Features:** Densities per 100 tokens; **idiom delta** over windows (pairs from Module 5)
  `cell 8.2: lexicons: densities & deltas`
* **Visuals (local):** Idiom vs. coherence scatter (coherence proxy from Module 6); hedging bars.
  `cell 8.3: lexicons: visuals — idiom/coherence & hedging bars`
* **Output:** `outputs/lexicons/style_signals.parquet`

---

## Module 9 — NLI Consistency (`roberta-base-mnli`)

**Colab-free:** Use **`roberta-base-mnli`** (not `large`), stride adjacency only (window i vs i+1) to cap costs.

* **Installs/Model:**
  `cell 9.1: transformers: load NLI pipeline (roberta-base-mnli)`
* **Features:** Entailment/neutral/contradiction rates across adjacent windows; contradiction spikes.
  `cell 9.2: NLI: window adjacency checks (E/N/C rates)`
* **Visuals (local):** Contradiction rate timeline (matplotlib).
  `cell 9.3: NLI: visuals — contradiction timeline`
* **Output:** `outputs/nli/nli_consistency.parquet`

---

## Module 10 — Change-point ensemble (`ruptures`)

**Fuses all prior features.** Uses **Pelt** + **Binseg** + **Kernel**; consensus ≥2 to mark candidate seams.

* **Installs:**
  `cell 10.1: ruptures: install`

  ```bash
  %pip install -q ruptures numpy pandas
  ```
* **Fusion & detection:**
  `cell 10.2: ruptures: feature fusion matrix` (columns: PPL, semantic drift, topic churn, idiom Δ, parse-depth var, burstiness, contradiction rate, repetition)
  `cell 10.3: ruptures: detectors & consensus seams`
* **Visuals (local):** **Change-point overlay with shaded overlaps** (per detector and consensus).
  `cell 10.4: ruptures: visuals — seam overlay`
* **Output:** `outputs/ruptures/hybrid_seams.parquet`

---

## Module 11 — Calibration & Labeling (`scikit-learn`)

**Colab-free:** If no labeled refs, run unsupervised thresholds + **isotonic** on synthetic pseudo-labels (optional). Smooth on adjacency graph.

* **Installs:**
  `cell 11.1: sklearn: imports & config`

  ```bash
  %pip install -q scikit-learn
  ```
* **Calibration/Labeling:**
  `cell 11.2: calibration: build/fit scalers (refs if available)`
  `cell 11.3: labeling: classify windows {Human, Synthetic, Hybrid(H→S/S→H), Uncertain} + graph smoothing`
* **Visuals (local):** Reliability curves; if refs exist, simple confusion/precision-recall.
  `cell 11.4: calibration: visuals — reliability/variance`
* **Output:** `outputs/calibration/labels.parquet`

---

## Module 12 — Schema writer & final report (JSON + HTML)

**Writes your exact machine-readable schema; assembles strictly from earlier artifacts.**

* **Installs (optional):**
  `cell 12.1: schema: define & validate (pydantic/jsonschema)`

  ```bash
  %pip install -q pydantic jsonschema
  ```
* **Assemble & export:**
  `cell 12.2: schema: assemble & validate doc_overview, segments, hybrid_seams`
  `cell 12.3: schema: write JSON & generate minimal HTML report`
* **Visuals (final only, no new tech):**

  * **Timeline heatmap** of attribution labels & confidence (from Module 11).
  * **Hybrid map** overlay with S→H/H→S markers (from Modules 10–11).
* **Outputs:**

  * `outputs/final/content_complete_summary.json`
  * `outputs/final/report.html`

---

## Execution Order (strict)

**1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12**
If a visualization depends on future tech, the **entire viz stays with that future module**.

---

# Colab-Free Performance Guardrails (hard requirements)

* **Batch caps**: SBERT/NLI batch ≤ 8; PPL batch ≤ 8; max windows \~2,000 by default.
* **Model choices**: `distilgpt2`, `all-MiniLM-L6-v2`, `roberta-base-mnli`.
* **Memory controls**: Truncate sentences to ≤128 BPE tokens for PPL; pool embeddings at window level; avoid storing full token-level outputs.
* **Caching**: Honor `HF_HOME=/root/.cache/huggingface` (default); do not clear cache between modules.
* **GPU optional**: Auto-detect CUDA; otherwise use CPU with progress bars suppressed.

---

# Open-Source Reference Corpora (for Module 11 calibration)

You said you have no refs; here are **Colab-friendly** options with clear licensing and balanced coverage. Use small samples per domain to keep runtimes low.

## Human-authored (positive “human” class)

* **Wikipedia (Good/Featured articles)**: High-quality human prose; scrape via Hugging Face datasets: `wikipedia` (filter by quality tags if available) or fetch curated lists.
* **Project Gutenberg (non-fiction essays, public domain)**: Varied style; lightweight text.
* **AG News (train split)**: Human news headlines + descriptions (short, focused; good for sanity checks).
* **WMT News Crawl (year subset)**: Human news sentences; pick a small year slice (e.g., 2019 EN).

## Synthetic (positive “synthetic” class)

* **HC3 (Hello-SimpleAI/HC3)**: Human vs. ChatGPT Q\&A pairs across domains; widely used baseline for H/S detection.
* **Human-LLM pairs on Hugging Face** (small, curated sets; e.g., “llm-detect-ai-text” style datasets) — sample lightly to avoid domain bias.
* **Self-generated synthetic** (fallback): Use **distilgpt2** to create synthetic paraphrases of human windows (clearly marked as synthetic for calibration only). This keeps everything open & reproducible when third-party corpora are constrained.

## Hybrid post-edit (for S→H / H→S cues)

* **Create controlled hybrids** (recommended, reproducible):

  * **H→S**: Take human windows → paraphrase with `distilgpt2` (temperature \~0.7) → retain semantics, reduce idioms.
  * **S→H**: Take synthetic windows → light human edits (typos/idioms/clarifications) applied by your team.
  * These give you labeled seams to tune change-point thresholds without external dependencies.

> **Calibration plan:** Start with **unsupervised thresholds** → evaluate on the above small references → optionally fit **isotonic** on a tiny validation split. Always keep an **Uncertain** band to reflect overlap.

---

# Acceptance Criteria (“good enough to share”)

* Each module contains **its own installs/imports**, **features**, and **only its dependent visualizations**.
* **Cell IDs** follow the schema exactly.
* **All required visuals** are produced in their assigned modules:

  * Drift heatmap (5), Topic timeline (6), Entropy/repetition (7), Contradiction timeline (9), Change-point overlay (10), Timeline heatmap + Hybrid map (12), PPL distributions (4).
* **Final JSON** strictly matches your schema; report renders on Colab free.
* Runs end-to-end on **Colab free** with defaults (CPU OK, GPU optional).