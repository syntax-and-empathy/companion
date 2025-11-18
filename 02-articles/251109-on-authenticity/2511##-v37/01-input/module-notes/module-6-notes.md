Experts notes went into updated code.
--
Code Reviewer Notes:
Here are the notes I’d add for **Module 6 (BERTopic)**—short, paste-ready, and focused on QA + futureproofing rather than code.

---

### ✅ What’s solid

* Uses Module-2 window spans and preserves `(article_id, version_id, win_id)` alignment throughout.
* Determinism knobs in place (UMAP `random_state`, KMeans `random_state`, fixed seeds) and embeddings cached (`.npy` + `embedding_rows.parquet`).
* Graceful fallbacks: HDBSCAN→smaller min cluster→KMeans; recorded in `metadata.json`.
* Clear artifacts: `topics.parquet`, `topic_info.parquet`, `topic_metrics.parquet`, plots, and bundle with manifest/checksums.

---

### ⚠️ Add these to the notes (behavioral assumptions & caveats)

1. **Window alignment assumption**

   * All “churn” metrics align windows **by `win_id`** across versions. This presumes the window generator is stable across edits. If text length shifts, some windows may not be semantically comparable.
   * Mitigation: (for interpretation) treat reassignment/overlap as *positional* rather than semantic unless a similarity gate is applied.

2. **Determinism & runtime**

   * Seeds set for NumPy/torch; UMAP has `random_state=42`. HDBSCAN is deterministic given inputs.
   * For full reproducibility on CPU: consider fixing BLAS thread envs (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.). On GPU, PyTorch/CuDNN can introduce non-determinism; current code prefers CPU unless `LSA_ALLOW_CUDA=1`.

3. **Dependency install risk**

   * `hdbscan` sometimes compiles from source on niche platforms. Current cell doesn’t enforce wheels-only; be aware in constrained environments.

4. **Empty/short windows**

   * Windows with very few characters/tokens produce low-information embeddings and can inflate noise. Current pipeline embeds all windows (including empty strings) and does not filter.
   * Interpretation note: elevated **noise rate** may reflect short/boilerplate windows, not just clustering weakness.

5. **Metadata clarity**

   * When KMeans fallback is used, `metadata["hdbscan_params"]["min_cluster_size"]` currently reflects a value derived from the fallback dict (may show `k`). Record an explicit field, e.g., `"clustering_algo": "hdbscan" | "kmeans"` and store the *actual* HDBSCAN params separately from KMeans `k`. (Purely a documentation/interpretation concern.)

6. **Topic labels**

   * `topic_label` pulls top terms from BERTopic where available; for KMeans fallback it uses `km_<id>`. That’s expected—just note that labels are **not** globally re-aligned across runs if the clustering changes.

7. **Metrics interpretation**

   * Per-version **entropy** excludes noise; a second metric includes noise. Use both when diagnosing: decreases in entropy alongside reduced topic count and increased mean coherence are a good “regularization” cue (already flagged).
   * **Hungarian overlap** uses window intersections; ties can produce alternative optimal matchings. Treat Jaccard as *approximate* overlap signal, not identity mapping.

8. **Visuals**

   * Timelines color topics by `topic_id` (noise in gray). For high topic counts, the scatter can overplot; reading the figure is easier when filtered to the top-k topics by support. (Purely a UX note; artifacts are fine.)

9. **Scaling**

   * Embeddings are stored once per run and reused for metrics—good. For very large corpora, memory is dominated by `N × d` embeddings; keep `BATCH` moderate to avoid spikes.

10. **Offline/air-gapped runs**

    * SentenceTransformer will fetch the model if not cached. If internet is unavailable, pre-cache `all-MiniLM-L6-v2` in the environment; otherwise the install cell is fine.

---

### 🧪 Quick QA checks to keep in the notes

* **Consistency check:** `len(embeddings) == topics.parquet.shape[0] == embedding_rows.parquet['row_id'].max()+1`.
* **Noise rate per version:** `% noise` by version ≤ expected threshold for your corpus; big spikes usually indicate lots of short windows.
* **Fallback log:** `metadata.json` should include `"fallback": null` in the normal case; if KMeans triggered, sanity-check that `topic_info.parquet` has `km_*` names.
* **Births+deaths sanity:** In `topic_metrics.parquet` for each pair `(v, v+1)`, births and deaths shouldn’t both be zero unless topics carry forward 1-to-1.

---

### Optional (nice-to-have later; not blocking)

* Persist the BERTopic model (`.save`) so top terms/representations are inspectable without refitting.
* Record **noise\_rate** per version in `topic_metrics.parquet` to aid triage.
* Consider a **similarity gate** for reassignment (e.g., only count windows with token Jaccard or embedding cosine ≥ τ across versions) to separate content drift from position drift.

If you capture the above in the notes, Module 6 looks solid and ready to feed into downstream analysis.
