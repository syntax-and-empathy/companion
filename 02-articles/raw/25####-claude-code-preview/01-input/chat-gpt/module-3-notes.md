From our expert:
Nice—let’s lock Module 3. Add these to your notes so downstream work (and reviewers) have zero ambiguity:

# Add to Module-3 notes before moving on

## Provenance & config

* `version_order_source: "filename_prefix"` (01–04 rule confirmed).
* `pipeline`: `en_core_web_sm`; `spacy_version`; `components_enabled: ["tagger","lemmatizer","parser"]` (NER disabled).
* `nlp.max_length` value and `n_process/batch_size` used for `nlp.pipe`.
* `window_alignment_policy: "from_module_2_spans"` and `alignment_rule: "full_containment"` (or your final rule).
* `articles`, `versions_per_article_min/max`, `expected_versions: 4`.

## Label sets actually used

* `coord_labels_used`: `["cc","conj","parataxis"]`
* `subord_labels_used`: `["mark","advcl","ccomp","xcomp","acl","relcl","csubj","csubj:pass"]` (adjust to your exact set).
* Note any model-specific quirks (e.g., if `relcl` isn’t emitted and you proxied via `acl`).

## Discourse marker lexicon

* `lexicon_name`, `lexicon_size`, `lexicon_sha256`.
* `dm_match_policy`: `"lowercase; multiword-first; word-boundary; no overlap double-count"`.

## Schema commitments (so Module 10/11 don’t guess)

* **Doc-level table (`syntax_discourse.parquet`)** includes exactly:
  `article_id, version_id, version_tag, doc_id, n_sents_spacy, mean_sent_len_tok_spacy, std_sent_len_tok_spacy, depth_mean, depth_median, depth_std, depth_cv, depth_p90, depth_max, clauses_per_sent_mean, clauses_per_sent_median, clauses_per_sent_max, coord_rate, subord_rate, coord_subord_ratio, dm_density_per_100toks, dm_types, dm_type_token_ratio`.
* **Window-level table (`syntax_discourse_windows.parquet`)** includes exactly:
  Keys/spans: `article_id, version_id, version_tag, doc_id, win_id, win_label, span_basis, char_start, char_end, sent_start_index, sent_end_index, is_partial_tail`
  Metrics: `n_sents_win, n_tokens_win, depth_mean_win, depth_std_win, depth_max_win, clauses_per_sent_mean_win, clauses_per_sent_max_win, coord_rate_win, subord_rate_win, coord_subord_ratio_win, dm_density_per_100toks_win, dm_types_win, dm_type_token_ratio_win`.

## Sanity metrics (record the numbers you observed)

* `sent_boundary_disagreement_rate` (spaCy vs. M2 windows under your alignment rule).
* For each article: min/max/median of `depth_mean`, `coord_rate`, `subord_rate`, and `dm_density_per_100toks`.
* Any NaN rates triggered by short docs/windows (and why).

## Plot inventory (for reproducibility)

* List generated files per slug: `len_vs_depth_<slug>.png`, `coord_subord_stack_<slug>.png`.
* Note plot cap policy if you added one (e.g., “plotted first 10 slugs: false/true”).

## Known limitations (explicit)

* Parse-depth is model-dependent (small English pipeline); treat cross-domain comparisons cautiously.
* Clause counting is a proxy (finite verbs + subord labels), not gold-standard clause parsing.
* Discourse markers lexicon isn’t exhaustive; density TTR is a relative signal.

## Hand-off to Module 4 (what the next module can rely on)

* Module 4 (perplexity/fluency) should reuse the **same version order** and may **optionally** consume M2 windows for sentence-window PPL summaries.
* Char spans from M2/M3 are authoritative for mapping evidence snippets; do **not** re-segment.
* Join keys to use downstream:

  * Doc-level: `(article_id, version_id)` (or `doc_id`)
  * Window-level: `(article_id, version_id, win_id)`

If you add those items to the notes, Module 3 is fully documented and ready to fuse with later modules.
"""
From our code reviewer:
Absolutely—here’s a tight, copy-pasteable “Module 3 Notes” addendum you can drop into the notes file before moving to Module 4.

# Module 3 (spaCy) – Final Notes

## Environment & reproducibility

* Uses `spacy>=3.7,<3.8` with `en_core_web_sm` (NER excluded). First run may download the model.
* CPU-only. Tunables via env:

  * `LSA_SPACY_MAXLEN` (default 2,000,000) for long docs
  * `LSA_SPACY_PROCS` (default 1) and `LSA_SPACY_BATCH` (default 16) for `nlp.pipe`
* Wheels-first installs; base stack (NumPy/SciPy/etc.) left untouched.

## Inputs & ordering

* Document ordering comes from `version_id` (derived from filename prefix `01..04` in earlier modules).
* Preferred doc basis for parsing: `text_clean` if present, else `text`.
* If Module 2 windows exist (`outputs/nltk/fw_burstiness_windows.parquet`), they are honored using the `span_basis` recorded there.

## Parsing & sentence statistics (how they’re computed)

* Parser: `en_core_web_sm` with components except NER; sentence boundaries from spaCy.
* **Sentence depth**: max head-distance from each token to the sentence root; per-sentence depth is the max over tokens.
* **Finite clause heuristic**: counts tokens as finite predicates if `(pos_ == "VERB" and not AUX)` and (`VerbForm=Fin` in morph or `tag_ ∈ {VBD,VBP,VBZ}`).
* **Coordination/subordination**:

  * Subordination deps counted from: `{mark, advcl, ccomp, xcomp, acl, relcl, csubj, csubjpass, csubj:pass}`.
  * Coordination deps counted from: `{cc, conj, parataxis}`.
  * Reported as **rates per token** (`coord_rate`, `subord_rate`) and their ratio (`coord_subord_ratio`).

## Discourse markers

* Lexicon path: `lexicons/discourse_markers_en.txt`. If missing, a default list is created.
* Matching is **greedy, multi-word first, non-overlapping**, case-insensitive, with word boundaries.
* Metrics:

  * `dm_density_per_100toks` = 100 × (counts / doc tokens)
  * `dm_types` = number of distinct markers observed
  * `dm_type_token_ratio` = `dm_types / dm_total` (NaN if no matches)

## Windows alignment (Module 2 integration)

* Window metrics are computed **only if** M2 windows are present.
* Alignment policy: **full containment**—only sentences whose `[start_char, end_char)` lie entirely within a window’s `[char_start, char_end)` are included.
* Basis string for windows (`text` vs `text_clean`) follows each window’s recorded `span_basis`; falls back to any parsed basis if needed.
* Windows table: `outputs/spacy/syntax_discourse_windows.parquet`.

## Artifacts & contracts

* **Document-level** features: `outputs/spacy/syntax_discourse.parquet`
* **Window-level** features (optional): `outputs/spacy/syntax_discourse_windows.parquet`
* **Metadata**: `outputs/spacy/metadata.json` (records spaCy version, model, enabled components, window policy, lexicon info, article/version counts)
* **Plots**: `outputs/spacy/plots/`

  * `len_vs_depth_<slug>.png` (sentence length vs depth scatter by version)
  * `coord_subord_stack_<slug>.png` (stacked subordination/coordination rates by version)
* **Bundle**: `outputs/spacy/bundles/module3_artifacts_<timestamp>.zip` with checksums and `bundle_index.json`.

## Sanity ranges (quick QA)

* `depth_mean`: typically 1–8 for prose; spikes may indicate unusual sentence segmentation.
* `clauses_per_sent_mean`: usually 1–2.5; values >>3 often mean noisy clause detection or very complex syntax.
* `coord_subord_ratio`: \~0.5–2 for most prose; extreme ratios suggest style shifts (lists vs. heavy subordination).
* `dm_density_per_100toks`: often 0.1–2.0 for expository text; 0 is normal for terse copy.
* Empty or ultra-short docs produce NaNs by design (guarded to avoid div-by-zero).

## Known limitations / future polish

* **Model size**: `en_core_web_sm` is light; upgrading to `en_core_web_md`/`lg` improves POS/dep accuracy (especially clause/subord detection) at higher cost.
* **Finite clause heuristic** is approximate; auxiliaries and embedded clauses can be under/over-counted in edge cases.
* **Sentence boundary drift** vs. Module 2: spaCy boundaries may differ from NLTK’s; windows mitigate this by using char spans, but document-level stats are spaCy’s view.
* **Discourse markers**: the default list is small; consider swapping to a richer, curated English DMs list (e.g., PDTB-style connectives) and supporting multi-lingual variants in a later module.
* **Alignment rule**: full containment can yield zero-sentence windows near boundaries; acceptable for now, but we may add a “≥X% overlap” policy if needed.

## Operational tips

* For long documents that exceed `nlp.max_length`, set `LSA_SPACY_MAXLEN` higher or pre-chunk before parsing.
* Performance: prefer increasing `LSA_SPACY_BATCH` before bumping `LSA_SPACY_PROCS`; small models are memory-light but process-bound.
* If the model download is blocked (offline runs), install wheel + model ahead of time in the environment.

That’s everything I’d capture so downstream readers and Module 4 have a clear contract to build on.
