Module 0 Feedback:
Thanks for sharing `module-0.ipynb`. I reviewed it against our roadmap rules (tech-scoped, dependency discipline, cell IDs, local scope, Colab-friendliness). Here’s a precise, developer-oriented handoff to tighten Module 0 before we start Module 1.

# What’s good

* Clear **foundations-only** scope: helpers, determinism, paths, light sanity checks. No tech installs — ✅ aligns with “Module 0 is not a technology module.”
* **Determinism**: seeds set — good.
* **Matplotlib smoke test** present — good signal that plotting works in Colab.
* **Config/dataclass** present — useful for later orchestration.

# Gaps to close (actionable)

## 1) Cell ID completeness (must fix)

* One code cell lacks a `cell <major.minor>: ...` header.
  **Action:** Add an ID so **all** code cells match the format.

  * Recommended label: `cell 0.1: foundations: environment verification & core imports`

## 2) Module 0 scope boundaries (guardrails)

* Ensure Module 0 **never** installs or imports technology packages (e.g., `textstat`, `wordfreq`, `nltk`, `spacy`, `transformers`, SBERT, BERTopic, rapidfuzz, ruptures, sklearn).
  **Action:** In the helper(s) that auto-import, **explicitly block** the above tech names with a comment:

  * “Technology imports are deferred to their modules; raise if requested here.”

## 3) Input handshake stub (lightweight; no tech)

* Current file doesn’t expose a minimal interface for later modules.
  **Action:** Provide a tiny **non-invasive** handshake that **does not** tokenize or segment, but asserts input presence and type:

  * Accept either `TEXT: str` or `DOCS: List[{'doc_id', 'text'}]`.
  * If only `TEXT`, wrap into `DOCS=[{'doc_id': 'doc_0001', 'text': TEXT}]`.
  * Do **not** do stopwording/segmentation here. Save that for Module 1/2.
  * If neither present, raise a **clear** `ValueError("Module 1 requires DOCS or TEXT.")` — this message is for the next module to catch.

## 4) Directory & logging hygiene

* I didn’t see any `outputs/` scaffolding created.
  **Action:** In Module 0, create just the **root** output dir (not tech-specific subdirs):

  * `Path("outputs").mkdir(exist_ok=True)`
  * Emit a small `outputs/_env.json` capturing:

    * `python_version`, `platform`, `is_colab`, `timestamp_utc`
    * `rng_seed` (used in this run)
    * `matplotlib_backend`
      This will help debugging without stepping into tech territory.

## 5) Version capture & environment detection

* You’re capturing versions (good), but make it explicit and Colab-aware.
  **Action:** Log:

  * `sys.version`, `sys.platform`, `os.environ.get('COLAB_GPU','0')` or `"COLAB_RELEASE_TAG" in os.environ` as `is_colab`
  * `pip list --format=json` *optional* (write to `outputs/_pip_list.json`) — keep size in mind.

## 6) Visualization smoke test robustness

* The current smoke test is fine, but **failures should not crash downstream**.
  **Action:** Wrap with try/except (it already is), and record pass/fail into a simple in-memory status registry (e.g., `FOUNDATION_STATUS = {...}`) and mirror to `outputs/_foundation_status.json`.

## 7) Sentence/windowing helpers (regex only, optional)

* You included a regex splitter/windowing stub — good. Make sure it’s clearly labeled as **heuristic** and **not used by Module 1** for metrics (Module 1 uses its own minimal regex splitter for its viz).
  **Action:** Add a comment: “Heuristic; authoritative tokenization/segmentation deferred to Module 2+.”

## 8) Side-effects: keep them minimal

* No tech installs in Module 0 — keep it that way. Avoid populating any `outputs/textstat_lex/...` paths; those belong to Module 1.

# Minimal acceptance checklist (Module 0)

* [ ] **Every code cell** starts with a correct ID header (`cell 0.x: ...`).
* [ ] No imports/installs of technology packages (listed above).
* [ ] Creates `outputs/` (root) only.
* [ ] Writes `outputs/_env.json` and `outputs/_foundation_status.json`.
* [ ] Exposes a **non-tokenizing** input handshake function that the Module 1 “imports & sanity checks” cell can call (e.g., `ensure_docs(TEXT, DOCS)`), raising a clear `ValueError` if neither is present.
* [ ] Matplotlib smoke test result recorded; failure **does not** stop the notebook.

# Suggested cell map (if you need labels)

* `cell 0.0: foundations: helpers (no installs, no heavy imports)` ✅ (present)
* `cell 0.1: foundations: environment verification & core imports` ⚠️ add ID
* `cell 0.2: foundations: imports, determinism, version capture` ✅ (present)
* `cell 0.3: foundations: outputs/ root & environment logs` (create `outputs/`, write `_env.json`)
* `cell 0.4: foundations: input handshake (ensure_docs)` (TEXT/DOCS wrapper, no tokenization)
* `cell 0.5: foundations: path utilities & status reporter` (writes `_foundation_status.json`)
* `cell 0.6: foundations: sentence segmentation & windowing (regex heuristic)` ✅ (present; document “heuristic only”)
* `cell 0.7: foundations: visualization smoke test (matplotlib only)` ✅ (present)

# Why this matters before Module 1

* Module 1 will rely on:

  * `ensure_docs(TEXT, DOCS)` (no NLP work)
  * `outputs/` root existing
  * deterministic seeds & environment logs (for reproducibility)
* Keeping Module 0 **tech-free** prevents dependency bleed and ensures our “technology-scoped module” rule holds.

If you ship these fixes, Module 1 can proceed cleanly with its own installs/imports, features, and visuals without retrofitting Module 0 later.

"""

Module 0 Triage:

Awesome—thanks for locking the cell IDs. Here’s a straight, code-free review of the **0.# Foundations** you now have, with keep/why and a couple of narrowly-scoped risks to watch as we move into Module 1. I’ll only call out action items if they’re essential; otherwise “keep as-is.”

---

### cell 0.0 — helpers

**Status:** ✅ keep as-is.
**Why:** Centralizes admin utilities (`_pip`, `try_import`, version helpers) without touching heavy libs. That aligns with the “single imports cell” rule and keeps later cells clean.

---

### cell 0.1 — minimal installs (only-if-missing; wheels-only)

**Status:** ✅ keep as-is.
**What’s good:**

* Explicit ABI order (NumPy → SciPy → scikit-learn → pandas → matplotlib).
* Wheels-only (`--only-binary=:all:`) avoids slow/fragile source builds.
* “Only if missing” respects Colab’s baseline and avoids the upgrade churn we saw in the logs.&#x20;

**Heads-up for later modules (no change here):** If a technology later *requires* a newer base (e.g., `umap-learn` pulling sklearn ≥1.6), we must either (a) pin that tech to a compatible version or (b) explicitly bump base in **that** module with a clear fail-fast message. Do **not** casually re-install base packages in random modules—this is exactly what caused the silent drift noted in the log (NumPy 2.3.2, sklearn 1.5.2 vs 1.6.x).&#x20;

---

### cell 0.2 — imports, determinism, version capture

**Status:** ✅ keep as-is.
**What’s good:**

* Single place importing heavy libs (NumPy/Pandas/Matplotlib/SciPy/sklearn).
* Determinism covered (seed + thread caps), with **opt-in** hash guard via `LSA_HASH_GUARD`.
* Thread override via `LSA_THREADS` is practical for CPU-only runs.

**Tiny suggestion (no change required):** If a later module spawns subprocesses, call out that `PYTHONHASHSEED` influences *new* interpreters only; we already document that.

---

### cell 0.3 — configuration & directories

**Status:** ✅ keep as-is.
**What’s good:**

* Portable defaults (`/content`) with an env override.
* Minimal, focused config fields that align to the roadmap (language, window size/stride, thresholds).

**Future nicety (optional):** A one-line “schema preview” in the cell output can help reviewers sanity-check types; not required now.

---

### cell 0.4 — status logging & self-report

**Status:** ✅ keep as-is (after your appended cell-level JSON).
**What’s good:**

* `report_status()` gives per-module pass/fail with timestamps.
* `dump_status_json()` writes UTF-8 machine-readable status—useful later when we export consolidated artifacts.
* Cell ends with a single JSON self-report, meeting your rule #9.

---

### cell 0.5 — offset-preserving normalization

**Status:** ✅ keep as-is.
**What’s good:**

* Correct CRLF→`\n` collapse with offset maps both directions.
* Idempotent whitespace normalization and Unicode line separators mapped to `\n`.
* Left-nearest `orig_to_norm` is consistent; if we need right-nearest later we can add a flag then.

---

### cell 0.6 — sentence segmentation & windowing

**Status:** ✅ keep with the improved regex.
**What’s good:**

* Regex now guards against false splits on initials (`(?<!\b[A-Z]\.)`).
* Windows derived from sentence spans preserve character offsets, which we’ll need for per-segment evidence.

**Scope note:** This is a lightweight heuristic (by design). If the expert later requests language-aware segmentation, we’ll stage that as its own “technology module” so we don’t bloat foundations.

---

### cell 0.7 — visualization smoke test

**Status:** ✅ keep as-is.
**What’s good:**

* Uses `plt` from the imports cell (no duplicate imports).
* Asserts that plotted axes actually contain data → catches silent rendering failures early.

---

## Cross-cell policy checks (important to keep)

* **No base upgrades in foundations:** We only install core libs if missing; we don’t pin/downgrade Colab’s stack. That prevents the resolver warnings and mid-run kernel restarts your log captured (NumPy 2.3.2, sklearn mismatches, etc.).&#x20;
* **Single heavy-imports cell (0.2):** Preserves reproducibility and makes failures obvious.
* **Wheels-only installs:** Cuts off the “tried to build from source” path that often trips Colab.
* **Self-reporting:** Every foundations cell now emits concise JSON—good for automation.

## What to watch as we proceed to Module 1

* **Tech modules must not re-install base packages.** If a technology needs a higher floor (e.g., sklearn ≥1.6), we’ll either pin that tech appropriately or gate the module with a clear error and rationale—no silent changes. Your server log shows exactly why we’re strict here.&#x20;
* **Constraints:** We did not add a global constraints file in foundations (by your direction to keep it lean). If we hit a package that aggressively upgrades bases, we’ll add a *module-local* install cell that: (1) pins that package’s version; (2) uses `--no-deps`; and (3) fails fast with guidance if incompatible—keeping foundations untouched.

If this all reads right to you, I’m ready to apply the same discipline to **Module 1 (Lexical Stylometry & Readability)**: self-contained installs (only tech libs), imports inside the module, and we stop at the first visualization until it renders cleanly.

"""

Yes—here are the **additions I recommend you append to the Module 0 notes** so the dev has zero ambiguity going into Module 1:

* **Run order & re-run rule:** Execute 0.0 → 0.1 → 0.2 → 0.3 → 0.4 → 0.5 → 0.6 → 0.7. It’s safe to re-run from **0.2** onward at any time; re-running **0.1** is only for missing packages.
* **Single-import policy:** Only **0.2** may import `numpy/pandas/matplotlib/scipy/sklearn`. Later cells must use those bindings—no re-imports.
* **Thread determinism:** Set `LSA_THREADS=1` by default (in 0.2); allow overrides via env var for profiling, but record the value in the self-report.
* **Seeding policy:** Use `LSA_SEED` (default 12345) to seed `random` and `numpy`; note that `PYTHONHASHSEED` affects only new processes.
* **Installs policy (0.#):** **Only-if-missing**, wheels-only (`--only-binary=:all:`), ABI order: NumPy → SciPy → scikit-learn → pandas → matplotlib. Never upgrade bases in foundations.
* **Installs policy (modules):** Any technology install must (a) avoid base libs, (b) include an explicit version/range, and (c) fail fast if it would force a base upgrade. (Document that upgrades require an explicit, reviewed deviation.)
* **Self-report contract:** Every cell prints one JSON object: `{"cell_id", "status": "pass|fail", "checks": {...}}`. Visual cells must assert **non-empty** plots.
* **Offset mapping semantics:** 0.5 uses **left-nearest** `orig_to_norm`. Note the invariant: `len(norm_to_orig) == len(text)` and `orig_to_norm[j] ≤ orig_to_norm[j+1]`. Call out that grapheme clusters (emoji, ZWJ) are normalized via NFKC and may change display width.
* **Sentence splitter limits:** 0.6’s regex reduces false splits on initials but will still miss edge cases (e.g., abbreviations lists, quotes, multilingual). Future modules may replace it with a tokenizer (e.g., spaCy/BLingFire) when that technology module is in scope.
* **Paths & portability:** `LSA_BASE_DIR` overrides default `/content`. All artifacts live under `lsa_outputs/`. No raw corpus persists beyond what’s needed for the module.
* **Hygiene checks:** Notebook must restart & **Run all** clean on a fresh Colab VM. No cell should depend on side effects outside its declared inputs/outputs.
* **Acceptance checklist (Module 0):**

  1. 0.1 installs nothing on a typical Colab runtime;
  2. 0.2 prints versions + seed + threads;
  3. 0.5/0.6 smoke tests pass;
  4. 0.7 renders a non-empty plot;
  5. 0.4 can write `module_status.json` without error.

If you want, I can also add a short **“pip guard note”**: all module `%pip` cells must include explicit versions and must not reference base packages; any deviation is a blocker until reviewed.
