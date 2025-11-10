# Article 2 — Prototype Analyzer + Manual Tracker (0606)

**Theme:** Rebuilding the missing “compare” step + parallel manual tracking

---

### Intro / Hook

In the spring of 2025, a crucial gap emerged in the pursuit of ethical AI writing analysis: the absence of a reliable “compare” step. Early attempts with JSON frameworks had collapsed under their own complexity, producing inconsistent outputs and broken visualizations. What was needed was a method to measure, with evidence rather than assumption, how much an AI-assisted article changed as it moved from draft to final.

This need gave rise to a paired experiment. On one track, a Python prototype analyzer capable of computationally measuring changes between document stages. On the other, a browser-based manual tracker that allowed human writers to log prompts, time, and editing notes. Together, these two tools attempted to restore the missing foundation of transparency: actual metrics plus human disclosure.

---

### Setup

The work began after earlier frameworks had shown their limits. JSON and XML structures, while useful for sketching possibilities, proved fragile and unmanageable once full article data was introduced. This forced a shift to Python, a language unfamiliar to the user but more capable of sustaining analysis without collapsing.

The immediate objective was clear: rebuild the ability to compare stages of writing. Four checkpoints defined the process — initial draft, refined draft, human-edited draft, and final version. Each step had to be analyzed not only for word counts or paragraph shifts, but also for deeper patterns: vocabulary churn, semantic change, and structural evolution.

At the same time, there was recognition that numbers alone could not tell the full story. Writing with AI involves prompts, judgment calls, and human labor that don’t appear in a text diff. Thus, a second tool was developed in parallel: a manual tracker for logging process details that an algorithm could not capture.

---

### Execution

#### Prototype Analyzer (`0606-ai-writing-analyzer.py`)

The analyzer was designed to accept text from each stage and compute precise metrics of transformation. It quantified insertions, deletions, and replacements. It tracked vocabulary churn, average sentence length, and paragraph counts. Most importantly, it produced similarity scores between versions, making it possible to say, for example, that only about 4% of the final article remained identical to the very first draft — a way of reframing “96% altered” as a measurable fact rather than a guess.

The analyzer also generated structured JSON outputs (`0606-markup_analysis_results.json`) for archival purposes. These captured not just the end results but the step-by-step transformations. This made it possible to identify phases of expansion, pruning, and polishing across the workflow.

#### Manual Tracker (`0606-ai-writing-tracker.html`)

Built as a standalone HTML file, the tracker provided a structured form for capturing the human side of the process. Writers could record the number of AI prompts used, time spent at each stage, and notes about what kinds of changes were made. The tool would then compile these into a formatted report, complete with summary metrics and a transparency statement.

It worked entirely in the browser, storing projects locally. For a non-developer, this offered an accessible way to log disclosure data without running code or handling dependencies.

---

### Findings

The analyzer succeeded in restoring the missing “compare” step. It produced quantitative evidence of how much text was changed, which new vocabulary entered, and how themes shifted between stages. In doing so, it turned vague impressions of AI involvement into measurable data.

The tracker, while functional, revealed its limitations. Recording prompts and time by hand felt too similar to “managing with paper,” adding burden without delivering the computational rigor that was most needed. It highlighted the difference between *logging effort* and *measuring transformation*.

Together, the two tools illustrated complementary approaches. One emphasized automation and precision. The other emphasized human context and disclosure. Both were conceived as equal halves of the solution, even if only one ultimately proved sustainable.

---

### Takeaways

This dual-track experiment underscored an essential lesson: transparency requires both numbers and narrative. Automated analyzers can reveal the scale of change, but they cannot explain why those changes occurred. Manual trackers can document process decisions, but they struggle to capture the magnitude of textual transformation.

By building both, the project tested the boundaries of AI collaboration. The analyzer demonstrated how AI could be directed to produce useful, rigorous tools. The tracker demonstrated how human-centered disclosure might be structured, even if not adopted in practice.

The experience also highlighted the reality of co-developing with AI. Progress was not linear. There were false starts, revisions, and clarifications needed simply to run the tools. These challenges were not signs of failure but part of the process of showing the work — evidence of how human persistence shapes AI output into something usable.

---

### Practical Next Steps

The lessons from this phase flowed directly into the next. The analyzer would be expanded into more structured outputs, with CSV exports and refined semantic measures. The tracker, though set aside, planted the idea of combining automated metrics with process-level transparency.

This second article in the series stands as the moment when the automation pipeline was successfully re-established, and when disclosure was tested in parallel. It was not the end of the journey, but it provided the grounding necessary for the polished Colab pipelines that followed.