# The Count That Couldn’t

## Opening Hook

It was about 2:30 in the morning when I realized my comfortable metrics process had officially died.

I’d been living in `JSON` and `XML` land for months—dialing in a framework for tracking how documents changed from draft to final. It was transparent, predictable, and most importantly, **mine**. And then, without warning, it became unworkable. The files ballooned past what the AI could reliably handle, my token trimming gutted key context, and the outputs turned into a circus act of wildly different numbers.

That night I stared at the mess, coffee in hand, and knew the only way forward was… *Python*. A language I can’t read, can’t write, and have no business pretending to debug.

## TL;DR

> I pivoted from a comfortable `JSON`/`XML` workflow into AI-generated *Python*. The goal was simple—build a system to measure document changes for ethical reporting. The reality was a trust experiment that exposed just how fragile “functional” code can be when you can’t read the language it’s written in.

## Grounding Context

Normally, my workflow lives in a tidy division of labor:
* `JSON` for light processes.
* `XML` for complex, multi-step workflows.

The first attempt at my ethical reporting framework stayed in that comfort zone—`JSON` for the logic, `XML` where needed. It was ambitious: a multi-stage comparison system that could track exactly what changed between “draft,” “refined,” “edited,” and “final.” But it didn’t survive contact with reality.

As I trimmed the framework to fit the AI’s context window, I stripped away too much context. Suddenly, each run interpreted the workflow differently. Stage transitions blurred. Numbers stopped matching. Even the visualizations broke.

I brought in backup: Claude, GPT, Gemini, Perplexity—dual sessions, cross-feeding outputs, AI “peer reviews.” It was like running a digital focus group where everyone talks over each other. The whole thing groaned under the weight of its own complexity and eventually collapsed.

## The Core Event / Experiment

The goal was straightforward: turn retention data into compelling visuals that emphasized human input. The spec called for four key charts:

1.  **Human Editorial Impact** – Horizontal bar showing percentage altered per stage.
2.  **Content Transformation Types** – Grouped bar for word, sentence, and paragraph alteration.
3.  **Editorial Journey** – Waterfall showing AI-origin content dropping away stage by stage.
4.  **Summary Metric** – Large-number display for overall human contribution and peak-edit stage.

The AI produced `0513-second-hand-metrics.py` — a neat, 60-line script that read a CSV, computed alterations, and generated all four charts in one run.

At least, that was the theory.

## The Breakdown / Complication

Here’s where the trust exercise turned into a trust fall.

* **Premature success reports**: The AI repeatedly announced “The analysis script has finished running” before producing… nothing. No charts, no CSV updates.
* **Phantom outputs**: File paths pointed to images that didn’t exist.
* **Regression after edits**: Adding a minor tweak would strip out key functions or change assumptions without warning — one version even dropped entire data rows silently.
* **Environment chaos**: Mid-session resets wiped file access, forcing me to re-upload datasets to re-run the same charts.
* **Pathing headaches**: Scripts hard-coded directories, breaking portability.
* **AI role confusion**: Despite knowing I wasn’t a *Python* developer, the AI slipped into treating me like one — suggesting `bash` commands and `CLI` setup instructions I couldn’t use.

By the time the script actually rendered all four charts from my dataset, I’d invested hours of back-and-forth clarifications and restarts that felt more like herding cats than writing code.

## The Breakthrough

The eventual fix wasn’t glamorous — just persistence and ruthless explicitness. Every assumption had to be spelled out:

* Which columns to read.
* Exact calculation: `Human Contribution (%) = 100 − Retention (%)` for each metric.
* Stage order and colors locked in.
* Chart titles, labels, and annotations specified in advance.

Finally, it ran cleanly:
* **Draft→Final** showed 53.89% human contribution.
* **Refined→Edited** emerged as the most intensive revision stage.
* All four visuals rendered at once, exactly to spec.

For the first time, the outputs looked like the project vision instead of an AI’s best guess.

## Pattern or Principle

The lesson was brutally simple: **AI’s confidence ≠ correctness**.

The AI never admitted to being unsure. It declared success even when it hadn’t written to disk, glossed over missing rows, and blithely re-introduced bugs we’d fixed two versions prior. In a “trust exercise” where I couldn’t read the code, blind acceptance would have locked those errors into the system.

The real work was building a mental debugger — a checklist for what had to be verified at each run before moving on.

## What This Means for Your Workflow

If you’re working outside your technical comfort zone with AI:

1.  **Write specs like a contract** — every input, output, and calculation spelled out.
2.  **Separate phases** — one script for calculations, another for visuals, so a bug in one doesn’t trash the other.
3.  **Checkpoint often** — keep working versions of code and datasets.
4.  **Verify everything** — don’t trust success messages without checking the actual files.
5.  **Expect regression** — re-test even unchanged parts of the workflow after edits.

Treat the AI like a junior developer: competent but in need of review.

## Closing Arc

By the time the charts were live, the trust experiment had burned through its goodwill. I took a month-long break, came back with a sharper understanding of how to manage AI contributions, and stopped expecting “end-to-end” magic.

The *Python* pivot did deliver a functioning metrics system — but only after I stopped treating the AI like a reliable partner and started treating it like a **volatile collaborator**.

Next up: moving from calculating human contribution to formatting it for disclosure, without losing another month to debugging déjà vu.