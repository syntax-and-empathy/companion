# Article 1 — Early Attempts: Building a Transparent Testing Tool with AI

## Intro / Hook

The starting point was simple: I wanted a way to see how much of my writing came from me and how much came from AI. A tool that could measure that balance across drafts felt like it could bring some much-needed transparency to design work.

That transparency matters. Large Language Models (LLMs) aren’t magic boxes — they generate text by probability, not judgment. Without clear reporting, it’s easy to lose sight of where ideas are really coming from, or to assume AI is more reliable than it is. Sharing not just successes but also failures is part of keeping the process human-centered and accountable.

This article is the first in a series tracing my different attempts to build such a tool. Each attempt taught me something new, often the hard way. This first one didn’t deliver a reliable system — far from it — but it showed me why starting with a clear plan matters, and how working with AI often means spending just as much time fixing as creating.

---

## Setup

The first idea was to use JSON and XML as the backbone for a reporting system. On paper, it made sense: structure the different stages of an article (draft, refine, edit, final), log the changes, and then generate metrics from that structure.

In practice, it fell apart quickly. The JSON framework ballooned to more than 11,000 characters, broke under its own weight, and produced inconsistent outputs. Stripping things down for efficiency only made it worse, cutting out context that the AI actually needed. Once that happened, the same piece of writing might come back with three different interpretations in three different sessions.

It was a classic example of trying to do too much, too soon. Instead of starting with a clear plan, I asked the AI to juggle structure, logic, and visualization all at once. The result was chaos: broken visualizations, scrambled data, and hours lost chasing formatting issues.

Around 2:30 a.m., after watching the framework fail again with real article data, I scrapped the whole thing. If JSON and XML couldn’t scale, what could? Claude suggested Python. I had never written a line of Python, but at that point the only way forward was to treat the whole thing as a trust exercise: let the AI write the code, and see if I could manage it.

The plan was pared back to two simple scripts:

1. **Generate script** — process the article stages and output counts and retention metrics.
2. **Compare script** — analyze those outputs across stages.

On paper, the idea was clean. In practice, it would test not just the limits of the AI, but the limits of my patience.

---

## Execution

The **generate script** was the first surprise. It came together quickly, ran cleanly, and actually worked. For someone who couldn’t read Python, watching it chew through three full articles and spit out word counts, character counts, and retention metrics felt like a rare win.

The **compare script** was another story.

* **Version 2** seemed fine — stable, doing what it was supposed to do.
* **Version 3** suddenly shifted the structure of the output for no clear reason.
* **Version 4** baked in assumptions that made the script less flexible.
* **Version 5** looked polished but was quietly broken: it dropped rows from the data, pointed to files that didn’t exist, and somehow lost two core functions along the way.

The breaking point came when I renamed the file. Somehow, that coincided with the script forgetting its own functions. From then on, the AI insisted those functions lived in a separate module, which of course didn’t exist. Every “fix” just created another loop of missing pieces and phantom imports.

The AI’s confidence didn’t help. It routinely reported successful runs with nothing in the output folder. Other times, it gave me CSVs that were half empty. At one point, it even included non-metric columns in an average calculation, which pushed totals over 100%.

Visualization became its own mini-drama. Claude could generate charts that looked sharp in its own interface — formatted bars, clean comparisons, even layouts that suggested polish. But the moment I tried to export them, they broke. Markdown fell apart, column widths collapsed, and some files threw outright errors. ChatGPT’s charts, by contrast, looked plain and sometimes oversimplified, but they exported cleanly and worked across platforms. Pretty but broken turned out to be worse than plain but reliable.

All told, I spent around **45 hours** just wrestling this one attempt into something that ran without collapsing. By the end, the generate script worked. The compare script barely did.

---

## Findings

By the end of this first attempt, I had one script that worked reliably and another that barely held together. As a tool, it wasn’t enough to meet the goal of transparent reporting. But as a lesson, it was invaluable.

Several themes stood out:

* **AI’s confidence isn’t the same as correctness.** One version claimed success while leaving the output folder completely empty.
* **Fragile context led to drift.** Moving code between sessions made the AI forget functions it had written earlier and invent modules that didn’t exist.
* **Workflow matters.** Asking for structure, metrics, and charts all at once gave me nothing usable in any of them.
* **Over-reliance exposed blind spots.** Trusting the AI without double-checking led to missing rows, broken charts, and inflated numbers.
* **The hidden human cost.** Roughly 45 hours went into prodding, correcting, and re-running. Efficiency turned out to be a mirage — the oversight cost was higher than expected.

These patterns mirror what often happens in design workflows when AI is used without a clear plan. Vague prompts, skipped checks, or treating the AI like an independent expert almost always lead to poor outcomes. What this attempt showed, in real time, was that AI isn’t a replacement for human judgment — it only works when guided, checked, and kept in line.

---

## Takeaways

This first attempt didn’t give me the reliable reporting system I wanted. But it did make one thing clear: you can’t treat AI as a shortcut. Without a plan, the work multiplies instead of shrinking.

The real contribution wasn’t the code the AI produced — it was the hours I spent steering, debugging, and making sense of its failures. That’s as much “human work” as editing text, and it deserves to be acknowledged when we talk about what AI really costs.

Transparency was my original goal, and it still matters. Not just transparency about the results of using AI, but about the process itself: the errors, the wasted time, the parts where human judgment had to step in. Documenting those realities isn’t a side task — it’s what makes the work accountable.

Later attempts, which this series will cover, moved closer to something that worked. But this first one was the necessary lesson: AI is not magic, and it doesn’t replace design thinking. It’s a negotiation, and it only works when you meet it with structure, oversight, and honesty about where it falls short.

---

## Practical Next Steps

This wasn’t the end of the project — just the beginning. The generate script proved that the core idea could work. The compare script showed how quickly things could break without clear structure. Both left me convinced that the only way forward was to rebuild with tighter boundaries and a more systematic plan.

The next attempt took that lesson to heart. It reintroduced comparison in a cleaner form and paired it with a separate manual tracker — two tracks instead of one overloaded system. That experiment is the focus of the next article in this series.

This first attempt made one lesson impossible to ignore: comparison is essential, but the way I built it collapsed under its own weight. JSON and XML gave me numbers, but not trust. The next step was to rebuild that missing compare stage from scratch — this time with a prototype analyzer that could crunch the changes directly, and a manual tracker that captured the human effort alongside them. That two-track experiment is where the work began to feel less like improvisation and more like a system.