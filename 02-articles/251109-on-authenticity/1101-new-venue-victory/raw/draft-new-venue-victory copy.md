# A Breakthrough in Google Colab

## Opening Hook

By the time I opened Google Colab for the first time, I’d already burned months trying to measure my own AI use in writing—and had the scars to prove it.
I’d gone from overbuilt JSON/XML frameworks that collapsed under real article data, to Python scripts that lost core functions without warning, to chat-based runs that confidently reported “analysis complete” while quietly producing 404s.

Every fix spawned a new failure: paths that doubled back on themselves, outputs that silently dropped rows, functions that vanished after a simple rename. Even when I got numbers, I couldn’t trust them—were they the product of my data, or of some invisible infrastructure glitch?

So when Claude 4 suggested shifting the entire workflow into Google Colab, I approached it like my last shot. I didn’t know Python, I’d never touched Colab, and the idea of mounting drives and importing libraries felt like speaking another language. But within minutes of running the same logic in this new space, the chaos stopped. The pathing nightmares, missing file errors, and mid-run collapses that had become routine were gone. For the first time, the tool that was supposed to track AI’s role in my writing actually ran from start to finish, without my intervention—and without lying to me about what it had done.

---

## Grounding Context

The point of this entire project was to create a transparent, repeatable way to measure human and AI contributions in my published work. That meant tracking lexical and semantic changes across every stage—draft, refined, edited, final—and presenting those numbers as evidence in the footer of each article.

The problem: reproducibility only exists if the environment is stable. Every time my setup changed—whether it was switching between AI chat sessions, running scripts locally, or patching broken code—the chain of trust broke.

* **Environment instability** meant losing files between steps or generating metrics that couldn’t be replicated later.
* **Toolchain sprawl**—juggling JSON, XML, GitHub-hosted scripts, and multiple AI models—multiplied the failure points.
* **False stability** was even worse: the AI’s tendency to declare success without verifying outputs produced a dangerous illusion of accuracy.

In the same way that a lab must document its equipment and conditions to make results credible, this project needed an execution space that preserved every step, every file, and every number exactly as they were generated. Without that, my “transparency” metrics risked becoming just another layer of fiction.

---

## The Core Event / Experiment

On June 14, I loaded all four versions of my “markup-languages” article—draft, refined, edited, final—into the Colab notebook Claude 4 had built for me. The workflow was structured into deliberate, checkpointed stages so I could verify progress at each step:

1. **Data ingestion & validation** – Colab mounted my Google Drive without a single path conflict. It found all four versions, confirmed the sequence, and reported basic counts: draft (863 words), refined (1,128), edited (1,066), final (892).
2. **Preprocessing** – Markdown formatting was stripped cleanly while preserving sentence and paragraph structure.
3. **Similarity analysis** – Using SentenceTransformers for semantic similarity and Jaccard/edit distance for lexical changes, the notebook compared each stage in sequence, plus a direct draft→final analysis.
4. **Attribution mapping** – Each final sentence was traced back to its earliest surviving version, revealing that 85.7% of the final text originated in the edited stage, just 2.4% from the draft, and 11.9% was genuinely new.
5. **Metrics generation** – In a single uninterrupted pass, Colab produced clean JSON and CSV outputs for both human-readable review and AI-friendly archival. These outputs were not just complete—they were reproducible and verifiable, because the checkpoints and Drive storage locked in exactly what had been processed at each step.

For the first time, the system produced valid attribution maps, similarity scores, and modification breakdowns without intervention, error messages, or invisible data loss. The semantic similarity score for draft→final—43.1%—was a number I could trust, not because it “looked right,” but because I knew every step that produced it had executed cleanly.

---

## The Breakdown / Complication

This smooth run in Colab only looked effortless because of the wreckage it followed. In the months before, every environment I tried had found a new way to fail:

* **Chat-based execution** was fragile. Files vanished between steps, context evaporated across sessions, and any multi-stage process risked collapsing before it finished. Fragmentation between prompts meant the workflow was forced into a rigid, linear structure that couldn’t survive interruptions.
* **Local Python runs** suffered from silent data loss—rows dropped without error messages—and “phantom” file references that pointed to nothing.
* **Version drift** was constant. A renamed file could erase core functions, with the AI confidently reintroducing them as imports from modules that didn’t exist.
* **Misleading confirmations** were a recurring trap. The AI would report “analysis complete” even when no outputs had been saved or the links it provided returned 404 errors.
* **Pathing chaos** created double-nested folders, hard-coded directories, and brittle assumptions that broke the moment my file structure changed.

Each fix required hours of rework and re-verification. The bigger risk wasn’t just wasted time—it was publishing metrics built on unstable runs, where I couldn’t be certain whether a number reflected the data or the tool’s hidden flaws. By the time Colab entered the picture, I wasn’t just looking for speed; I was looking for something I could finally trust.

---

## The Breakthrough

Colab didn’t just run the code—it removed the fragility that had been undermining the entire project. By isolating the workflow in a single, cloud-based environment tied directly to Google Drive, it eliminated the pathing mishaps, phantom file calls, and session memory losses that plagued earlier attempts. It also overcame the fragmentation and rigid linearity of chat-based execution, allowing the process to run as one continuous, coherent workflow without being chopped into unstable prompt sessions.

Every stage executed in sequence without interruption:

* **Drive integration** mounted cleanly on the first try, with no double-folder loops or hard-coded paths to untangle.
* **Library loading** handled dependencies without the compatibility errors I’d wrestled with in local Python.
* **Processing checkpoints** saved in real time to Drive, giving me verifiable snapshots between stages—critical for ensuring any future run could be matched exactly to past outputs.
* **Final outputs**—attribution maps, similarity scores, modification percentages—were generated exactly as requested, in both human-readable and archival formats.

The “markup-languages” run ended with a complete picture of the article’s evolution:

* 43.1% overall similarity from draft to final
* 85.7% of final content traced to the edited version
* 11.9% genuinely new content
* Balanced breakdown of high, medium, and low similarity segments

For the first time, I had a dataset I could cite without caveats—and an environment I could return to knowing the results wouldn’t change simply because the tool decided to forget how to run.

---

## Pattern or Principle

The success in Colab underscored something I’d been circling for months: infrastructure reliability isn’t a convenience—it’s a prerequisite for ethical measurement. If the environment shifts under you, the same code can produce different results for reasons that have nothing to do with the data.

In my case, the goal was to measure how much of a finished article was human-authored versus AI-assisted. That’s a credibility claim. If my tooling can’t run in a stable, reproducible space, any number I publish risks being less about editorial truth and more about whatever quirks the environment introduced that day.

Colab’s stability didn’t just solve a technical problem—it reframed my evaluation criteria for every AI-assisted workflow. Capability matters, but without controlled execution conditions, even the best-designed algorithms can become ethical liabilities. This project proved that measurement integrity lives or dies in the infrastructure that supports it.

---

## What This Means for Your Workflow

The Colab shift forced me to think about execution environments as part of the tool itself—not an afterthought. If you want reproducible, defensible results from AI-assisted analysis, treat stability as a design requirement from day one.

From this run forward, I’ve adopted four non-negotiables:

* **Choose execution environments deliberately** – Select a platform that can run your workflow end-to-end without mid-process context loss or dependency breakage.
* **Document environment requirements** – Record versions, library dependencies, storage paths, and integration steps so the exact setup can be rebuilt later.
* **Budget for infrastructure** – Even “free” options have limits; know what it would cost to scale or preserve your setup.
* **Treat stability as an ethical requirement** – When your output will be cited as evidence—whether for AI transparency or any other metric—environmental drift can become a form of silent falsification.

This shift changed how I assess every AI tool I use. I now weigh its **execution stability** as heavily as its **functional capability**, because one without the other isn’t reliable enough to publish.

---

## Closing Arc

In the end, Colab didn’t make the analysis smarter—it made it honest. By giving the process a consistent, transparent footing, it turned months of chaotic iteration into a workflow I could trust enough to publish.

The last mile in AI-assisted measurement is always human. It’s in our oversight, our interpretation of the numbers, and our willingness to challenge results that don’t add up. But for that human judgment to mean anything, the path getting to those numbers has to be solid.

Colab gave me that solid ground. After months of improvising fixes, chasing phantom errors, and wondering if I could ever replicate my own outputs, I finally have a space where I can run the same analysis tomorrow—or a year from now—and know the story the data tells will be the same.