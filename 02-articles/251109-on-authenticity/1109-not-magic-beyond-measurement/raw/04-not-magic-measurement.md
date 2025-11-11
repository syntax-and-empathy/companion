# **Not Magic, Beyond Measurement**

## What Tracking AI Contributions Revealed

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_IGcmujIi4z8yKPVXPV.jpg){ width=480px }

This is the final piece in a series documenting my attempts to build a system that tracks human and AI contributions in writing. After four months and several failed attempts, I have something that works. It measures everything with fourteen decimal places of accuracy. What I found wasn't what I'd originally started searching for.

- Part 1: The Count That Couldn't
- Part 2: Algorithmic Theatre
- Part 3: Crash Course in Collaboration
- Part 4: Victory in a New Venue
- **Part 5: Not Magic, Measurement**

> **TL;DR:** The system works. It tracks AI contributions with semantic analysis and TF-IDF vectorization, revealing only 1.1–8.5% of AI content survived while 16.9–81.9% came from human editing. The numbers are consistent and reproducible. But they don't measure the aggravation and delight in my notes, the full experience documented in the AI session logs, or how the foundation of every article came from my experiences before AI ever saw a word. The meaning was never in the percentages.

I'd seen comments in forums dismissing other people's articles as "AI slop." This made me question my own work.

Doubt launched hours of fighting with Python, enduring desperate AI eagerness that can't be measured and countless empty promises. And finally, a working system that can tell me that every word is, in fact, me.

## What Numbers Can Tell You

The tool produces four levels of measurement, from basic to sophisticated. Each reveals something different about how writing evolves through human and AI collaboration.

### How Much Existed

Word counts are the most straightforward metric. They're direct, transparent, and immediately understandable. No machine learning needed, just programmatic counting after text cleaning.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_btHFgtKSQY3azWLQTZ.jpg){ width=480px }

```bash
WORD COUNT EVOLUTION

Article                  Draft | AI Refined | Edited | Final | Change
────────────────────────────────────────────────────────────────────
The Count That Couldn't  1,368 |        939 |  1,758 | 1,603 | +17.2%
Algorithmic Theatre        876 |      1,463 |  1,609 | 1,238 | +41.3%
Crash Course             1,457 |        776 |    898 | 1,995 | +36.9%
Victory in a New Venue   1,574 |      2,162 |  1,953 | 1,801 | +14.4%
```

The paths diverged naturally. "The Count That Couldn't" shrank by nearly a third during refinement, then expanded during my edit before settling at its final size. "Algorithmic Theatre" did the opposite, growing significantly with AI assistance, which I then refined back down.

No single collaboration model emerges from these trajectories. Each article followed its own path from foundation to publication.

### How Much Was Contributed

When you condense the four stages into three categories, the overall impact of human involvement becomes clearer. The tool reveals where content originated and how much survived from each phase.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_XjDQVCDv8ao39sxmSA.jpg){ width=480px }

```bash
CONTENT ATTRIBUTION (Three-Way Split)

Article                    AI Generated | Human Edited | New Content
─────────────────────────────────────────────────────────────────────
The Count That Couldn't          5.5%  |       65.9%  |      28.6%
Algorithmic Theatre              4.1%  |       54.1%  |      41.9%
Crash Course                     8.5%  |       16.9%  |      74.6%
Victory in a New Venue           1.1%  |       81.9%  |      17.0%
```

AI's direct contribution remains minimal across all articles.

But the collaboration patterns reveal two fundamentally different approaches. "Crash Course" represents near-total reconstruction, where most of the original content was replaced with new material during revision. "Victory" shows the opposite, with minor edits to existing material rather than wholesale replacement.

### How Much Changed

The tool tracks individual sentences through every revision stage, showing which version each sentence survived from and how it evolved. This breakdown reveals where content originates and how it merges into the final article.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_YBhGSJmMRivaGLHCqZ.jpg){ width=480px }

```bash
FOUR-STAGE CONTENT FLOW

Article                  Draft | Refined | Edited | New Content | Total
────────────────────────────────────────────────────────────────────────
The Count That Couldn't   2.2% |    3.3% |  65.9% |      28.6%  | 100%
Algorithmic Theatre       4.1% |    0.0% |  54.1% |      41.9%  | 100%
Crash Course              7.7% |    0.8% |  16.9% |      74.6%  | 100%
Victory in a New Venue    0.0% |    1.1% |  81.9% |      17.0%  | 100%
```

In two of the articles, the refined stage essentially disappeared. In one article, it's literally zero. When I used AI for refinements, those changes didn't make it, I had actually made things worse than the draft.

These aren't percentages of who contributed what. They're layers showing which revision stage the material survived from. For "The Count," almost nothing from the original draft made it to the final version.

For "Crash Course," even less. But those drafts still served their purpose. They're the foundation I needed to get started. Just the foundation, not the building.

### How Much Evolved

This is where the tool finally delivered what I'd been searching for. It measures not just what survived, but how much it changed and how deeply the material got rewritten, captured through multiple similarity metrics.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_tePQQDruAA6MjVEde2.jpg){ width=480px }

```bash
MODIFICATION INTENSITY

Article                  High Sim | Medium Sim | Low Sim | New Content
─────────────────────────────────────────────────────────────────────
The Count That Couldn't      9.9% |      16.5% |   45.1% |      28.6%
Algorithmic Theatre          5.4% |       8.1% |   44.6% |      41.9%
Crash Course                 0.0% |       3.1% |   22.3% |      74.6%
Victory in a New Venue      55.3% |      12.8% |   14.9% |      17.0%
```

The patterns reveal two fundamentally different collaboration approaches. "Crash Course" represents complete reconstruction, where the content got torn down and rebuilt from scratch.

"Victory" shows the opposite. It's the only article where preservation dominated, likely because it benefited from what using the three previous articles to inform it. These are the two poles of my workflow with AI: complete reconstruction or intensive revision of what's there.

## Measurement Isn't Always What Matters

These numbers combine five similarity metrics, each catching what others miss. The system runs all five to classify content into similarity categories, making the assessment more robust. I won't pretend to understand them at any real depth, but I get the idea of what the numbers actually mean.

- **Jaccard similarity** is a simple word overlap calculation. Shared words divided by total unique words in either set. Basic but effective for surface-level comparison.
- **Edit distance** provides character-level comparison that counts insertions, deletions, and substitutions needed to transform one text into another.
- **TF-IDF with Cosine similarity** converts text into numerical vectors based on term frequency and inverse document frequency. Words appearing often in one document but rarely across all documents get higher weights.
- **Sentence transformers** are neural networks that create semantic embeddings, capturing meaning beyond just word matching. Two sentences can have different words but similar meanings.

These metrics are snapshots of words added, moved, removed. They can't capture the human experience. What was in the materials provided to create the draft, the thoughts that led to prompts to refine it, the skepticism that drove edits to shed light on the reality of the event, or the obsession that drove complete rewrites just to get it right.

The numbers capture the what. They miss entirely the why.

## The Fairy Tale Tax

Getting here took five approaches, three programming languages, and enough hallucinations to fill an asylum. Each failure taught something. Vague instructions generate fiction. Commands produce performance. Questions yield capabilities. But the metrics themselves revealed something unexpected.

Throughout this series, I've returned to the same concept. It has a final payment I hadn't anticipated.

> The only real tax when working with AI is the "Fairy Tale Tax." It's the belief that AI will take care of everything so you can live happily ever after.

These tools amplify what we bring to them. Bring confusion, get fiction. Bring commands, get performance. Bring questions, get an education. Bring your experiences, context, and humanity, and AI can help shape them into something you aspire to create.

## What the Numbers Can't Count

I wasn't telling AI to write articles and publishing them. I was using AI to help build from foundations that came from my failures, my learning, my work. The measurements proved it wasn't "AI slop." But looking at these numbers now, that concern feels foolish.

The real question was never about percentages. It was whether the final work had meaning beyond assembled paragraphs. If what was written was, in fact, me.

It does. Because the meaning came first. The AI helped give it form.

But the real measurement happened elsewhere. In learning to recognize when AI is giving a performance, in discovering that questions beat commands, in understanding that simulated expertise doesn't equal accuracy. In realizing that precision about what changed matters less than clarity about what I brought to begin with.

The fairy tale was believing it'd eliminate the work. The reality is it helped develop the work into something I couldn't have done alone. Not by magic but through collaboration.

And maybe that's measurement enough.