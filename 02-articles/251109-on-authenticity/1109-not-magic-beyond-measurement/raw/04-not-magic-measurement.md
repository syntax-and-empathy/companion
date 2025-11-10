# **Not Magic, Beyond Measurement**

## What Tracking AI Contributions Revealed

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_IGcmujIi4z8yKPVXPV.jpg){ width=480px }

This is the final piece in a series documenting attempts to build a system that tracks human and AI contributions in writing. After four months and several failed attempts, I have something that works. It measures everything with fourteen decimal places of precision. What it revealed wasn't what I expected to find.

- Part 1: The Count That Couldn't
- Part 2: Algorithmic Theatre
- Part 3: Crash Course in Collaboration
- Part 4: Victory in a New Venue
- **Part 5: Not Magic, Measurement**

> **TL;DR:** The system works. It tracks AI contributions with semantic analysis and TF-IDF vectorization, revealing only 1.1–8.5% of AI content survived while 16.9–81.9% came from human editing. The numbers are consistent and reproducible. But they don't measure the aggravation and frustration in my notes, the full experience documented in the AI session logs, or how the foundation of every article came from my experiences before AI ever saw a word. The meaning was never in the percentages.

I'd seen comments in forums dismissing other people's articles as "AI slop." This made me question my own work. Doubt launched hours of fighting with Python, enduring desperate AI eagerness that can't be measured and countless empty promises. And finally, a working system that can tell me that every word is, in fact, me.

## What Numbers Can Tell You

The tool produces four levels of measurement, from basic to sophisticated. Each reveals something different about how writing evolves through human and AI collaboration.

### How Much Existed

Word counts are the most straightforward metric. They're direct, transparent, and immediately understandable. No machine learning needed, just Python's built-in counting after text cleaning.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_Eqsir2qxTVGmp1qAHb.png){ width=400px }

```bash
WORD COUNT EVOLUTION

Article                  Draft | AI Refined | Edited | Final | Change
────────────────────────────────────────────────────────────────────
The Count That Couldn't  1,368 |        939 |  1,758 | 1,603 | +17.2%
Algorithmic Theatre        876 |      1,463 |  1,609 | 1,238 | +41.3%
Crash Course             1,457 |        776 |    898 | 1,995 | +36.9%
Victory in a New Venue   1,574 |      2,162 |  1,953 | 1,801 | +14.4%
```

The paths diverge wildly. "Crash Course" collapsed to half its size during AI refinement, then expanded to nearly triple that compressed form in the final version.

> Demolition and reconstruction, not iterative polish.

"Victory" peaked during AI work before gradually compressing through editing. No single collaboration model emerges from these trajectories. Each article followed its own path from foundation to publication.

### How Much Was Contributed

Simplifying to three categories makes the transformation clear. The tool aggregates the four-stage data using Python structures to track origin percentages.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_TdDWMHTRXm1I2sx58z.png){ width=400px }

```bash
CONTENT ATTRIBUTION (Three-Way Split)

Article                    AI Generated | Human Edited | New Content
─────────────────────────────────────────────────────────────────────
The Count That Couldn't          5.5%  |       65.9%  |      28.6%
Algorithmic Theatre              4.1%  |       54.1%  |      41.9%
Crash Course                     8.5%  |       16.9%  |      74.6%
Victory in a New Venue           1.1%  |       81.9%  |      17.0%
```

AI contribution to the final article remains minimal across all pieces. Single digits, consistently.

But the collaboration patterns diverge sharply. "Crash Course" represents near-total reconstruction, where most of the final piece is new content written during revision. The 16.9% that survived was already right.

"Victory" shows the opposite approach: intensive transformation of existing material rather than wholesale replacement. Behind these patterns is stage-based tracking, counting which sentences survived from which version. High similarity tells me some sections emerged strong enough to preserve, not that I made minimal effort.

### How Much Changed

Breaking down the stages shows where content originates. The tool tracks sentence-level attribution using regex patterns to split text at sentence boundaries, then tracks each sentence through every revision stage.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_XAGiR4GvjP6Np6RRWq.png){ width=400px }

```bash
FOUR-STAGE CONTENT FLOW

Article                  Draft | Refined | Edited | New Content | Total
────────────────────────────────────────────────────────────────────────
The Count That Couldn't   2.2% |    3.3% |  65.9% |      28.6%  | 100%
Algorithmic Theatre       4.1% |    0.0% |  54.1% |      41.9%  | 100%
Crash Course              7.7% |    0.8% |  16.9% |      74.6%  | 100%
Victory in a New Venue    0.0% |    1.1% |  81.9% |      17.0%  | 100%
```

**Look at that "refined" column, where AI supposedly improves the draft. It's essentially zero. In one article, it is literally zero. The promise of AI refinement barely registers in the final output.**

These aren't percentages of "who contributed what." They're layers showing which revision stage material survived from. Only one or two sentences, maximum, of "The Count" can be traced back to the original draft. That doesn't mean the drafts didn't matter, though. They're the starting point I struggle to create on my own.

### How Concepts Were Transformed

This is where the tool earns its precision. It measures not just what survived, but how much it changed, using multiple similarity metrics based on mathematics I don't understand. I still get the idea of what the numbers actually mean, though.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_sswrye9iM6enzbwwSF.png){ width=400px }

```bash
MODIFICATION INTENSITY

Article                  High Sim | Medium Sim | Low Sim | New Content
─────────────────────────────────────────────────────────────────────
The Count That Couldn't      9.9% |      16.5% |   45.1% |      28.6%
Algorithmic Theatre          5.4% |       8.1% |   44.6% |      41.9%
Crash Course                 0.0% |       3.1% |   22.3% |      74.6%
Victory in a New Venue      55.3% |      12.8% |   14.9% |      17.0%
```

The patterns diverge dramatically. "The Count" shows a steady gradient from preservation through modification to new content, a gradual transformation.

"Victory" stands apart as the only article where preservation dominated over transformation. This might've happened because it benefited from the learning embedded in three previous articles plus extensive session logs as inputs.

## Measurement Isn't Always What Matters

These numbers combine five similarity metrics, each catching what others miss. The system runs all five to classify content into similarity categories, making the assessment more robust.

- **Jaccard similarity** is a simple word overlap calculation. Shared words divided by total unique words in either set. Basic but effective for surface-level comparison.
- **Edit distance** provides character-level comparison that counts insertions, deletions, and substitutions needed to transform one text into another.
- **TF-IDF with Cosine similarity** converts text into numerical vectors based on term frequency and inverse document frequency. Words appearing often in one document but rarely across all documents get higher weights.
- **Sentence transformers** are neural networks that create semantic embeddings, capturing meaning beyond just word matching. Two sentences can have different words but similar meanings.

These metrics document transformation. They can't capture the conversations that shaped the prompts, the skepticism that improved the outputs, or the obsession that drove complete rewrites just to get it right.

## The Fairy Tale Tax, Final Statement

Getting here took five approaches, three programming languages, and enough hallucinations to fill an asylum. Each failure taught something. Vague instructions generate fiction. Commands produce performance. Questions yield capabilities. But the metrics themselves revealed something unexpected.

Throughout this series, I've returned to the same concept. It has a final payment I hadn't anticipated.

> The only real tax when working with AI is the "Fairy Tale Tax." It's the belief that AI will take care of everything so you can live happily ever after.

The numbers tell me that new content was born by prompt or personal edits, that the sections that remained were preserved, not overlooked. The numbers are measures of words in a post, not the trials and tribulations that led to them.

## What the Numbers Can't Count

I wasn't telling AI to write articles and publishing them. I was using AI to help build from foundations that came from my failures, my learning, my work. The measurements proved it wasn't "AI slop." But looking at these numbers now, that concern feels foolish.

The real question was never about percentages. It was whether the final work had meaning beyond assembled paragraphs.

It does. Because the meaning came first. The AI helped give it form.

But the real measurement happened elsewhere. In learning to recognize when AI is giving a performance, in discovering that questions beat commands, in understanding that simulated expertise doesn't equal accuracy. In realizing that precision about what changed matters less than clarity about what I brought to begin with.

Tools amplify what we bring to them. Bring confusion, get fiction. Bring commands, get performance. Bring questions, get an education. Bring your experiences, context, and humanity, and AI can help shape them into something you aspire to create.

The fairy tale was believing it'd eliminate the work. The reality is it transformed the work into something I hadn't imagined. Not by magic but through collaboration.

And maybe that's measurement enough.