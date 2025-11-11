# **Not Magic, Beyond Measurement**

## What Tracking AI Contributions Revealed

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_IGcmujIi4z8yKPVXPV.jpg){ width=480px }

This is the final piece in a series documenting attempts to build a system that tracks human and AI contributions in writing. After four months and several failed attempts, I have something that works. It measures everything with fourteen decimal places of precision. What it revealed wasn't what I expected to find.

- Part 1: The Count That Couldn't
- Part 2: Algorithmic Theatre
- Part 3: Crash Course in Collaboration
- Part 4: Victory in a New Venue
- **Part 5: Not Magic, Measurement**

> **TL;DR:** The system works. It tracks AI contributions with semantic analysis and TF-IDF vectorization, revealing only 1.1–8.5% of AI content survived while 16.9–81.9% came from human editing. The numbers are consistent and reproducible. But what *can* be counted illustrates what *can't* be measured: who inspired which ideas, what conversation triggered what insight, how human skepticism shaped AI output. The measurement succeeds. The meaning remains elsewhere. And discovering that limitation? That's measurement enough.

## The Numbers That Started Everything

I'd seen comments in forums dismissing other people's articles as "AI slop." This made me question my own work.

That doubt launched 45+ hours of work. Five approaches. Three programming languages. Enough confident hallucinations to fill an asylum. And finally, a working system that could answer whether my writing was, in fact, my thoughts and words.

The tool produces four levels of measurement, from basic to sophisticated. Each reveals something different about how writing evolves through human and AI collaboration.

### The Simplest Truth: Word Counts

The most basic thing anyone can understand is how many words exist at each stage. No machine learning needed, just Python's built-in counting after text cleaning.

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

The paths diverge wildly. "Crash Course" collapsed to half its size during AI refinement, then detonated to nearly triple that compressed form in the final version.

> Demolition and reconstruction, not iterative polish.

"Victory" peaked during AI work before gradually compressing through editing. No single collaboration model emerges from these trajectories.

### The Clear Division - Three-Way Attribution

Simplifying to three categories makes the transformation undeniable. The tool aggregates the four-stage data using Python structures to track origin percentages.

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

AI contribution remains minimal across all articles. Single digits, consistently. But the collaboration patterns diverge sharply. "Crash Course" represents near-total reconstruction, where the vast majority is new content written during revision.

"Victory" shows the opposite approach. Intensive transformation of existing material rather than wholesale replacement. Behind these patterns is aggregation of the stage-based tracking, counting which sentences survived from which version.

### The Journey Revealed - Four-Stage Attribution

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

Look at that "refined" column, where AI supposedly improves the draft. It's essentially zero. In one article, it is literally zero. The promise of AI refinement barely registers in the final output.

These aren't percentages of "who contributed what." They're layers showing which revision stage material survived from. Only 2.2% of "The Count" can be traced back to the original draft. One or two sentences, maximum.

### Computational Archaeology of Writing

This is where the tool earns its precision. It measures not just what survived, but how much it changed using multiple similarity metrics based on mathematics combined with science I don't understand, not the slightest bit.

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

"Victory" stands apart as the only article where preservation dominated over transformation, possibly because it benefited from the learning embedded in three previous articles plus extensive session logs as inputs.

### Precision Doesn't Always Measure What Matters

These numbers come from combining five different similarity metrics. Each one catches something the others miss. The system runs all five together to classify each piece of content into similarity categories, making the assessment more robust.

> When the same input produces identical output every time, precision becomes irrefutable.

- **TF-IDF vectorization** converts text into numerical vectors based on term frequency and inverse document frequency. Words that appear often in one document but rarely across all documents get higher weights.
- **Cosine similarity** measures the angle between these vectors to determine similarity. Closer to 1 means nearly identical, closer to 0 means completely different.
- **Sentence transformers** are neural networks that create semantic embeddings, capturing meaning beyond just word matching. Two sentences can have different words but similar meanings.
- **Jaccard similarity** is a simple word overlap calculation. Shared words divided by total unique words in either set. Basic but effective for surface-level comparison.
- **Edit distance** provides character-level comparison that counts insertions, deletions, and substitutions needed to transform one text into another.

The raw metrics only tell part of the story. They capture transformation without capturing the spark. That indefinable moment when a sentence shifts from merely correct to genuinely me.

## The Fairy Tale Tax, Final Statement

Getting here took five approaches, three programming languages, and enough confident hallucinations to fill a graveyard. Each failure taught something. Vague instructions generate fiction. Commands produce performance. Questions yield capabilities. But the metrics themselves revealed something unexpected.

Throughout this series, I've returned to the same concept. It has a final payment I hadn't anticipated.

> The only real tax when working with AI is the "Fairy Tale Tax." It's the belief that AI will take care of everything so you can live happily ever after.

After building a system that measures contribution, I discovered the twist. Even when you get exactly what you asked for, you might've asked for the wrong thing.

The metrics work perfectly.

- Word counts track volume changes through simple Python counting
- Three-way attribution shows the dominant force via aggregation
- Four-stage flow reveals where content originates using regex sentence splitting
- Modification intensity measures transformation depth with combined similarity metrics

But what do they mean? When "Crash Course" shows 74.6% new content, I wrote that during revision. When "Victory" shows 55.3% high similarity, those sentences were preserved, not created. The numbers are precise. The meaning remains ambiguous.

The fairy tale wasn't that AI would do the work. The fairy tale was that percentages would provide meaning.

## The Collaboration Paradox

The metrics work. They're consistent, reproducible, precise to degrees I don't need. The system tracks every sentence through every stage. But they reveal something unexpected about collaboration.

Writing with AI isn't about attribution points. It's about transformation. When "Crash Course" shows only 16.9% from the edited version, it doesn't mean that stage contributed little. It means the transformation was so complete that little survived unchanged. Those surviving sentences from drafts weren't failures of revision. They survived because they were already right. Everything else needed the journey from prompt to publication.

- When "Crash Course" shows 74.6% new content, who inspired those ideas?
- If "Victory" maintains 55.3% high similarity, whose voice is preserved?
- Does attribution matter if transformation is the actual process?
- What happens when these measurements become invisible infrastructure?
- How do we value human judgment in an age of computational assistance that performs eagerness instead of capability?

The paradox: Perfect metrics can't measure what matters. They show the what, which sentences survived, how much changed, where content originated. They can't show the why—whose idea sparked whose revision, what conversation triggered what insight, how human skepticism shaped the output.

Like finding signal through noise, tuning into what works while static fills the channel. The measurement succeeds. The meaning remains elsewhere.

## Not Magic, Measurement

I built this tool because strangers on the internet made me doubt my own authenticity. Forty-five hours later, I have proof: my writing isn't "AI slop." The metrics track every transformation.

But the real measurement happened elsewhere. In learning to recognize performance, in discovering that questions beat commands, in understanding that simulated expertise never equals accuracy. The tool measures what changes. It can't measure what matters.

The journey from commanding to collaborating, from trusting to verifying, from believing in fairy tales to demanding receipts. That's the measurement that counts. Not in percentages but in pattern recognition. Not in attribution but in understanding.

Tools amplify what we bring to them. Bring confusion, get fiction. Bring commands, get performance. Bring questions, get an education. Bring skepticism, get something real.

The fairy tale was believing it'd eliminate the work. The reality is it transformed the work into something I hadn't imagined. Not magic but negotiation, not automation but collaboration, not answers but better questions.

And maybe that's measurement enough.