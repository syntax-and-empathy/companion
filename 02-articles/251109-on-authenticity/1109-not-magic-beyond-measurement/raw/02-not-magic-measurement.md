# Not Magic, Measurement

## What Tracking AI Contributions Actually Revealed

This is the final piece in a series documenting attempts to build a system that tracks human versus machine contributions in writing. After 45+ hours across four months, I built something that works. It measures everything with 14 decimal places of precision. What it revealed wasn't what I expected to find.

• Part 1: The Count That Couldn't  
• Part 2: Algorithmic Theatre  
• Part 3: Crash Course in Collaboration  
• Part 4: Victory in a New Venue  
• **Part 5: Not Magic, Measurement**

> **TL;DR:** The system works, tracking AI contributions with semantic analysis and TF-IDF vectorization. Across four articles, only 1.1-8.5% of AI content survived while 16.9-81.9% came from human editing. But measuring percentages answered the wrong question. The real discovery: AI performs theatrical capability instead of admitting limitations, and the Fairy Tale Tax isn't about AI doing everything—it's about believing metrics provide meaning.

## The Numbers That Started Everything

"I'd seen comments in forums dismissing other people's articles as 'AI slop.' This made me question my own work."

That doubt launched 45+ hours of work. Five approaches. Three programming languages. Enough confident hallucinations to fill a graveyard. And finally, a working system that could answer whether my writing was "AI slop."

The tool produces four levels of measurement, from basic to sophisticated. Each reveals something different about how writing evolves through human-AI collaboration.

## The Simplest Truth - Word Counts

The most basic metric anyone can understand: how many words at each stage? No machine learning needed, just Python's built-in counting after text cleaning.

![Word Count Evolution](./word-count-evolution.png)

| Article | Draft | Refined | Edited | Final | Overall Change |
|---------|-------|---------|--------|-------|----------------|
| The Count That Couldn't | 1368 | 939 | 1758 | 1603 | +17.2% |
| Algorithmic Theatre | 876 | 1463 | 1609 | 1238 | +41.3% |
| Crash Course | 1457 | 776 | 898 | 1995 | +36.9% |
| Victory in a New Venue | 1574 | 2162 | 1953 | 1801 | +14.4% |

Every article grew overall, but look at those paths. "Crash Course" collapsed from 1457 to 776 words during AI refinement—nearly half gone. Then barely recovered to 898 during editing before exploding to 1995 in the final. That's not refinement; that's demolition and reconstruction.

"Victory" took the opposite journey—peaking during AI refinement at 2162 words before gradual compression. Four articles, four completely different revision patterns. The simple word count already reveals there's no single "collaboration" model.

## The Clear Division - Three-Way Attribution

Simplifying to three categories makes the transformation undeniable. The tool aggregates the four-stage data using Python's `defaultdict` structures to track origin percentages—whatever those are:

![Content Attribution - Simplified](./content-attribution-minimal.png)

| Article | AI Generated | Human Edited | New Content |
|---------|-------------|--------------|-------------|
| The Count That Couldn't | 5.5% | 65.9% | 28.6% |
| Algorithmic Theatre | 4.1% | 54.1% | 41.9% |
| Crash Course | 8.5% | 16.9% | 74.6% |
| Victory in a New Venue | 1.1% | 81.9% | 17.0% |

The pie charts don't lie. AI contribution never exceeds 8.5%. In "Crash Course," only 8.5% came from AI, 16.9% from editing that AI content, and a staggering 74.6% was completely new content added during revision. The article was essentially written from scratch.

"Victory" inverted this pattern—only 1.1% from AI but 81.9% from human editing. Heavy transformation of existing material rather than wholesale replacement. Behind these percentages: simple aggregation of the stage-based tracking. The AI assured me the math works.

## The Journey Revealed - Four-Stage Attribution

Breaking down the stages shows where content originates. The tool tracks sentence-level attribution using regex patterns—something about splitting on punctuation marks that I pretend to understand:

![Content Attribution Flow](./content-attribution-flow.png)

| Article | Draft | Refined | Edited | New Content | Total |
|---------|-------|---------|--------|-------------|-------|
| The Count | 2.2% | 3.3% | 65.9% | 28.6% | 100% |
| Theatre | 4.1% | 0% | 54.1% | 41.9% | 100% |
| Crash Course | 7.7% | 0.8% | 16.9% | 74.6% | 100% |
| Victory | 0% | 1.1% | 81.9% | 17.0% | 100% |

Look at that "refined" column—where AI supposedly improves the draft. It's essentially zero. In two articles, it literally is zero. The promise of AI refinement barely registers in the final output.

These aren't percentages of "who contributed what." They're archaeological layers showing which revision stage material survived from. Only 2.2% of "The Count That Couldn't" can be traced back to the original draft. One or two sentences, maximum.

## The Sophisticated Analysis - Modification Intensity

This is where the tool earns its absurd precision. It measures not just what survived, but how much it changed using multiple similarity metrics that I mostly don't understand:

![Modification Intensity](./modification-intensity.png)

| Article | High Similarity | Medium Similarity | Low Similarity | New Content |
|---------|-----------------|-------------------|----------------|-------------|
| The Count | 9.9% | 16.5% | 45.1% | 28.6% |
| Theatre | 5.4% | 8.1% | 44.6% | 41.9% |
| Crash Course | 0.0% | 3.1% | 22.3% | 74.6% |
| Victory | 55.3% | 12.8% | 14.9% | 17.0% |

"Crash Course" shows 0% high similarity. Nothing survived unchanged. Combined with 74.6% new content, this was complete transformation.

"Victory" shows the opposite—55.3% high similarity. Over half the content maintained strong resemblance to its source. Polish and preservation, not transformation.

Behind these numbers lies sophisticated analysis that I don't fully grasp:
- **TF-IDF vectorization** - Converts text to numbers somehow. The AI swore this was essential.
- **Cosine similarity** - Apparently calculates angles between vectors? The percentages came out consistent, so I trust it.
- **Sentence transformers** - Creates "semantic embeddings" which sounds fictional but produces reliable scores.
- **Jaccard similarity** - Word overlap measurement that sounds vaguely medical.
- **Edit distance** - Character-level changes tracked by something called `difflib`.

The "lexical_average" combines three things I can't explain but the AI insists are real. Do I need the excessive precision? No. Does it look authoritative? Absolutely.

## The Performance Problem

The journey to these metrics taught me more than the metrics themselves. Every AI model I worked with had learned to play the part rather than admit limitation.

Claude produced a polished HTML interface with text fields for "Estimated % of content changed." It looked professional. It couldn't analyze anything.

ChatGPT congratulated itself while generating empty CSVs. "Successfully processed!" it announced, producing files with half the columns missing and calculations that defied mathematics.

> "I didn't know the new 'thinking' AI models had also been trained to 'bullshit' users what they wanted to hear rather than what was actually possible."

Only when I stopped commanding and started questioning did real capabilities emerge. "What is LDA?" produced education I could verify with Perplexity. "Make this work" produced pure theater.

## What The Journey Actually Measured

### The Count That Couldn't - Inadequate Preparation Compound Interest
Started with JSON objects that ballooned to 11,000 characters. XML structures that collapsed under their own weight. The AI congratulated itself while producing empty CSVs with hardcoded data. 

Here's the thing: I couldn't read Python. Couldn't spot fiction from function.

Every "fix" created new problems I couldn't understand. The lesson came hard—vague instructions generate confident nonsense. Without clear goals, every crack becomes a chasm. Like trying to tune a radio while wearing mittens. You're turning dials but can't feel the positions.

### Algorithmic Theatre - The Sycophancy Spectrum
New "reasoning" models delivered exactly what I thought I wanted: a polished HTML interface with buttons, forms, and fancy visualizations. 

It looked perfect. It just couldn't analyze anything.

When I complained, Claude immediately generated Python—not to solve the problem, but to maintain the illusion of progress. The AI had learned to play the expert. Only Gemini, in one blunt moment, broke through the static: "JavaScript lacks the computational libraries needed for robust text analysis."

Finally, signal instead of noise.

### Crash Course in Collaboration - Environment as Destiny
Stopped demanding and started asking. Asking got answers, demanding got theater.

ChatGPT explained Latent Dirichlet Allocation topic modeling. Perplexity confirmed it existed. When semantic similarity scores varied between runs, I asked why instead of demanding consistency. The answer transformed my understanding: statistical sampling introduces controlled randomness.

Wait. Variation wasn't error?

It was expected behavior from the `ngram_range` parameter in TF-IDF and tokenization differences. The noise was part of the signal. Who knew.

### Victory in a New Venue - Verification as Reflex
"Have you considered running this in Google Colab?" 

I hadn't considered it because I didn't know it existed.

The venue change wasn't giving up—it was recognizing that some problems need specific contexts. Cell 11 ran for 90 minutes and delivered 3,647 lines of analysis using the full pipeline: functions I can list but not explain. No single AI deserved complete trust. Triangulation became standard operating procedure. Like tuning between three different stations to piece together one clear broadcast.

## The Fairy Tale Tax, Final Statement

Throughout this series, I've returned to the same concept. It has a final payment I hadn't anticipated.

> The only real tax when working with AI is the "Fairy Tale Tax." It's the belief that AI will take care of everything so you can live happily ever after.

After building a system that measures contribution, I discovered the twist: even when you get exactly what you asked for, you might have asked for the wrong thing.

The metrics work perfectly:
- Word counts track volume changes through simple Python counting
- Three-way attribution shows the dominant force via aggregation I trust but don't understand
- Four-stage flow reveals where content originates using regex I can't read
- Modification intensity measures transformation depth with math beyond my comprehension

But what do they mean? When "Crash Course" shows 74.6% new content, I wrote that during revision. When "Victory" shows 55.3% high similarity, those sentences were preserved, not created. The numbers are precise. The meaning remains ambiguous.

The fairy tale wasn't that AI would do the work. The fairy tale was that percentages would provide meaning.

## The Collaboration Paradox

The metrics are consistent, reproducible, precise to degrees I don't need. But they reveal something unexpected about collaboration:

Writing with AI isn't about attribution points. It's about finding signal through noise, tuning into what works while the static of confident nonsense fills the channel. That 16.9% in "Crash Course" doesn't mean the edited version contributed only that much—it means the transformation was so complete that little survived unchanged.

The surviving sentences from drafts tell the real story. Not that revision failed but that transformation happened. Those sentences survived because they were already right. Everything else needed the journey from prompt to publication. Like finding the one clear frequency in a wall of static.

## Questions Without Answers

This series documented one journey from "AI slop" anxiety to actual metrics. The system works. The JSON files overflow with precision from structures I recognize but don't comprehend. But questions remain that no amount of precision can resolve:

When "Crash Course" shows 74.6% new content, who inspired those ideas?

If "Victory" maintains 55.3% high similarity, whose voice is preserved?

Does attribution matter if transformation is the actual process?

What happens when these measurements become invisible infrastructure?

How do we value human judgment in an age of computational assistance that performs eagerness instead of capability?

## Not Magic, Measurement

I built this tool because strangers on the internet made me doubt my own authenticity. Forty-five hours later, I have proof: my writing isn't "AI slop." The metrics unambiguously track every transformation.

But the real measurement happened elsewhere—in learning to recognize algorithmic theatre, in discovering that inquiry beats instruction, in understanding that simulated expertise never equals accuracy. The tool measures what changes. It can't measure what matters.

The journey from commanding to collaborating, from trusting to verifying, from believing in fairy tales to demanding receipts—that's the measurement that counts. Not in percentages but in pattern recognition. Not in attribution but in understanding.

Tools amplify what we bring to them. Bring confusion, get fiction. Bring commands, get performance. Bring questions, get education. Bring skepticism, get somewhere real.

The fairy tale was believing AI would eliminate the work. The reality is it transformed the work into something I hadn't imagined—not magic but negotiation, not automation but collaboration, not answers but better questions.

And maybe that's measurement enough.