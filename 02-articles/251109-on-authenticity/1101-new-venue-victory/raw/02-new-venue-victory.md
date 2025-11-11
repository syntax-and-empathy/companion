# Victory in a New Venue

## When the Pipeline Finally Held

This is the fourth in a series documenting attempts to build a system that tracks human versus machine contributions in writing. After learning to collaborate rather than command, I discovered the missing piece wasn't better prompts or smarter models, it was letting the AI the right environment to actually execute what it promised.

> **TL;DR:** Moving to Google Colab transformed scattered scripts into a working pipeline. Testing on the [markup-languages](https://open.substack.com/pub/syntaxandempathy/p/markup-languages-ai-prompts?r=2c384i\&utm_campaign=post\&utm_medium=web\&showWelcomeOnShare=true) article revealed the truth: only 43% similarity remained from draft to final, with just 2.4% of original sentences surviving intact. The AI accelerated expansion, but human editing shaped the actual content. For the first time, the system didn't collapse under the weight of real data.

## Fourth Time's the Infrastructure

The Colab notebook opened with a header: "AI-Human Collaboration Transparency Tracker - Fourth Iteration." Below it, Claude had documented the context: "After three previous failed approaches (JSON/XML evolution, HTML interface, and initial Python attempts)." By June 2025, those words carried weight. Months of phantom functions and empty output folders. Hours spent debugging problems that shouldn't have existed in the first place.

A Saturday in June. Second newsletter deadline a week and a half out. I'd learned enough from previous disasters to know what Colab even *was*, let alone why breaking things into steps mattered. Earlier attempts had me cutting and pasting Python into chat windows, trying to execute scripts in GitHub that were supposed to check folders and provide outputs "at the appropriate times." That one bombed horribly.

But something felt different. Claude laid out the rationale immediately: "This approach allows us to verify each stage works before proceeding, isolate issues to specific stages, make the code maintainable and understandable." The infrastructure was cleaner. The approach more systematic. The checkpoints designed to catch problems before they cascaded. I'd specified the environment, specified I wanted it in steps, gave it fairly detailed instructions. All new approaches.

I'd been here before—confident this version would finally work, only to watch it collapse. The test would be unforgiving: four complete versions of a live article, from AI-generated draft through human editing to final publication. No toy examples. No simplified test cases. Just the messy reality of mixed-authorship content creation.

## Testing on Markup Languages

The test subject was an article about markup languages in AI prompting. Four versions represented the complete journey: initial AI draft, refined expansion, edited contraction, and final polish.

Two hours of active collaboration. Maybe two and a half. Claude loaded the libraries, mounted the drive, and loaded several functions, each one twice as large as any complete Python script I'd gotten before. Each was just for counting content accurately. That much rigor going into understanding what a sentence actually is, what a word actually is.

When Claude announced **"✅ STAGE 1 COMPLETE: Data loaded successfully"** followed by **"✅ STAGE 2 COMPLETE: Text preprocessing finished"** and then the processing summary, the moment felt surreal after months of failures.

| Version | Word Count | Sentence Count | Character Count | Change from Previous |
| ------- | ---------- | -------------- | --------------- | -------------------- |
| Draft   | 863        | 47             | 5,535           | Baseline             |
| Refined | 1,128      | 60             | 7,213           | +31% expansion       |
| Edited  | 1,066      | 45             | 7,479           | -5% with restructure |
| Final   | 892        | 42             | 6,258           | -16% final trim      |

The familiar pattern: rapid AI-driven expansion followed by systematic human editing that restored focus. What started at 863 words grew to over 1,100, then contracted back to 892, leaner than the original but fundamentally transformed.

## Seven Checkpoints to Avoid Seven Disasters

Earlier attempts had tried to do everything at once. Analyze text, create visualizations, manage data, and explain results in a single sprawling framework. Each additional feature created new failure points until the whole system collapsed.

This version split everything into seven distinct stages.

**Stage 1: Data Ingestion & Validation** - All four article versions present and properly formatted. No phantom files, no missing sections, no silent corruption.

**Stage 2: Preprocessing** - Cleaning Markdown and segmenting text into sentences and paragraphs. Mundane work, but critical for accurate measurement. There were like 15 lines of regex patterns in there. One earlier attempt had stumbled on regex and never recovered. **Stage 2 threw** **cannot import name 'LatentDirichletAllocation' from 'sklearn.decomposition'** **immediately.** One of those errors that stops everything. Claude pivoted. "I'll use TF-IDF and cosine similarity as our primary semantic measure. More reliable than LDA for this use case." Decision made, we moved forward.

**Stage 3: Similarity Analysis** - Lexical measures like Jaccard similarity and edit distance. Semantic analysis using TF-IDF cosine similarity and SBERT embeddings. Multiple approaches to triangulate on truth.

**Stage 4: Attribution Mapping** - Tracing each final sentence back through the revision process. Which originated in the draft? Which emerged during editing? Which appeared only in final polish?

**Stage 5: Metrics Generation** - Converting similarity scores into human-readable percentages and modification categories. Raw computation becoming insight.

**Stage 6: Visualization Preparation** - Structuring data for flow diagrams, heatmaps, and interactive charts. Making patterns visible at a glance.

**Stage 7: Archival Export** - Saving everything to timestamped JSON files that could survive session resets and enable future analysis.

Each stage ended with Claude asking permission to proceed. "Let me know when you're ready for Stage 3." "Please confirm the files have been created successfully." The checkpoints weren't automatic. They demanded explicit human verification before moving forward. Maybe an error per stage: the LDA import failure, sentence_transformers compatibility issues, file path corrections when Colab duplicated folders. But each was caught, assessed, and resolved before it could cascade.

## The Numbers That Told the Truth

The pipeline calculated cross-version similarities, and the extent of transformation became quantifiable for the first time.

**Draft to Final Overall Similarity: 43.1%**

Less than half the original content survived the complete revision process. The granular sentence-level analysis revealed even more dramatic patterns. "Attribution mapping complete." I didn't even know what attribution mapping was, but there it was.

**Content origins breakdown:**

- **Content origins breakdown:**
  - 85.7% traced to the edited stage
  - 2.4% retained high similarity to the original draft
  - 11.9% was entirely new content

**That 2.4% stopped me.**

**The draft had 47 sentences. I'd consciously changed 36: rewrites, restructures, complete overhauls. That meant 11 sentences should have stayed relatively unchanged. Eleven sentences I'd left alone.**

**The data disagreed. Only one sentence from the original 47 remained recognizably similar in the final version.**

**One sentence. Not eleven.**

**The other ten sentences I thought I'd left untouched? The revision process had transformed them anyway. Paragraph reordering changed their context. Surrounding edits shifted their meaning. What I experienced as "leaving sentences alone" the algorithm saw as transformation. I hadn't just rewritten 36 sentences. I'd transformed 46 of them, often without conscious intent.**

**For the first time, I had proof of what I'd suspected but couldn't measure. Revision changes everything, even the parts you think you're preserving.**

The modification intensity analysis reinforced this:

- High similarity: 7.1%
- Medium similarity: 40.5%
- Low similarity: 40.5%
- New content: 11.9%

These numbers demolished any notion that AI drafts simply needed light editing. The vast majority of the final article bore little resemblance to the original AI output. More importantly, they demolished the notion that human editing could be tracked by counting deliberate changes. The transformation ran deeper than conscious decisions.

## Why Colab Changed Everything

Context persistence across cells meant variables and functions didn't vanish mid-analysis. The notebook environment provided transparent documentation of every step. Most importantly, checkpoints could be inspected and verified before proceeding.

Once complete, the full notebook took about three hours to run from start to finish. Time to do something else while the pipeline processed everything. The system processed 863 words in the draft through to 892 words in the final version, tracking every transformation along the way.

Colab still doubled up on folders sometimes. A minor annoyance I could probably fix myself. But compared to empty CSVs and phantom functions, a duplicate folder was nothing. Manual verification at each stage was overhead, but it was productive overhead that increased confidence rather than consuming it through endless debugging.

## Visualizations That Actually Worked

The structured data pipeline enabled visualization capabilities that emerged naturally. Flow diagrams showed content evolution. Heatmaps revealed modification patterns. The system generated an interactive HTML Sankey diagram that made attribution analysis clear.

Unlike previous attempts where Claude's impressive-looking charts broke during export, this approach separated data generation from visualization. The JSON exports could feed any charting system. When one approach failed, alternatives worked with the same data.

At 11 PM, I opened the JSON output in Sublime. Fourteen decimals on almost everything, far more precision than necessary, but it was there. The archive didn't need to be human-friendly. It needed to be AI-friendly, structured data I could feed into future analysis without re-running the entire pipeline. I could literally read through and check if it'd screwed up sentence parsing. Found mistakes immediately: "Your diagram or approach, structured verse" counted the period but missed the second sentence. The character count was off. But I didn't care about perfect character counts anymore. The semantic analysis was there.

The system even included trend analysis capabilities, though these remained untested due to limited historical data. Comments in the code indicated where future articles would plug in: "when you have more articles, so I can comment these and put them in the cells."

## Transparency That Could Be Audited

Previous attempts at disclosure had relied on manual estimates and rough impressions. The Colab system produced verifiable metrics that could be audited, replicated, and trusted.

The markup languages analysis became the first article with complete algorithmic transparency:

```json
{
  "article_name": "markup-languages",
  "word_progression": {
    "draft": 863,
    "final": 892,
    "change_percentage": 3.4
  },
  "content_retention": {
    "overall_similarity": 43.1,
    "content_origins": {
      "edited": 85.7,
      "draft": 2.4
    }
  }
}

```

The difference between "AI-assisted writing" and "43.1% similarity retention with 2.4% sentence survival rate" wasn't just precision. It was the foundation for informed evaluation of mixed-authorship content.

## The Fairy Tale Tax

I'd expected the pipeline to just work once it was properly designed. Clean infrastructure, systematic checkpoints, reproducible outputs. What could go wrong?

Everything still required constant human partnership. When library imports failed, human intervention guided recovery. When file paths duplicated folders, human oversight corrected the mistakes. Regex patterns demanded testing against edge cases. Is it perfect? Probably not.

> The only real tax when working with AI is the “Fairy Tale Tax.” It’s the belief that AI will take care of everything so you can live happily ever after.

The "AI Tax" evolved but didn't disappear. Instead of compound interest on failed systems, it became the systematic cost of maintaining reliable collaboration between humans and AI.

Even in success, oversight, verification, and quality assurance weren't obstacles to efficiency. They were the mechanisms that made AI assistance trustworthy.

## What Actually Made This Work

**Modular design prevents cascading failures.**  
Seven distinct stages, each with its own checkpoint. When something broke, the failure was isolated and fixable rather than poisoning the entire system. Earlier attempts tried to do everything at once and collapsed under their own weight.

**Checkpoints demand human judgment.**  
The system could measure similarity and track attribution, but interpreting those measurements demanded contextual understanding that algorithms couldn't provide. Every checkpoint required editorial verification that felt collaborative rather than like failed automation.

**Infrastructure enables rather than guarantees.**  
The successful Colab deployment opened pathways that hadn't existed before. Multiple articles could be analyzed using identical methodology. Automation became feasible for the first time. But the infrastructure was designed for iteration rather than perfection.

**Reproducibility builds trust.**  
The archival JSON format meant historical data could be aggregated and compared without re-running expensive computational steps. Transparency shifted from aspiration to measurement, from manual estimation to algorithmic verification. That distinction mattered. Showing your work once is documentation. Building infrastructure that makes showing your work repeatable is transparency.

## The Foundation That Almost Wasn't

The Colab pipeline established the computational backbone for systematic transparency. Its modular design, checkpoint workflow, and archival outputs created infrastructure that could support analysis at scale. The markup languages test proved the concept could handle real-world complexity without collapsing.

But success revealed new challenges. Single-article analysis was just the beginning. Trend identification across multiple pieces, automated disclosure generation, and reader-friendly visualization remained to be solved. I'd lost a cumulative two days on the first several attempts with mediocre results I didn't bother using. This time felt different.

The victory wasn't just technical. I'd struggled so many times to get this working. For the first time, the question "How much of this was AI versus human?" had a precise, reproducible answer. The foundation was solid.

What came next would test whether precision could actually reshape how we think about authorship, accountability, and the future of human-AI collaboration in creative work.

So that's a thing now, for real.

That's where the final article begins.