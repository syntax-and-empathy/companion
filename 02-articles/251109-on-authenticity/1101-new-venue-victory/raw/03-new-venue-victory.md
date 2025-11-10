# Victory in a New Venue

## The Notebook That Finally Counted

This is the fourth in a series documenting attempts to build a system that tracks human versus machine contributions in writing. After learning to collaborate rather than command, I discovered the missing piece wasn't better prompts or smarter models, it was letting the AI specify the right environment to actually execute what it promised.

Part 1: The Count That Couldn't

Part 2: Algorithmic Theatre

Part 3: Crash Course in Collaboration

**Part 4: Victory in a New Venue**

Part 5: Not Magic, Negotiation

> **TL;DR:** After five failed attempts across three different languages and enough false starts to fill a graveyard, moving to Google Colab transformed scattered scripts into a working pipeline. Testing on the [markup-languages](https://open.substack.com/pub/syntaxandempathy/p/markup-languages-ai-prompts?r=2c384i\&utm_campaign=post\&utm_medium=web\&showWelcomeOnShare=true) article revealed the truth: only 43% similarity remained from draft to final, with just 2.4% of original sentences surviving intact. The AI accelerated expansion, but human editing shaped the actual content. For the first time, the system didn't collapse under the weight of real data.

## Python's Laboratory

I'd already paid the tuition. Python scripts that produced empty CSVs. HTML interfaces that couldn't actually analyze anything. Semantic similarity scores that changed every time I blinked. Each failure had taught me something, but none had produced what I actually needed.

> Then Claude asked  "Have you considered running this in Google Colab?"

I hadn't considered it because I didn't know it existed. Twenty minutes later, I was staring at what looked like Google Docs had a baby with a Jupyter Notebook.

**[Image of colab cell with code in it]**

### Class In Session

The notebook provided the means to split scripts into stages, something that would have been multiple files, and undoubtedly exponential failures with how I had approached it before. By now, I had learned to request complex or long items in chunks, effectively making the AI in question work in steps rather than spitting out a not-so-comprehensive artifact. 

I watched as Claude transformed from code generator to educator. It wasn't just solving my problem—it was ensuring I understood the solution.

```python
# First, we need permission to access your Drive
from google.colab import auth
auth.authenticate_user()

# This creates our credentials - think of it as your ID badge
import google.auth
creds, _ = google.auth.default()

# Now we can actually talk to Google Drive
# Just like having the right key for the right door
```

Claude produced a script twice the size of anything I had received before for every step, each designed to run in its own cell in the notebook followed by a request for confirmation.

Each function included not just what it did, but why it mattered. After months of black-box solutions, I had a patient tutor explaining every step. This was acknowledging the actual complexity of the problem and helping the dope that showed up with a silly little script understand the purpose of the code.

### And Then, Secret CELL 11

Up to this point every cell ran for less than a minute. Each followed by a brief confirmation upon completion, checkpoints in the process, tidbits for human verification that all was well.

```plaintext
#CELL 10: QUICK EXECUTION FOR EXISTING DATA
```

Prior cells had been monolithic in comparison to what I had been punished with time and time again. Cell 10 was a fraction by comparison, a meager 34 lines of code. Then there was an inconspicuous cell. A mere 3 lines of code.

> No title in CAPS proclaiming its presence. No educational explanation. No warning.

I still don't know Python, but I know you don't have to be able to read or write it to miss hints at it's purpose. I don't know that I would have picked up on it then but I see it plain as day now. 

```plaintext
combined_results, footer_metrics = run_complete_analysis_from_existing(
    article_versions, preprocessor
)
```

The cell ran past a minute, the spinner kept spinning, and nothing happened. And then it whispered "🔍 Analyzing draft → refined, Analyzing 87 vs 61 sentences..." a few minutes later and it whispered a little bit more "🔍 Analyzing refined → edited, Analyzing 61 vs 98 sentences..." After 20 minutes I started cleaning my office.

> I had paid my taxes in full, and this is when the Fairy delivered. An hour and a half later.

When the results appeared the moment was anticlimactic to say the least. The question had been answered in a concise and meaningful way. Results that I failed to copy from the notebook.

For the sake of completeness, I used The Count That Couldn't to provide the following.

```plaintext
📊 Attribution Summary:
  Total sentences in final: 91

  Origin Distribution:
    edited: 60 sentences (65.9%)
    new_in_final: 26 sentences (28.6%)
    refined: 3 sentences (3.3%)
    draft: 2 sentences (2.2%)

  Modification Levels:
    high_similarity: 9 sentences (9.9%)
    medium_similarity: 15 sentences (16.5%)
    low_similarity: 41 sentences (45.1%)
    new_content: 26 sentences (28.6%)

💾 Complete analysis saved: /content/output/the-count-that-coudnt_complete_analysis.json
📊 Footer metrics saved: /content/output/the-count-that-coudnt_footer_metrics.json
```

I was underwhelmed, but there was JSON.

## Fourteen Decimals of Overkill

When I started I had requested data to be used for visualizations, and a subset of the data for visualizations I could add at the end of the articles. The json for the footer looked like the information provided in Colab. Checkpoint data from steps 1 and 2, slightly more satisfying at 86 lines.

> Spoiler, it turns out the article was about using Mermaid AI, not markup.

Then there was the complete dataset. An overwhelming 3647 lines documenting every calculation traceable, every metric visible, every error identifiable. Every sentence from every version of the article with the relevant data.

**Lines 130-141**

```
      "detailed_analysis": [
        {
          "source_index": 0,
          "source_sentence": "Mermaid Mapping: Charting My Content Creation Workflow\n\nHow AI-assisted diagramming illuminated my process and paradoxically made it more human\n\nI set out to create a visual diagram of my content workflow—a seemingly straightforward task",
          "best_match": {
            "index": 0,
            "lexical": 0.17050707785339678,
            "semantic": 0.8208606839179993,
            "combined": 0.49568388088569804,
            "target_sentence": "Mermaid AI: A Tool Trial for Diagramming\n\nTesting the capabilities of AI-assisted diagram creation\n\nTL;DR: When I decided to try Mermaid AI for diagramming my content creation workflow, I was simply curious to see if it would be beneficial"
          }
        },
```

**Lines 743-753**

```
      "detailed_analysis": [
        {
          "source_index": 0,
          "source_sentence": "Mermaid AI: A Tool Trial for Diagramming\n\nTesting the capabilities of AI-assisted diagram creation\n\nTL;DR: When I decided to try Mermaid AI for diagramming my content creation workflow, I was simply curious to see if it would be beneficial",
          "best_match": {
            "index": 0,
            "lexical": 0.4705279444460853,
            "semantic": 0.8537111282348633,
            "combined": 0.6621195363404743,
            "target_sentence": "Mermaid AI: A Tool Trial for Diagramming\n\nTesting the capabilities of AI-assisted diagram creation\n\nTL;DR: What began as a simple test of Mermaid AI for diagramming evolved into a revealing exploration of human-AI collaboration"
          }
```

**Lines 1484-1495**

```
      "detailed_analysis": [
        {
          "source_index": 0,
          "source_sentence": "Mermaid AI: A Tool Trial for Diagramming\n\nTesting the capabilities of AI-assisted diagram creation\n\nTL;DR: What began as a simple test of Mermaid AI for diagramming evolved into a revealing exploration of human-AI collaboration",
          "best_match": {
            "index": 0,
            "lexical": 0.5566687671556464,
            "semantic": 0.9299373626708984,
            "combined": 0.7433030649132724,
            "target_sentence": "Mermaid AI Test Run\n\nTesting the capabilities of AI-assisted diagram creation\n\n## TL;DR:\n\nMy Mermaid AI test became a study of human-AI collaboration"
          }
        },
```

**Lines 2063-2074**

```
    "detailed_analysis": [
      {
        "source_index": 0,
        "source_sentence": "Mermaid Mapping: Charting My Content Creation Workflow\n\nHow AI-assisted diagramming illuminated my process and paradoxically made it more human\n\nI set out to create a visual diagram of my content workflow—a seemingly straightforward task",
        "best_match": {
          "index": 0,
          "lexical": 0.20689374730565077,
          "semantic": 0.6455597877502441,
          "combined": 0.42622676752794747,
          "target_sentence": "Mermaid AI Test Run\n\nTesting the capabilities of AI-assisted diagram creation\n\n## TL;DR:\n\nMy Mermaid AI test became a study of human-AI collaboration"
        }
      },
```

An archive with precision that boarders on parody. A summary of the data, the actual data, an index of the data, and so on. Fourteen decimals on every datapoint; everything necessary for it to be AI-friendly so I can use the numbers again in the future.

I ran the same article through three more times and the results have been consistent. All fourteen decimal points.

> "I think I love you right now," I typed, "Now I need visuals."

Complete, reliable, transparency.

## The Fairy Tale Tax

I came with a Python script that worked, it had a "statistically acceptable level of variance", but it worked. I paid the Fairy Tale Tax several times:

- Blindly following an AI producing code I couldn't read that failed time and time again.
- Expecting AI to deliver what I couldn't really define because I heard what I wanted.
- Repeated frustration that resulted in me giving up more than once.
- And then some.

This time the tax wasn't about believing in magic, that it would simply work out because I was using AI. It was about learning that having a vast amount of information at your fingertips might qualify as intelligent but it doesn't make you smart. Understanding how to into that vast amount of knowledge does.

> The only real tax when working with AI is the “Fairy Tale Tax.” It’s the belief that AI will take care of everything so you can live happily ever after.

The "AI Tax" evolved but didn't disappear. Instead of compound interest on failed systems, it's become the systematic cost of maintaining reliable collaboration between humans and AI. Oversight, verification, and quality assurance aren't obstacles to efficiency. They're the mechanisms that make AI assistance trustworthy.

## Why Colab Changed Everything

Context persistence across cells meant variables and functions didn't vanish mid-analysis. The notebook environment provided transparent documentation of every step. Most importantly, checkpoints could be inspected and verified before proceeding.

Once complete, the full notebook took about three hours to run from start to finish. Time to do something else while the pipeline processed everything. The system processed 863 words in the draft through to 892 words in the final version, tracking every transformation along the way.

Colab still doubled up on folders sometimes. A minor annoyance I could probably fix myself. But compared to empty CSVs and phantom functions, a duplicate folder was nothing. Manual verification at each stage was overhead, but it was productive overhead that increased confidence rather than consuming it through endless debugging.

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

In whole, I've approximately 60 hours of work across five attempts, producing four failed implementations, three paradigm shifts, two working solutions, and one uncomfortable truth: sometimes the answer you get is less important than discovering you've been asking the wrong question.

I finally have what I originally asked for. It has cells to measure trends over time, it produces visualizations so you don't have to read a fourteen decimal dataset to make sense of it. It's version 1.3.

That's where the final article begins.