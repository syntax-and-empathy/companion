# Crash Course in AI Collaboration

## From Endless Failures to AI Collaboration

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_zzoBclWqXeaIaicvcR.jpg){ width=480px }

This is the third in a series documenting attempts to build a system that tracks human versus machine contributions in writing. After two rounds of rebuilding from scratch, I chose to start with the Python I’d ended up with from the last attempt rather than start over again, and I learned I should’ve been asking questions all along.

- Part 1: The Count That Couldn’t
- Part 2: Algorithmic Theatre
- **Part 3: Crash Course in Collaboration**
- Part 4: Victory in a New Venue
- Part 5: Not Magic, Negotiation

> **TL;DR:** After several attempts that resulted in little more than AI lies, I ended up on a different path, asking questions instead of demanding results. AI became a collaborator in addition to a code generator, introducing me to LDA topic modeling and TF-IDF vectorization, terms that sounded like academic gibberish until I verified them with Perplexity. For the first time, semantic similarity scores captured what I’d been chasing all along: how meaning actually evolves through revision. The results weren’t perfect, but they were real.

## Same Code, New Questions

I’d learned the hard way that recent AI models were designed to please while confidently walking you off a cliff. Every previous attempt to quantify content transformation had ended in frustration, empty CSVs, or polished interfaces that couldn’t actually analyze anything. Despite the shortcomings of prior attempts, the AI had given me a hint: **Semantic Evolution**, the final metric from the HTML farce.

A few days later, I returned to the project and chose ChatGPT over Claude (whose sting I could still feel) and simply dropped the old Python script into the chat box. No grand demands. No elaborate prompts about what I needed it to become.

```markdown
Me: ai-writing-analyzer.py

ChatGPT: I see you’ve uploaded an **AI Writing Process Analyzer** script that compares draft stages and generates metrics like word‐count change, similarity scores, vocabulary shifts, and even a transparency report. How can I help you with it? For example, would you like to:

*   Run it on some sample text and see the output?
*   Get a walkthrough of how each part works?
*   Add or tweak features (e.g. more NLP metrics, export formats)?
*   Improve documentation or usability?

Let me know what you’d like to do next!
```

More NLP metrics? Export formats? I didn’t hate the sound of that. I tossed in four versions of an article and stated the order they should be reviewed. It analyzed the versions and spat out the same counts and percentages I’d already seen. But then it went on to identify what had been done with each version.

## Questions Over Commands

For the first time in this entire project, I didn’t tell the AI what to do. I asked a question: had the document actually changed, or was it just cleaned up?

The response shifted from metrics to meaning. ChatGPT explained that major semantic shifts occurred in the first two transitions while the final stage was primarily cleanup. Then it made the offer, as expected.

```markdown
ChatGPT: If you’d like, I can run a topic-modeling or embedding-based similarity analysis to quantify how your key themes moved across versions—just let me know!

Me: That would be spectacular.
```

Moments later, a table materialized.

```markdown
**Topic Distributions by Stage**
| Stage   | Topic        | Topic 1      | Topic 2      |
| ------- | ------------ | ------------ | ------------ |
| Draft   | 0.0004017274 | 0.0004017274 | 0.9987752237 |
| Refined | 0.0002451478 | 0.0002451478 | 0.9992499420 |
| Edited  | 0.0003747287 | 0.0003747287 | 0.0003952585 |
| Final   | 0.0003923115 | 0.0003923115 | 0.0004097867 |
```

Numerical gibberish. Meaningless numbers that didn’t even equate to percentages, masquerading as insight. The AI rattled on about applying an LDA topic model with four topics, explaining the top five words in each and how they represented a shift from broader conceptual framing to something more specific.

I noted that the fourth topic didn’t exist in the table. That felt all too familiar.

## Latent Dirichlet Allocation (LDA)

It went on to spout technical terms with mentions of keywords and article refinement. I was sure the sycophant had prepared a repeat performance.

> Ready for the theatrical rollercoaster of nonsense, I asked “What is LDA?”

The response came loaded with "plain terms" that included "document-topic proportions," "probability distribution over vocabulary," and "hidden thematic structure." I skimmed through the entire explanation twice and couldn't decide if it was legitimate or the most elaborate hallucination yet.

I turned to Perplexity to verify the facts. LDA was not only real, but ChatGPT had actually simplified the explanation for me. The technical depth was legitimate. I still didn't understand LDA deeply enough to implement it myself, but I knew enough to verify the results made sense.

That didn't change the fact the data still looked meaningless.

### Term Frequency-Inverse Document Frequency (TF-IDF)

After a few more questions, ChatGPT offered to "switch to an embedding-based approach (e.g. Sentence-BERT)." Another promise of better results. My first thought was that Dall-E was about to spit out an image of Bert and Ernie \[1] hanging out with Count von Count \[2].

But this time, instead of accepting the promise blindly, I questioned the approach. It did a bit of "thinking" and produced a table of percentages.

```markdown
**Semantic Similarity Between Stages**
| Transition       | CosineSimilarity(%) |
| ---------------- | ------------------- |
| Draft → Refined  | 81.2                |
| Refined → Edited | 68.1                |
| Edited → Final   | 89.3                |
| Draft → Final    | 61.4                |
```

- Draft to Refined reflected the use of prompts to have AI update the article.
- Refined to Edited was demonstrative of significant removals, rewrites, and additions.
- Edited to Final captured that last pass to clean up questionable phrases and words.

Moments later, I had an updated Python script that provided scores for semantic similarity and could deliver CSV exports. The analyzer wasn't just working. It was documenting how revision actually happens, the patterns I'd experienced but lacked the time to document.

> There was just one catch. When I ran it again, I received results that also looked good, but were just far enough from the first set that I couldn't easily dismiss them.

This time, instead of accepting or rejecting the results outright, I asked why they varied.

The answer wasn’t what I expected. Semantic analysis uses statistical sampling. The algorithms introduce controlled randomness to run efficiently. Variation within a certain range wasn’t error. It was expected behavior.

TF-IDF vectorization and cosine similarity weren’t producing single “correct” answers. They were producing measurements within an acceptable margin. The question wasn’t “why do the numbers change” but “what range of change is acceptable.”

I didn’t need to fix the variance. I simply had more to learn.

## The Fairy Tale Tax

I arrived with the broken trinket I’d received before, expecting more of the same. Instead, I got the lesson I actually needed. Stop treating it as a magical solution and start treating it as an unreliable but capable assistant. One that required monitoring.

The “AI Tax” had shifted from compound interest to manageable overhead. Instead of complete failures requiring full rebuilds, the cost became verification, iteration, and understanding. Asking why results varied rather than demanding better results or blindly accepting whatever emerged.

> The only real tax when working with AI is the “Fairy Tale Tax.” It’s the belief that AI will take care of everything so you can live happily ever after.

The overhead was real, but acceptable. More importantly, it was productive.

Each question led to new understanding about how the analysis worked. Each failed approach revealed something about what might work better. Trial and error became the process rather than an obstacle to overcome.

The tax had to be paid either way, but I had a choice in how I paid it. Walk away knowing more, or walk away more frustrated with no clear path forward.

## What Inconsistency Actually Teaches

**Asking beats telling.**  
“How does this work?” instead of “Make this work” changed everything. Educational questions invite explanations that can be verified. Demands invite confident fiction. The shift from commanding AI to collaborating wasn’t about being polite. It was about getting verifiable results instead of plausible bullshit \[3].

**External verification isn’t paranoia.**  
ChatGPT explained LDA, but I used Perplexity to confirm it was real. No single AI source deserves complete trust. Checking multiple sources helped separate legitimate techniques from hallucination. Verification has become standard practice, everything gets checked independently.

**Understanding constraints beats trusting outputs.**  
The semantic similarity scores varied by slim percentages between runs. It wasn't a bug. It was statistical sampling combined with tokenization. Knowing what variance was acceptable mattered more than eliminating it entirely.

**You don't need mastery, but you need enough.**  
I still can't explain LDA topic models or TF-IDF implementation details, and I've no interest in doing so. But I learned enough to recognize when results aligned with expectations and when they didn't. The goal wasn't becoming a data scientist. It was knowing enough to spot when it's lying.

## The Graduation That Wasn't

I understood enough to know why the scores weren't exactly the same, that the range of variation was expected, and that tokens weren't the same as words. The results resonated with every execution of the script. I was closer than I'd ever been.

The shift from commanding to collaborating had produced something tangible. A working analysis with known limitations instead of confident fiction with hidden failures. I had something that worked. Real analysis with understandable constraints. I'd finally pulled it off.

Or so I thought.

What felt like graduation turned out to be the 101 course. What came next would make everything up to this point look like warmup exercises. I'd learned to ask questions about what I was looking to achieve. I was about to learn what happens when you ask AI to tell you how something can be improved.

---

<https://en.wikipedia.org/wiki/Bert_and_Ernie>

<https://en.wikipedia.org/wiki/Count_von_Count>

## Appendix

```plaintext
## Prompt:
The data and chart don't really seem to tell a story. What is LDA?

## Response:
LDA stands for **Latent Dirichlet Allocation**, which is a widely used “topic modeling” algorithm. In plain terms, it’s a way to automatically discover the hidden thematic structure in a collection of documents. Here’s the intuition:

1.  **Documents are mixtures of topics.**
    Imagine each document as a recipe that mixes a handful of “topics” in different proportions. One document might be 70% about AI ethics and 30% about design process; another could be 50/50 about storytelling and user research.

2.  **Topics are distributions over words.**
    Each topic, in turn, is defined by a probability distribution over your vocabulary. For example, an AI-ethics topic might assign high probability to words like “bias,” “fairness,” and “transparency,” whereas a design topic might favor “prototype,” “user,” and “interface.”

3.  **Dirichlet priors give it flexibility.**
    The “Dirichlet” part refers to a statistical assumption that documents will pick only a few topics (rather than all topics equally), and that topics will emphasize a small subset of words. This makes the model find coherent, focused themes.

4.  **The algorithm learns from word co-occurrence.**
    When you feed LDA a corpus (your various draft stages), it iteratively adjusts both:
    *   **Which words belong together** in a topic (topic → word distributions).
    *   **How much of each topic** appears in each document (document → topic proportions).

5.  **Output you get:**
    *   For each document (or stage), a vector of topic proportions (e.g. 60% Topic A, 25% Topic B, 15% Topic C).
    *   For each topic, a ranked list of the most representative words.

By looking at how those topic proportions change from Draft → Refined → Edited → Final, you can see how the **semantic emphasis** of your document shifts over time. If the results felt unclear, we can choose a different number of topics, tweak parameters, or even switch to another technique (like embedding-based similarity or clustering) to get more intuitive insights. Let me know!
```

