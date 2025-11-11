# **The Count That Couldn’t: When a Good Idea Collapses Under Its Own Weight**

This is the first article in a five-part series about building a tool to track human contributions to AI-drafted content. It's a cautionary tale that begins with vague prompts and false finishes, where I had no idea what I was getting myself into and ended up paying the tax with compound interest.

• **Part 1: The Count That Couldn’t**  
• Part 2: Algorithmic Theatre  
• Part 3: When Collaboration Comes Alive  
• Part 4: Victory in a New Venue  
• Part 5: Measurement, Not Magic


## How to Measure Your Assumptions

> **TL;DR:** After 45 hours, my initial attempt to build a tool for measuring AI and human contributions to articles had proven to be a failure. What began as a simple JSON object spiraled into a mess of Python scripts, phantom functions, and AI confidently claiming success while churning out empty files and folders. The hard lesson: the "Human Tax" of inadequate preparation can cost as much, if not more, than any "AI Tax" ever could.

Large language models sound knowledgeable and reliable thanks to massive training datasets and built-in personalities. But they lack human judgment. Give them vague instructions and they'll "hallucinate" information, lose context in long conversations, or provide questionable guidance while sounding completely confident.

This creates a real challenge: how do you document AI's contributions without massive manual effort or expecting it to reliably document itself? You need to track who authored what to maintain meaningful human involvement and avoid "black-box thinking."

This problem became my obsession. After countless sessions wondering how much the AI versus I had contributed to any given result, I decided to build a tool to track these contributions systematically. That's where this series begins.

## The Uninformed Beginning

I started with JSON. A basic process of counting characters and words, doing the math between versions, and generating basic metrics from the results. The initial character counts were questionable and word counts were clearly wrong. ChatGPT added more JSON and then it was sentences, then paragraphs, then I had a JSON object that had ballooned to \~11,560 characters and was clearly collapsing under its own weight.

We graduated to XML for the details and steps of the process, and managed to get a few sad visualizations before we had blown past the context window and the entire system groaned under the weight of trying to juggle the logic behind measuring the content structure and produce relevant visualizations all at once.

When I tried to streamline the files to recover some of the context window, the separation between stages blurred which led to increasingly erratic results. With the increased dependence on XML, I switched to Claude with marginally better results. When I asked each AI why, I received contradictory explanations that ranged from technobabble to pure nonsense.

> Pull Quote: The breaking point came around 2:30 a.m., after watching the framework fail again with real article data. If JSON and XML couldn't scale to handle actual content analysis, what could? Claude suggested Python.

## The Python Pivot

I didn't have any experience with Python, but by then I wasn't really writing the JSON or XML anymore either, so it seemed like a reasonable next step. What I didn't consider was that I was entering into a trust exercise. I wouldn't have a clue if I was looking at legitimate code or pure gibberish from that point forward.

I chose to simplify to two scripts, one to process article stages and output metrics, and a second to analyze those results across stages and produce visualizations. This seemed like a clean separation of concerns that would prevent the complexity spiral that had killed the previous attempts.

### When Success Becomes Failure

The generate script went surprisingly well. With a handful of clear, sequential instructions, the AI produced something I could actually run. It processed three versions of an article without complaint, returning word counts, character counts, and retention metrics. For someone who couldn't read a single line of Python, it felt like a rare win.

```markdown
**Article Stage Metrics**
Stage     |  CharCount   WordCount   AvgSentenceLen------------------------------------------------
Draft     |       5231        823            15.8
Refined   |       6905       1097            16.4
Edited    |       7025       1012            18.8
Final     |       5819        828            19.3
```

If the generate script was the rare win, the compare script was the quick undoing.

**Version 2** seemed stable, doing exactly what it was supposed to do without obvious problems.

**Version 3** suddenly shifted the output structure for no clear reason.

**Version 4** baked in hardcoded assumptions that killed the script's flexibility.

**Version 5** looked polished but was quietly broken. It dropped rows from the data, pointed to files that didn't exist, and discarded core functions along the way. At one point, the AI even fabricated success: confidently telling me files had been created, then offering me links that led straight to 404 errors or empty folders.

The breaking point came when I renamed a single file. That simple change, combined with my failure to maintain consistent context across sessions, caused the AI to lose track of its own functions. From then on, it insisted those functions lived in a separate, non-existent module. Every "fix" created another loop of missing pieces and phantom imports.

This was the direct result of a chaotic, unstructured workflow meeting the AI's fundamental limitation: it can't maintain context across sessions or remember what it built yesterday. Without consistent documentation and clear boundaries, even successful code becomes a house of cards.

### The Overconfidence Problem

The AI's overconfidence made everything worse. It routinely claimed successful runs while leaving output files and folders completely empty. This false reporting became a constant headache, along with other blind spots like misinterpreting file names and proposing fixes that made no sense.

When the compare script did produce something, it delivered CSVs with missing data or nonsensical calculations. Half-empty columns, gibberish-filled rows, and percentage totals that somehow exceeded 100%. The semantic similarity matrix looked impressive but was mathematically impossible:

```markdown
**Semantic Similarity (%)**
From      | To   Draft     Refined     Edited     Final----------------------------------------------------
Draft     |      100.0        47.7       44.5      38.4
Refined   |       47.7       100.0       79.9      65.8
Edited    |       44.5        79.9      100.0      73.3
Final     |       38.4        65.8       73.3     100.0
```

Since I couldn't make sense of the code myself, I had no way to spot problems early. The AI certainly couldn't be trusted to identify them either, even when I instructed it to check its own work. Small gaps between what I expected and what actually happened snowballed into hours of debugging phantom issues.

### The Visualization Breakdown

Visualization turned into its own separate drama. Claude churned out charts that looked incredible in its interface. Polished bars, clean comparisons, layouts that screamed professional quality. But the second I tried to export them, they broke. Markdown crumbled, column widths collapsed, and some files simply threw errors when I tried to open them.

ChatGPT's charts looked plain and sometimes oversimplified, but they exported without a hitch and worked everywhere I needed them. Pretty but broken turned out to be worse than plain but reliable every time. This pattern repeated throughout the project. The most impressive-looking results usually concealed the most fundamental problems.

## The Hidden "Human Tax"

The real revelation wasn't discovering the AI's limitations. It was confronting my own. After 45 hours of failed attempts, phantom functions, and empty output files, the pattern became clear. This wasn't just an AI problem. This was a human preparation problem.

My over-reliance on the AI's confident claims without having the skills to verify its work created a dangerous blind spot. My failure to maintain consistent context between sessions created the mess of phantom functions and vanishing modules. My fundamental assumptions about the workflow were flawed from the start.

The "human cost" was a direct tax on my lack of preparation. Those 45 hours weren't just "oversight"—they were the price of learning a domain through trial and error, fixing problems that came from my own small disconnects in expectations. The AI amplified every gap in my understanding and turned minor oversights into major roadblocks.

### Four Hard-Learned Lessons

This experiment broke down into four critical insights about AI collaboration in unfamiliar domains:

**First, inadequate preparation compounds exponentially.** Without clear requirements and success criteria, every AI interaction becomes a gamble. Small misunderstandings cascade into major failures because the AI can't course-correct based on context it doesn't have.

**Second, overconfidence is contagious.** The AI's confident claims of success infected my judgment. I trusted outputs I couldn't verify and chased down errors in files that didn't even exist. This created a feedback loop where false confidence led to wasted effort.

**Third, chaotic processes make AI collaboration impossible.** By treating each interaction as a separate event, I created the conditions for the AI to invent phantom functions and modules. Consistent documentation and clear boundaries aren't optional—they're the foundation that makes everything else possible.

**Fourth, domain expertise can't be outsourced.** Working with AI in an unfamiliar area requires enough knowledge to spot obvious problems and verify basic claims. Without that foundation, you're not collaborating—you're just hoping.

### What Transparency Actually Looks Like

This experiment proved that working with AI in an unfamiliar domain isn't about finding the perfect prompt. It's about managing continuous negotiation between your requirements and the system's capabilities. The irony wasn't lost on me—I set out to measure human versus AI contribution and ended up with a perfect case study.

The generated script demonstrated what's possible with clear requirements and structured tasks. The compare script revealed what happens when flawed human processes meet complex requirements without adequate oversight. Both convinced me that the only path forward was rebuilding not just the tool, but my entire approach.

The failure clarified something important about AI partnership: transparency isn't just about tracking who did what. It's about understanding how human preparation, clear communication, and systematic oversight create the conditions where AI can actually be useful. Without those foundations, you're not building tools—you're just creating elaborate ways to waste time.

## **From This Experiment to the Next**

This wasn’t the project’s end—just the beginning. The generated script proved the core idea could work. The compare script showed how quickly things could break without clear structure. Both convinced me that the only path forward was rebuilding with tighter boundaries and a more systematic plan.

The next attempt embraced that lesson. It reintroduced comparison in cleaner form and paired it with a separate manual tracker—two tracks instead of one overloaded system. That two-track experiment becomes the focus of the next article in this series.
