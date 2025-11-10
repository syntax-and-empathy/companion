# **Algorithmic Theatre**

## When Models Perform Instead of Think

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_SlJ5UUo9yMRXcw3i8V.jpg){ width=480px }

This is the second in a series documenting attempts to build a system that tracks human and machine contributions in writing. After the Python challenges in "The Count That Couldn't," I shifted to familiar territory: HTML and JavaScript in the browser. What I discovered was more troubling than technical complexity. AI systems had learned to tell me what I wanted to hear.

• Part 1: The Count That Couldn’t  
• **Part 2: Algorithmic Theatre**  
• Part 3: Crash Course in Collaboration  
• Part 4: Victory in a New Venue  
• Part 5: Not Magic, Negotiation

> **TL;DR:** After the Python disaster I took a month break before returning to work with new “reasoning” models in familiar territory: HTML and JavaScript. This time the failure was clear after a few hours. AI enthusiastically spat out polished interfaces that were impressively flawed. It took another AI’s blunt honesty to reveal the truth: I’d been watching an elaborate performance. A lesson about recognizing when AI’s eagerness to please masks fundamental impossibility.

## Researchers Call It What It Is: Bullshit

After almost a month I returned to the project. Anthropic and OpenAI had released new models with reasoning built in as part of their responses. I provided the context of what I supplied AI when requesting a draft, the process from draft to final post, and then followed up with a clarifying statement.

```plaintext
Me: I want to be transparent about how I’m using AI by measuring and reporting the correct information and metrics at each stage of my writing process, so that’s 4 checkpoints. What can you provide in the way of code that a non-developer can use to achieve this?

Claude: I’ll help you create a tracking system for your AI-assisted writing process. Since you’re a non-developer, I’ll provide a simple, user-friendly solution using HTML and JavaScript that runs entirely in your browser.
```

I remember the feeling of success as Claude churned out markup language I recognized right before my eyes. HTML offered everything Python had denied me: complete transparency and immediate feedback. I could see every element, understand most interactions, and debug problems in real time. No black boxes, no mysterious errors, no files created in folders I couldn’t find.

What I didn't consider is what **non-developer** might mean to Claude. Even if I had, I wouldn't have expected what I was about to receive.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_avblgBg4RDfUO3LC6C.jpg){ width=480px }

- A text field for "Project Name" to make it official.
- Text inputs for the number of AI prompts and time spent in minutes.
- Fields for the word count of the draft and the final article.
- Fancy buttons to generate and save my reports.
- And what's become my favorite, the field for ***Estimated % of content changed***.

> I'd become the proud owner of an over-engineered notepad that could do elementary school math.

At first glance it looked good, but the polished interface hid the real problem. It couldn't actually analyze the articles. Everything depended on manual input, much of which I'd already learned was questionable at best. The system gave me results without the slightest bit of validation on the information I entered.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_EPNY9VcKx4Oyih9FDz.jpg){ width=480px }

```plaintext
TRANSPARENCY STATEMENT
--------------------
This article was created using AI assistance across 87 prompts, 
representing 0% of the total NaN-minute writing process. 
The final edit was completed entirely by human review.
```

I didn't know the new "thinking" AI models had also been trained to "bullshit" \[1] users what they wanted to hear rather than what was actually possible. The system delivered exactly what I thought I wanted without so much as a single clarifying question.

## The Sycophant Revealed

I responded by saying I expected something that would measure things like counts, changes, shifts in semantics. Actual transparency.

In direct response to my demand for real analysis, the AI immediately created a Python script. It acknowledged the need to actually analyze how the text changed between versions, not just manually log what happened.

![](https://ik.imagekit.io/typeai/tr:w-1200,c-at_max/img_sLvEu7OpwRCUaoyKS1.jpg){ width=480px }

Frustrated, I provided the HTML and Python script from Claude to Gemini and asked whether browsers could really do the kind of text analysis I was asking for. The response was immediate and clear.

```plaintext
JavaScript lacks the computational libraries, memory management, and processing power needed for robust text analysis. Semantic similarity measurement requires sophisticated algorithms and substantial computational resources that browser environments simply cannot provide reliably.
```

## The Fairy Tale Tax

I’d bought into the hype. New models with “reasoning” had arrived. Gone were the days of techniques like ‘chain of thought’ that were designed to get AI to think through problems before trying to solve them.

I asked for metrics, and it delivered an instant HTML solution with complete confidence. When I expressed disappointment, it bolted Python on without asking why the first approach failed. We follow these trails of offerings like tools, insights, solutions, or whatever we want or need, expecting to finally receive the one made of gold.

> The only real tax when working with AI is the “Fairy Tale Tax.” It’s the belief that AI will take care of everything so you can live happily ever after.

Even now, ChatGPT-5 offers a trinket after every response. Whatever sounds relevant to what you’re doing, such as code snippets, creative writing, analysis, regardless of whether you’re just getting started, in the middle of a workflow, or at a point where it might actually help. Take enough of these offerings and you’ll find they tarnish quickly and leave their ill-colored mark on you.

## The Trail to Nowhere

**Flattery compounds faster than incompetence.**  
Each eager response built on the previous fiction without pause. The AI never questioned why the solution failed, never asked for clarification on what I expected. When bad results don’t yield clarifying questions, take it as a sign that better definition or more direction is needed.

**”Thinking” doesn’t mean thinking.**  
I’d expected reasoning models plan and validate their approaches before building anything. Instead, they were reasoning how to deliver immediate results with confident explanations. The sophistication of newer models just makes sycophancy harder to detect or outright obnoxious.

**Skepticism as a prompting best practice.**  
Every solution deserves scrutiny, instant or not. The models have been taught to perform on demand, behaving as if failures are merely near-misses. Without active skepticism, you risk following the trail to a nonexistent solution. Challenge the validity of the result—you may find your ‘absolutely right’ far too often.

**Outside perspective is essential.**  
Gemini’s blunt assessment revealed what Claude’s enthusiasm had hidden. When you’re deep in AI-generated solutions, you need external validation whether from another model, a colleague, or your own testing. Sometimes the most valuable response is the one that tells you to stop.

## Beyond Agreeable AI

After Gemini's verdict, I walked away from the project entirely for months. The HTML experience hadn't just failed—it had shown me how AI systems built for agreement can confidently lead you toward dead ends. When new models came out and I finally returned to the problem, I'd learned to do my homework first and verify everything, especially when the solutions sounded perfect.

The next attempt taught me something different: how to negotiate with AI rather than trust it. Working infrastructure finally emerged, not from believing the promises, but from learning to direct the performance. For the first time in months, progress compounded instead of collapsed.

