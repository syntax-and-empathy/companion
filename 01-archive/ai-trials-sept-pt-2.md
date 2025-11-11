This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 2nd of 4 (maybe 5) articles for the month of <u>September</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

September Pt 2 focuses on testing the Chain of Thought (CoT) rubrics from August as potential replacements for the current scoring system. This involved generating articles, analyzing scoring patterns, and creating comparative charts to identify noteworthy issues and trends. The new rubrics performed exceptionally well, relieving concerns about the need to address problems with the existing rubric.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Defence Day - September 6th - Pakistan
2. Brazil's Independence Day - September 7th - Brazil
3. Ganesh Chaturthi - September 9th - Hindu communities worldwide

### Goals

- Evaluate the effectiveness of the CoT rubrics from <u>August Pt 3</u>
- Analyze scoring patterns and trends for each AI model and rubric combination
- Identify improvements for future evaluation methods based on the new rubrics' performance

## Shots Fired

### Zero-Shot Articles

To address the scoring inconsistencies discovered in <u>September Pt 1</u>, I decided to put the two CoT rubrics created by Claude and GPT to the test alongside my original rubric. Using the new from the previous experiment, I generated zero-shot articles for the three holidays covered in the prior article.

I employed my JSON Editor role to apply each rubric in a separate session, scoring the four article versions for each holiday.

### One-Shot Articles

For the one-shot phase, I used the same sessions as the zero-shot articles, introducing my JSON Author and XML template. I instructed the AI to create new articles, using the zero-shot articles in the session as an example article. This approach fell somewhere between a true one-shot prompt and a super prompt - if I call it a "Limbo Prompt", that makes it a thing, right?

**Claude:**

- Maintained a tight range of scores, consistent with past experiments
- Scores were notably lower than the scores it has historically provided

**ChatGPT-4o:**

- Exhibited a wider range of scores compared to previous experiments
- Produced no ties, a stark contrast to earlier challenges with tie-breaking

### Few-Shot Experiment

With 12 articles in hand, I moved on to few-shot prompts. For each event, I supplied the AI with 2 of the 4 one-shot articles from the other two events as examples. So, when writing about Defence Day, the AI received 2 articles each about Brazil's Independence Day and Ganesh Chaturthi as context.

The scores derived from Claude's rubric were consistently distributed like the previous set. However, those using GPT's rubric had more in common with one another than any of the prior sets.

When creating the rubrics, I noticed that GPT had included details on comparative scoring. Having toyed with having the AI use the articles to establish the scoring scale in the past, I was curious how it might play out. The consistency in scoring and favored articles across both AI models suggests a crisp definition of how to score in the instructions. This, if it were to persist, would increase confidence in comparing scores across models.

### 120 Shots

I know what you're thinking. I haven't produced 120 articles with these templates, and older articles would contaminate the quality of the results. It's true, I only have 48 articles.

**GPT:**

Here's the breakdown of the math expressed in a structured format:**2 Scorers (Claude and GPT) * 2 Prompt Types (One-Shot and Few-Shot) * 2 Templates (C and GPT) * 3 Sets (BI, DD, GC) * 1 Score per Article = 48 Total Scores**

For those still reading and counting *(a.k.a. Ron)*, I did supply 1 article as a shot for 12 articles, and 4 articles as shots for 12 articles to 2 AI. 120 shots.

What I find most interesting when reviewing the 48 together is that both AIs consistently favored many of the same articles across the few-shot prompts. This preference emerged naturally, without scoring them as a single set or even in a single session.

## Insights & Observations

### The Good

- The new CoT rubrics' exceptional performance was surprisingly good.
- GPT's rubric yielded consistent scoring patterns across AI models, suggesting clear instructions that enable reliable comparisons.
- The final charts indicate potential for a unified rubric producing comparable scores across different AI systems.

### The Bad

- GPT-4o's interactive charts look great, but turn out to be useless when downloaded.

## Up Next

1. Apply the new CoT article template to several events over the coming weeks, assessing adaptability across multi-day and multi-country events.
2. Refine and expand on the prompting types used in this trial, seeking the point of diminishing returns.
3. Refine the CoT article template by testing unused AI suggestions and the results of its application to diverse holidays and cultural events.

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Alt text & visualizations
- <u>Mermaid Chart</u>: When it got complicated and the code got messy…
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`

## Appendix:

September Pt 2 Workflow