This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 2nd of 3 articles for the month of <u>August</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

In this article, I look into a range of standalone prompts, from zero-shot prompts to more sophisticated "super prompts", using articles about Indonesia's Independence Day and Zhongyuan Jie as subject matter. As expected, more structured prompts generally produced higher-quality content.

File naming convention available in the appendix## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Indonesia's Independence Day - Indonesia
2. Zhongyuan Jie (a.k.a. Hungry Ghost Day) - China and other East Asian countries

### Goals

- Compare the performance of different prompting methods on different AI
- Identify patterns in output quality, length, and adherence to specified criteria

## Indonesia's Independence Day

To determine the impact of instruction detail on AI output, I first reviewed documentation from <u>Anthropic</u> and <u>OpenAI</u> to identify recommended prompting strategies. Based on this, I evaluated three distinct prompting methods combined with the versions with and without a template being provided.

**Zero-Shot Prompts**: These involve asking the AI to perform a task without any prior examples or context. The AI relies on its pre-trained knowledge to generate a response. This type of prompting is straightforward but may not always yield precise results due to the lack of specific guidance. [**<u>1</u>**]

For the initial articles, I provided each AI with a paragraph outlining the role, desired content, and expected process. After receiving the articles, I provided the template and requested that the AI rewrite the content accordingly. Surprisingly, the versions without the template scored higher, likely due to the sheer volume of content they produced.

File naming convention available in the appendix**Few-Shot Prompts**: Unlike zero-shot prompts, few-shot prompts provide the AI with a few examples of the desired output before asking it to perform a similar task. This helps the AI understand the context and format required, leading to more accurate and contextually appropriate responses. [1]

For the next few articles, I created new sessions and provided the JSON author role and 3 prior articles. Again, I gave a basic prompt, instructed the role to write an article using the 3 examples to indicate format and content, and then followed up with the template and requested a rewrite.

File naming convention available in the appendix## Zhongyuan Jie

Adding templates to few-shot prompts essentially met the definition for super prompts; however, I feel they still fall short considering I haven't incorporated a few other aspects, like tone guidance. That said, I was a little surprised that the prompts I have been working with can now be provided as a single prompt without the significant issues I experienced months ago.

**Super prompts:** Advanced, structured prompts designed to guide AI models in generating high-quality, relevant content. They go beyond basic prompts by incorporating multiple components that provide context, examples, and specific instructions, thus enhancing the AI's output quality and coherence. [1]

Hungry Ghost Day marked a milestone in my AI experiments, where I reached a new level of complexity and context being provided to each AI without at least one going up in flames. Previously, I had eliminated examples from my experiments to reduce the number of variables causing issues with the increasing sophistication of other prompts.

For the first time, I am providing a well-defined role, the article template, the process through which the AI breaks the writing tasks into smaller steps, and examples of prior articles.

File naming convention available in the appendixI can't express how satisfying it is to see these prompts execute without issue. It makes the countless number of challenges and time I lost to fixing them worth it.

## A Word on Words

Experiments for Indonesia's Independence Day more than demonstrated the tendency of AI to chat your eyes out through fluff and filler words and phrases to the detriment of the information being provided.

File naming convention available in the appendix. Unclipped title unavailable.## Insights & Observations

### The Good

- Consistent improvement in content quality as prompt complexity increased
- Super prompts excelled in producing well-rounded, high-quality content
- The foundational prompts I've been working performed issue free when supplied together

### The Bad

- It appears the well of excuses has run dry when it comes to writing something comprehensive.

### The Ugly

I created graphs that were a visual cacophony of drab colors and inconsistent bar heights that would make any data scientist wince. My minimal instructions to the AI, coupled with my fatigue induced by the last 48 hours of unrelated travel, led to charts that were more eyesore than insight. This fiasco stands as a testament to the old adage: garbage in, garbage out 🥱 📊

## Up Next

- Chain of Thought (CoT) prompting
- Scoring Criteria and template improvements

## References:

[1] <u>Perplexity</u>

## Appendix: Prompts & Responses

### File Naming Convention

AU = August

FN = Flooding of the Nile

ZJ = Zhongyuan Jie (中元节)

ZP = Zero Prompt

ZPT = Zero Prompt + Template

FP = Few Prompt

FPT = Few Prompt + Template

SP = Super Prompt

C = Claude

GPT = GPT-4o

### Zero Shot Prompt

```
As a Cultural Observances Analyst, I need you to write an article about Indonesian Independence Day. The article should include sections on the history of the day, modern practices, comparisons of those practices in varying areas, and the global perspective. Please write each section one at a time so that I may review it, finishing with the introduction and conclusion as the final pieces of the article.
```

### Few Shot Prompts

```
As a Cultural Observances Analyst, I need you to write an article about Indonesian Independence Day. Use these 3 past articles as examples of the expected format and content. Do not use the content of the articles directly as they are for other holidays.
```

