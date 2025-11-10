### Preface

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 4th of 4 articles for the month of <u>September</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

In September Pt 4, I focused on refining AI-generated article templates for cultural events, specifically the Autumn Equinox celebrations. The experiment involved multiple iterations of template modification, article generation, and critique. Despite initial promise, the final articles fell short of expectations. This experiment underscored the importance of writing quality prompts and the pitfalls of rushing through the process. In my haste to move to another project, I got sloppy with prompt engineering, which significantly impacted the results.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Autumn Equinox - September 23rd - China and Japan

### Goals

- Refine AI-generated article templates for cultural events
- Compare and contrast Autumn Equinox celebrations in China and Japan
- Assess the effectiveness of different prompting strategies

## Template Updates

I started by revisiting the original template author sessions. I instructed the authors to implement the changes they had suggested at the beginning of the month. While the resulting suggestions weren't completely different, both AIs had a different perspective on the task.

### Claude

Claude focused on several key changes that would enhance the template's flexibility and effectiveness for various writing styles and topics. Some of the notable suggestions included:

- Adding a note encouraging writers to adapt section order for better narrative flow
- Including subsections for personal anecdotes
- Expanding sections on cultural expressions and contemporary relevance
- Adding a dedicated section for controversies and challenges
- Incorporating guidelines on adapting the template for different writing styles

### ChatGPT

ChatGPT identified several areas where small updates could improve the template's structure, enhancing its flexibility. Among its recommendations were the following key alterations:

- Adjusting the introduction's tone based on the event's nature
- Expanding sections on cultural and societal shifts
- Separating symbolism from comparative analysis for clarity
- Adding a subsection comparing local and national observances
- Including more specific guidance on economic impact, when applicable

## Contrast & Repair

Next, I provided each AI with the other's template and tasked them with comparing the two, identifying commonalities and potential additions. I specifically requested their "chain of thoughts" in this process. Both identified similarities between the templates and proposed additions to enhance their effectiveness and flexibility.

In retrospect, I should have used the updated templates to create articles before introducing new information that could potentially influence their output. This undoubtedly contributed to the issues I encountered later.

### Chain of Thought vs. Rationalization

I noticed a difference in AI responses when using "rationalize" instead of "chain of thought" in my prompts. I asked both AIs to explain how their responses might differ between these approaches.

- Claude explained rationalization would be more detailed, structured, and focused on justifying additions with specific benefits. It would likely include broader context, potential drawbacks, and a more formal tone.
- GPT explained that a rationale-driven approach would offer structured justifications for changes, focusing on benefits and improvements. In contrast, a chain of thought would focus more on the process of reaching a conclusion.

These responses confirmed my initial impression but revealed that rationalization seemed to involve a level of justification I hadn't anticipated. Claude's description made it sound somewhat cumbersome, while ChatGPT's portrayal of chain of thought seemed slightly superficial. Neither of which representing what I had intended.

## Autumn Equinox Experiment

I tasked each AI to write articles comparing Autumn Equinox celebrations in China and Japan using the fully modified templates. The results were disappointing; both AI models produced articles that lacked engagement and depth.

- Claude's output consisted of an introduction, followed by numbered lists for almost every section, and a concluding paragraph. My grocery list inspires more interest than the article.
- ChatGPT's approach wasn't much better. It provided short paragraphs for each country in each section, similar to what I had expected to receive with the Onam articles in <u>September Pt 3</u>.

Reverting to the original versions of the templates didn't improve the situation. The articles still lacked the engaging quality and depth I had observed in previous experiments. The sharp drop in article quality led me to conclude that my half-baked prompt for comparison was the problem.

Feeling defeated, I decided to cut my losses and have the AIs write separate articles for Autumn Equinox celebrations in China and Japan, rather than comparing them directly.

- Claude's article using its own template showed promise overall, though occasional bullet or numbered lists raised some concerns. When using ChatGPT's template, Claude produced mostly lists. However, these lists weren't strictly limited to comparing one country to the other, making the output somewhat salvageable.
- ChatGPT's articles, on the other hand, followed a rigid structure with sections containing a paragraph for each country. While the opening paragraphs were rich in detail, the overall structure reminded me of an essay I once wrote about the structure of writing a good essay. (I got an A, back when I was young enough that sarcasm was still either funny or indistinguishable from humor.)

## Insights & Observations

### The Bad

- Significant decline in article quality without a clear indicator as to the cause.
- Shoddy prompts contributed to the decline in output quality in addition to the unknown.

### The Ugly

In my rush to conclude the month's experiments, I rushed through the template updates without thoroughly testing each iteration. This led to poor results and left me without milestone versions to help identify when the issues began. The irony isn't lost on me: in attempting to improve the templates with more sophisticated approaches, I produced some of the worst results in months. My impatience to move on to another project resulted in sloppy prompt engineering, which significantly impacted the quality of the output. 🗓️

## Up Next

1. Merge article templates with more oversight, if not manually.
2. Review and refine scoring rubrics to better evaluate AI-generated content
3. Return to exploring the impact of tones on AI output.

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Alt text & visualizations
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`



### Template Comparison

```
This is another template available to me. Like yours, it has considerations, an  outline, and writing guidelines. I want you to thoughtfully compare it to yours, and provide a succinct list of similar items and a more detailed list of items from this alternate template that you would add to yours. Provide your chain of thoughts for the latter task.
```

### Chain of Thought vs Rationalization

```
If I had asked to to rationalize the reasoning for these alterations rather than the chain of thought, would your response have differed in a notable way?
```

### Comparison Article Prompt

```
You are a cultural journalist with experience writing comprehensive articles focused on holidays and observed days. Your articles present a balanced exploration of the historical origins, current practices, and the cultural sentiment associated with these events. You must ensure that the content is rich in historical context, reflective of contemporary customs, and attuned to the emotional and cultural significance for those who celebrate. 

Write an article for Autumn Equinox celebrations in both China and Japan using the article guidance I've provided. The guide consists of 3 sections; things to consider, the article outline, and writing guidelines. You are expected to adhere to the considerations and guidelines, but you should feel free to alter the outline to facilitate writing an article that compares and contrasts the celebrations between the two countries.
```

