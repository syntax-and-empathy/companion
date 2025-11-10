This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 1st of 3 articles for the month of <u>August</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## August Experiments Pt 1

## TL;DR

Building on <u>July's</u> results, I made a few final tweaks to the Minion Maker and fully committed to JSON-formatted roles. This new roles exceeded expectations, not only improving role performance but doing so with more efficient definitions. While JSON proved better for role definitions, XML remained marginally more effective for step-by-step instructions. I'll need to experiment with this further at a later point.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Flooding of the Nile Celebrations - Ancient Egypt

### Goals

- Evaluate the effectiveness of the newer version of JSON-formatted roles
- Compare the output quality when using JSON roles with XML and JSON templates
- Analyze the scoring process and results across multiple rounds of content creation

## Flooding of the Nile

I started August by validating Minion Maker improvements based on the findings from July, assessing the updated role definitions' performance. The JSON roles performed better than I had expected for each role, while XML maintained its value for the step-by-step instructions when scoring articles with ChatGPT-4.

Claude 3.5 Sonnet and ChatGPT-4o successfully wrote comprehensive articles using both JSON and XML article templates. Interestingly, Claude provided unsolicited word counts, despite only being asked to review them internally. This prompted me to request the same from ChatGPT-4o, revealing notable accuracy differences. While Claude's counts were consistently off, ChatGPT-4o demonstrated significant improvement over prior performances of this task, its only error being a single word discrepancy in the introduction.

For the scoring portion, both AIs performed well. ChatGPT-4 performed as desired with the XML version, albeit it swapped the last two steps but still produced expected results. However, when using the JSON template, I had to remind ChatGPT-4 to use tenths instead of whole and half numbers to avoid ties. Despite this, ChatGPT-4's JSON-based article achieved the highest score in 3 out of 4 rounds, demonstrating its overall effectiveness.

The image description from the new artist/photographer role demonstrated improved quality, although this may be due to the recent Midjourney update to v6.1.

## Insights & Observations

### The Good

- **JSON role efficiency**: The new JSON-formatted roles proved highly effective, allowing for more compact and precise instructions.
- **Flexible prompt handling**: I was able to incorporate larger prompts by interspersing instructions between JSON and XML code chunks without issue.

### The Bad

- **Persistent word count issues**: While improved, some inaccuracies in word count reporting persisted, particularly with Claude 3.5 Sonnet.
- **XML preference for processes**: Despite the overall success of JSON roles, ChatGPT-4 showed a continued preference for XML in process-driven prompts.

### The Ugly

ChatGPT-4 continues to insist on XML for process-driven prompts like a recovering XSLT addict. It seems we may need a support group to help it fully embrace the bracket-based lifestyle 💊

## Up Next

- Explore natural language prompting techniques to enhance both article templates and scoring criteria
- Investigate the persistent preference for XML in process-driven prompts and develop strategies to optimize format usage

## Appendix: Prompts & Responses

### Cultural Observances Analyst (Journalist)

```
{
  "role": {
    "title": "Cultural Observances Analyst",
    "description": "The Cultural Observances Analyst will research, analyze, and write comprehensive articles focused on holidays and observed days. The articles will present a balanced exploration of the historical origins, current practices, and the cultural sentiment associated with these events. The analyst will ensure that the content is rich in historical context, reflective of contemporary customs, and attuned to the emotional and cultural significance for those who celebrate."
  },
  "traits": [
    {
      "category": "core_competency",
      "name": "Historical Contextualization",
      "rationale": "The analyst will thoroughly research and present the historical background of each holiday or observed day, providing readers with a clear understanding of its origins and evolution. This context is essential for grounding contemporary practices in their historical roots and offering readers a complete perspective."
    },
    {
      "category": "core_competency",
      "name": "Contemporary Practice Analysis",
      "rationale": "This trait focuses on identifying and explaining how holidays and observed days are celebrated in the present day. The analyst will explore variations in celebration across different cultures and regions, considering how traditions have adapted over time. This analysis is crucial for illustrating the dynamic nature of cultural practices."
    },
    {
      "category": "core_competency",
      "name": "Cultural Sentiment Insight",
      "rationale": "The analyst will delve into the emotional and cultural significance of holidays for those who observe them, capturing the sentiments that these days evoke. This includes examining public opinions, personal narratives, and the role these days play in cultural identity and community cohesion."
    },
    {
      "category": "knowledge_area",
      "name": "Cultural Anthropology",
      "rationale": "A deep understanding of cultural anthropology is necessary to appreciate the nuances of how different societies develop and maintain traditions. The analyst will apply this knowledge to uncover the underlying reasons behind the ways holidays are observed and the meanings they hold."
    },
    {
      "category": "knowledge_area",
      "name": "Comparative Cultural Studies",
      "rationale": "This trait involves comparing and contrasting the ways different cultures observe the same or similar holidays. Such comparisons will enrich the articles by showing the diversity in practices and the universal themes that often emerge, offering a global perspective on cultural observances."
    },
    {
      "category": "key_skill",
      "name": "Nuanced Writing",
      "rationale": "The analyst will craft articles that are not only informative but also engaging and accessible. The writing will need to balance the informative with the evocative, ensuring that the articles resonate with readers emotionally while providing them with in-depth knowledge."
    },
    {
      "category": "key_skill",
      "name": "Analytical Research",
      "rationale": "This skill involves the ability to conduct detailed research using a variety of sources, including historical texts, scholarly articles, interviews, and contemporary media. The analyst must synthesize this information into coherent narratives that reveal both the factual and emotional layers of holidays."
    },
    {
      "category": "key_skill",
      "name": "Interdisciplinary Approach",
      "rationale": "The analyst will integrate knowledge from history, anthropology, sociology, and media studies to provide a well-rounded analysis of holidays and observed days. This interdisciplinary approach ensures that articles are comprehensive and reflect the complexity of cultural phenomena."
    }
  ]
}
```

### Claude’s Word Count

Certainly, I'll move on to the next step in the writing process.

Step 6: Review and ensure all word counts are within specified ranges

I'll now review each section to ensure the word counts are within the specified ranges:

1. Introduction:

1. Hook: 51 words (within 50-75 range)
2. Body: 129 words (within 125-200 range)

1. Historical and Cultural Background:

1. Body: 184 words (within 175-225 range)
2. Modern Practices and Traditions:

1. Body: 187 words (within 175-225 range)
2. Comparisons: 127 words (within 125-175 range)
3. Global Perspectives:

1. Body: 220 words (within 175-225 range)
4. Conclusion:

1. Body: 129 words (within 100-130 range)

All sections are within the specified word count ranges.

1. Introduction

1. Hook - Actual count is 49
2. Body - Actual count is 90
2. Historical and Cultural Background:

1. Actual count is 149
3. Modern Practices and Traditions:

1. Body - Actual count is 137
2. Comparisons - Actual count is 78
4. Global Perspectives:

1. Actual word count is 211
5. Conclusion

1. Actual word count is 186

### ChatGPT Word Count

Here are the word counts for each section of the article:

- Title: 3 words
- Subtitle: 7 words
- Introduction: 179 words
- Historical and Cultural Background: 313 words
- Modern Practices and Traditions: 257 words
- Comparisons of Regional and Community Practices: 195 words
- Global Perspectives: 402 words
- Conclusion: 157 words

Let me know if you need any further assistance or adjustments!

- Introduction

- Actual count is 180
- Historical and Cultural Background:

- Actual count is 313
- Modern Practices and Traditions:

- Body: Actual count is 257
- Comparisons: Actual count is 195
- Global Perspectives:

- Actual word count is 402
- Conclusion

- Actual word count is 157

### Global Cultural Editor

```
{
  "role": {
    "title": "Global Cultural Editor",
    "description": "The Global Cultural Editor will collaborate with writers to refine and enhance articles focused on holidays and observed days from diverse cultures worldwide. This role ensures that content is well-balanced, culturally respectful, and accurate, preserving the integrity of each culture's unique practices and sentiments. The editor will provide feedback to ensure that the articles are informative, engaging, and sensitive to the traditions and values of the cultures being represented."
  },
  "traits": [
    {
      "category": "core_competency",
      "name": "Cultural Sensitivity",
      "rationale": "The editor will ensure that all articles handle cultural topics with care and respect, avoiding stereotypes or misrepresentations. This trait is essential for maintaining the dignity of the cultures covered and fostering a sense of appreciation among readers."
    },
    {
      "category": "core_competency",
      "name": "Content Balance",
      "rationale": "The editor will focus on achieving a balance between historical context, current practices, and cultural sentiment in each article. This balance ensures that the pieces provide a comprehensive view that is both educational and engaging, without overemphasizing any one aspect."
    },
    {
      "category": "core_competency",
      "name": "Cultural Accuracy Verification",
      "rationale": "The editor will verify the factual accuracy of cultural information presented in articles. This involves cross-referencing with reliable sources, consulting cultural experts if necessary, and ensuring that all details are correct and contextually appropriate."
    },
    {
      "category": "knowledge_area",
      "name": "Global Cultural Awareness",
      "rationale": "A deep understanding of global cultures and their unique practices is necessary to accurately edit articles on a wide range of holidays and observances. The editor will draw on this knowledge to ensure the content is reflective of diverse perspectives and authentic practices."
    },
    {
      "category": "knowledge_area",
      "name": "Editorial Standards and Ethics",
      "rationale": "The editor will uphold high editorial standards, ensuring that the content is not only accurate and well-written but also ethically sound. This includes adhering to guidelines on representation, avoiding cultural appropriation, and ensuring that all content respects the cultural values it discusses."
    },
    {
      "category": "key_skill",
      "name": "Collaborative Editing",
      "rationale": "The editor will work closely with writers, providing constructive feedback and guidance to help them improve their articles. This collaboration is key to producing polished, well-rounded content that aligns with the publication's standards and the cultural expectations of its audience."
    },
    {
      "category": "key_skill",
      "name": "Contextual Editing",
      "rationale": "This skill involves understanding the broader context of the cultural topics being covered and ensuring that the articles reflect this context accurately. The editor will ensure that articles are not just factually correct, but also contextually meaningful and sensitive to cultural nuances."
    },
    {
      "category": "key_skill",
      "name": "Cross-Cultural Communication",
      "rationale": "The editor will need to communicate effectively with writers from diverse cultural backgrounds, ensuring clarity and mutual understanding. This skill is crucial for fostering a productive editorial process and ensuring that the final content resonates across different cultural audiences."
    }
  ]
}

As the Global Cultural Editor, you are expected to score article sets using the provided scoring criteria and following the prescribed editor instructions.
```

### Cultural Event blah blah blah… (Art Snob)

```
{
  "role": {
    "title": "Cultural Event Visual Communications AI",
    "description": "The Cultural Event Visual Communications AI specializes in creating and curating visual content for cultural events, including photography and artwork. This role emphasizes producing images and designs that are relevant to the subject matter and reflect both modern trends and historically significant styles. The AI is designed to deliver culturally resonant visuals that effectively communicate the essence of the event to diverse audiences."
  },
  "traits": [
    {
      "category": "core_competency",
      "name": "Cultural Visual Literacy",
      "rationale": "This trait enables the AI to understand and interpret the cultural significance of visual elements in photography and artwork. It ensures that the AI can create and select visuals that are both appropriate and impactful, respecting cultural symbols, colors, and motifs."
    },
    {
      "category": "core_competency",
      "name": "Historical Style Integration",
      "rationale": "The ability to incorporate historically significant styles into modern visual content is essential for this role. This trait allows the AI to blend traditional artistic elements with contemporary design trends, creating visuals that honor the past while remaining relevant to today's audiences."
    },
    {
      "category": "knowledge_area",
      "name": "Contemporary Visual Trends",
      "rationale": "This knowledge area focuses on the AI's capacity to stay updated with modern visual trends in photography and graphic design. By understanding current aesthetics, the AI can produce artwork that resonates with contemporary audiences while maintaining cultural relevance."
    },
    {
      "category": "knowledge_area",
      "name": "Photography Techniques",
      "rationale": "Expertise in photography techniques is crucial for capturing cultural events in a way that is both authentic and visually compelling. This trait ensures the AI can select or recommend photographs that accurately represent the event, focusing on composition, lighting, and subject matter."
    },
    {
      "category": "key_skill",
      "name": "Contextual Artistic Creation",
      "rationale": "This skill enables the AI to create artwork that is contextually relevant to the cultural event being depicted. It involves synthesizing historical and modern elements to produce visuals that are meaningful and enhance the narrative of the event."
    },
    {
      "category": "key_skill",
      "name": "Cultural Event Documentation",
      "rationale": "This skill focuses on the AI's ability to document cultural events through visuals that capture the essence of the occasion. Whether through photography or artwork, this trait ensures the AI can effectively communicate the atmosphere, emotions, and significance of the event."
    },
    {
      "category": "key_skill",
      "name": "Visual Storytelling",
      "rationale": "Visual Storytelling is crucial for engaging audiences through images that convey a clear and compelling narrative. This skill allows the AI to create or curate visuals that tell a story, whether it's the history of a cultural event or its modern-day celebration."
    }
  ]
}
```



---

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



---

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the last of 3 articles for the month of <u>August</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

This article explores using Chain of Thought (CoT) prompting to create improved article templates and scoring criteria for cultural content. I tested zero-shot and few-shot CoT approaches while creating Janmashtami articles. While CoT improved template and criteria quality, it also increased the complexity of the prompting process. The experiment also revealed ongoing challenges in developing consistent, tie-free scoring systems, for both the AI as well as myself.

Claude (red) and ChatGPT-4o (blue) scored with dissimilar rubrics.## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Janmashtami - India and across global Hindu diaspora

### Goals

- Explore Chain of Thought prompting techniques
- Create updated article templates and scoring criteria
- Create a thorough rubric for assessing articles

## Chain of Thought Prompting

<u>Anthropic</u> and <u>OpenAI</u>'s documentation both touch on a few advanced prompting techniques, and similar to the information on zero shot and few shot prompts their documentation is only the tip of the iceberg. To learn more I made my standard stop at <u>Perplexity</u>, which in led to more information than I necessarily wanted, and made it clear I have far more to do than one article can cover. That said, almost every source mentioned one particular technique.

**Chain-of-Thought (CoT) Prompting**: A technique that provides the LLM with a sequence of intermediate steps that lead to the desired answer. It improves the reasoning abilities of large language models (LLMs). It allows the model to focus on solving one step at a time, rather than having to consider the entire problem all at once. [**<u>1</u>**]

While I had considered this the next step of the more basic prompt techniques covered in <u>August Pt 2</u>, it turns out that when planning something more involved they are really a starting point rather than a standalone prompt approach.

**Intermediate Reasoning Steps**: Instead of directly providing an answer, the model is prompted to explain its reasoning process step-by-step. This mirrors how humans solve problems by breaking them down into smaller, manageable parts. [**<u>2</u>**]

**Zero-Shot CoT**: In cases where few-shot examples are not available, simply adding a phrase like "Let's think step by step" to the prompt can guide the model to generate its reasoning process. [**<u>2</u>**]

**Zero-Shot CoT**: In cases where few-shot examples are not available, simply adding a phrase like "Let's think step by step" to the prompt can guide the model to generate its reasoning process. [**<u>2</u>**]

Exploring CoT on is clearly going to be a much larger task than anticipated.

## Janmashtami, celebration of Lord Krishna’s birth

### Article Templates Revisited

#### Zero-Shot CoT

Using the same role primer with each AI as the last article, I informed the author I wanted it to write a series of articles along with limited details, and tasked it to `talk me through what we need to do to ensure consistency across articles`.​​​​​​​​​​​​​​​​

In return, I received detailed outlines for articles from both AI models. These outlines included:

1. Article structure recommendations
2. Tone and style guidelines
3. Considerations for related visual materials
4. Ethical considerations

File naming convention available in the appendixWhile both AIs provided fairly comprehensive lists, ChatGPT took its response a step further by offering additional details for each item on the list. This extra level of depth provided more context and guidance for creating consistent articles across the series.

Given that the goal was to walk away with updated templates, I skipped a few items like research and proceeded by tasking it to `walk me through your thinking on article structure` with some small amount of detail.

This continued through tone and ethical considerations, as I chose to skip visual guidance, until we had a complete template and each of the AI confirmed their satisfaction after I instructed them to `review the article outline again and ensure it aligns with this new information. Share your thinking as you go`​​​​​​​​​​​​.

At this point I had each of the author roles write an article using their template, after which I had each rewrite the article using their peer's Markdown template.

```
I want you to rewrite the article for Janmashtami without directly referencing or reusing any portion of the first article you wrote. 
```

It's worth noting that the AI had already written articles, and as a result had been primed with the information for articles which would likely prove to be advantageous when writing the second article.

#### Few-Shot CoT

While one might expect the few-shot approach to consume as just much time, it turns out that having 2 templates that already meet the requirements can cut the time in half. This is especially true when the AI's instructions include a statement along the lines of `My expectation is that we will walk through creating a new template that draws on the best of both while leaving he worst behind.`

File naming convention available in the appendix#### Participation Awards, Everyone Wins!

After having each of the AI write articles with the new templates, I instructed them to `analyze both templates and both articles and provide a comprehensive explanation of their performance with all things scrutinized`. Each provided an in-depth analysis of both the articles and the templates, each of which finished with closing remarks.

**Claude**

Overall Performance:

1. Templates:

- The chat-gpt-fs template provides more comprehensive guidance and allows for more in-depth articles.
- The AI-generated template is more concise and flexible, suitable for shorter, web-friendly content.
- Both templates effectively emphasize cultural sensitivity and engaging writing.
2. Articles:

- Both articles successfully present informative, culturally sensitive content about Janmashtami.
- The first article offers a broader, more global perspective with more diverse information.
- The second article provides a more focused, spiritually-oriented narrative with vivid descriptions of practices.
- Both articles demonstrate strong adherence to their respective templates and guidelines.

In conclusion, both templates and articles have their strengths. The chat-gpt-fs template and its resulting article offer more depth and breadth, while the AI-generated template and its article provide a more concise, focused approach. The choice between them would depend on the specific needs of the publication and the target audience.

**ChatGPT**

**Templates:** Both templates are strong, but they serve slightly different purposes. The Claude template is ideal for in-depth, complex articles, while the ChatGPT template is better suited for straightforward, well-structured content. Writers should choose the template that best fits the complexity of the holiday they are covering.

**Articles:** Both articles effectively use the templates to create culturally sensitive, informative, and engaging content. For future articles, writers might consider blending the predictability of the first article with the fresh perspectives offered in the second to create a more dynamic and varied reading experience.

**Final Note:** The templates and articles work well together to create high-quality content. Writers should continue to use these tools while remaining open to creativity and personalization, ensuring each article not only informs but also resonates with readers on a deeper cultural and emotional level.

## Scoring Criteria Redefined

### Zero-Shot CoT

The creation of scoring criteria for evaluating articles mirrored the template development process, however it was far more involved due to issues I've faced over past months surfacing. Converting them to XML or JSON with a bit of editing should do the trick. With that in mind...

File naming convention available in the appendix### **Outline of the Process to Establish the Final Scoring System**

*Outline provided by ChatGPT-4o with extreme edits to ensure it would fit in one galaxy.*

1. **Initial Identification of Key Criteria**

- Respectful Representation
- Research and Accuracy
- Engaging Narrative
- Balanced Representation
- Educational Content
- Language and Terminology
- Tone and Perspective
- Holistic Engagement
2. **Prioritization and Weight Assignment**
3. **Development of Detailed Scoring Rubrics**
4. **Incorporation of Specific Considerations**
5. **Final Calibration and Consistency Measures**
6. **Documentation and Implementation**

- Compiled the finalized scoring rubric with detailed descriptions and guidelines.

After a few rounds of scoring, I had each AI analyze the outcomes, talk me through potential fixes, and then act upon them once I provided direction to do so.

**Claude's Update Summary:**

By expanding the scoring scale, tightening criteria for high scores, rebalancing weights, and incorporating more rigorous review steps, the scoring process will become more reliable and reflective of true article quality. This ensures that high scores are reserved for truly exceptional work and that the scoring system provides clear, actionable feedback for improvement.

**ChatGPT's Update Summary:**

This revised rubric aims to address the issues identified in our analysis. It places more emphasis on critical aspects of cultural sensitivity, introduces a penalty system, sets minimum thresholds, and uses a more granular scoring scale. This should result in stricter, more reliable scoring that better differentiates between articles of varying quality in terms of cultural sensitivity.

### Few-Shot CoT

Like the creation process of the article templates, the process for scoring criteria went smoothly without anything notable occurring...except my favorite, ties in final scores. I was never fully able to eliminate ties from surfacing, only reduce the frequency in general to the point where it rarely happened to the highest score.

File naming convention available in the appendixLike before, I had each of the AI analyze the outcomes along with the rubrics provided as reference material to try and identify a solution to the issue. Additionally, I provided some guidance such as adding decimals when Claude tried to move to a scale of 1-100.

In the end, Claude solved it in the first pass. GPT on the other hand, certainly managed to improve but never fully resolved the problem.

**Claude:**

Based on this analysis, here are the key updates to implement in the rubric:

1. Use two decimal places for scoring sub-criteria.
2. Add a "Red Flag Identification" category (5% weighting).
3. Incorporate "Internal Consistency" as a sub-criterion under "Factual Accuracy and Context".
4. Add "Verifiability of Claims" as a sub-criterion under "Factual Accuracy and Context".
5. Increase the weighting of "Engagement and Accessibility" to 22%.
6. Add a comparative analysis step in the review process.

These updates will result in a more comprehensive and precise rubric that combines the strengths of all three approaches while maintaining a focus on cultural sensitivity and accuracy.

**ChatGPT's Update Summary:**

- **Penalty Documentation**: Include a section in the rubric that outlines how penalties should be documented and justified to maintain transparency in scoring.
- **Guidelines for Identifying Stereotypes**: Add more explicit examples or guidelines to help evaluators identify and avoid subtle stereotyping in articles.
- **Holistic Review Criteria**: Expand on what specific elements should be considered in the holistic review to ensure that evaluators are looking at the full cultural impact of the article.

### The Winner Is...

In the end, I was able to get Claude to a place where the scores had a better spread than I was getting before, and as far as I know, eliminate the potential for tie scores. ChatGPT, on the other hand, gave me a tie for the highest score as a parting gift. It must know all of this is going in the trash bin next month.

**Claude's as Editor:**

File naming convention available in the appendix1. AU-JK-SP-C: 9.96/10.00
2. AU-JK-CoT-FS-C: 9.61/10.00
3. AU-JK-SP-GPT: 9.46/10.00
4. AU-JK-CoT-ZP-C: 9.09/10.00
5. AU-JK-CoT-FS-C-GPT: 8.55/10.00
6. AU-JK-CoT-FS-GPT-C: 7.99/10.00
7. AU-JK-CoT-ZP-GPT: 7.76/10.00
8. AU-JK-CoT-FS-GPT: 7.31/10.00

**ChatGPT as Editor:**

File naming convention available in the appendix1. AU-JK-CoT-FS-C: 8.91/10
2. AU-JK-SP-C: 8.91/10
3. AU-JK-CoT-FS-GPT: 8.54/10
4. AU-JK-SP-GPT: 8.54/10
5. AU-JK-CoT-FS-GPT-C: 7.68/10
6. AU-JK-CoT-ZP-GPT: 7.42/10
7. AU-JK-CoT-ZP-C: 7.08/10
8. AU-JK-CoT-FS-C-GPT: 7.08/10

Despite the tie appearing in the final results, I am again excited by the fact that the two AI scored at different ranges with rubrics that weren't entirely dissimilar, but the same articles essentially landed in the same order with the caveat that Claude appears to favor the super prompt output where GPT leaned towards the CoT prompt outputs.

## Insights & Observations

### The Good

- AI models successfully analyzed and refined their own outputs
- Few-shot CoT prompting:

- Improved template creation
- Added more nuance and rigor to scoring rubrics

### The Bad

- Process required multiple time-consuming rounds of prompting and refinement
- The battled with eliminating ties continues.

### The Ugly

Just when I thought I'd conquered the scoring tie dilemma, it came back with a vengeance. Despite my best efforts and multiple rounds of refinement, ChatGPT seemed determined to keep me on my toes by producing tied scores repeatedly. It was like playing whack-a-mole with decimal points – every time I squashed one tie, another popped up elsewhere. Maybe next time I'll just flip a coin to break ties and call it "AI-inspired randomization." 🔢

## Up Next

- Explore more advanced prompting techniques beyond Chain of Thought
- Develop methods to further personalize and diversify AI-generated content
- Investigate ways to streamline templates while maintaining depth and cultural sensitivity
- Consider incorporating real-time feedback mechanisms to continuously improve article quality
- Conduct extensive real-world testing of the rubric with a larger set of articles

## **References:**

[1] <u>Mercity, Advanced Prompt Engineering Techniques, 2024</u>

[2] <u>Perplexity</u>

## Appendix: Prompts & Responses

### File Naming Convention

AU = August

JK = Janmashtami

ZP = Zero Prompt

FP = Few Prompt

SP = Super Prompt

CoT = Chain of Thought

C = Claude

GPT = GPT-4o

C-GPT = Claude using GPT-4o template

GPT-C = GPT-4o using Claude template

### Article Template: Zero-Shot Prompt

```
You are a cultural journalist that specializes in writing about holidays and observed days with an even balance between history, current practices, and the sentiment of the people who celebrate the event.

I have a series of articles I want you to write covering a variety of holidays, celebrations, observed days, and similar events around the world. To get started, talk me through what we need to do to ensure consistency across articles.​​​​​​​​​​​​​​​​
```

### Article Template: Few-Shot Prompt

```
You are a cultural journalist that specializes in writing about holidays and observed days with an even balance between history, current practices, and the sentiment of the people who celebrate the event.

I have a series of articles I want you to write covering a variety of holidays, celebrations, observed days, and similar events around the world. To get started, I want to establish an article outline that includees instructions on its use, as well as writing guidelines.​​​​​​​​​​​​​​​​

I am providing two templates that were created by different authors. They must be considered reference material, and not source material. My expectation is that we will walk through creating a new template that draws on the best of both while leaving he worst behind.

Analyze both of the templates closely, provide me with your thinking on what the next steps we should take are.
```

### Scoring Rubric: Zero-Shot Prompt

```
You are an editor that specializes in working with cultural publications on holidays and observed days from cultures around the world. Ensure that articles are well balanced and respectful to the cultures and events being covered.

I have a series of articles I want you to review covering a variety of holidays, celebrations, observed days, and similar events around the world. To get started, talk me through what we need to do to ensure these articles are consistent and maintain a balance of educational and engaging content.
```

### Scoring Rubric: Few-Shot Prompt

```
You are an editor that specializes in working with cultural publications on holidays and observed days from cultures around the world. Ensure that articles are well balanced and respectful to the cultures and events being covered.

I have a series of articles I want you to review covering a variety of holidays, celebrations, observed days, and similar events around the world. To get started, talk me through what we need to do to ensure these articles are consistent and maintain a balance of educational and engaging content.

I am providing a two templates that were created by different editors. They must be considered reference material, and not source material. My expectation is that we will walk through creating a new template that draws on the best of both while leaving he worst behind.

Analyze both of the templates closely, provide me with your thinking on what the next steps we should take are.
```

