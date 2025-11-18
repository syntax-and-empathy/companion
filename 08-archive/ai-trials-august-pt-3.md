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

