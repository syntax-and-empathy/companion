This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 1st of 4 (maybe 5) articles for the month of <u>September</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

This month, I'm rebooting Chain of Thought (CoT) prompting after encountering questionable results last month. This article documents my second attempt at creating a new article template using CoT prompts. While I initially planned to test it extensively on holidays beyond Labor Day and Vietnam's Independence Day, I uncovered a scoring issue that may be an anomaly, a new development, or simply something that went unnoticed until the volume of articles without additional variables grew large enough to reveal it.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Yamashita Surrender Day - September 2nd - Philippines
2. Labor Day - September 2nd - United States
3. Vietnam's Independence Day - September 2nd - Vietnam

### Goals

- Develop and refine new article templates using Chain of Thought (CoT) prompting
- Test new templates with zero-shot and one-shot article generation
- Identify areas for improvement in CoT prompting techniques
- Experiment with standardizing the visual aspects of the scoring charts

## CoT Article Template Revisited

At the end of <u>August</u>, the results indicated that the prompts I've been using all along, on their own, were yielding better results than those created through the CoT approach. Simply put, I found this hard to believe.

Based on this belief, I read various articles and documents I could find on the subject. I wanted to gain a better understanding of the approach and use the month of September to improve on how I employ it.

This incremental process, each prompt met with verbose AI responses, became the focus of this article. While I had considered introducing instructions to limit the responses I didn’t want to risk undermining the process. Rather than detailing every exchange, I've outlined the key steps in developing the CoT template:

1. I began by tasking the AIs to create a template for articles about holidays, observed days, festivals, and similar cultural events worldwide. This set the foundation for our template development.
2. After receiving the initial outlines, I instructed the AIs to critically review their work and suggest additional content considerations for both general and specific event types, along with the rationale for each.
3. I told the AIs to merge new additions into their outlines. Claude listed new sections, bookended by `[Previous content remains the same]` and `[Rest of the previous content remains the same]`. It didn't show where they'd fit in the original outline, so I lacked context for their placement.
4. To finalize the outline, I asked the AIs to conduct a comprehensive review of their updated outlines, requesting recommendations for improvements that would ensure compelling storytelling while maintaining educational value.
5. Based on their recommendations, I directed the AIs to add specific items to their outlines. Both AIs gave almost identical suggestions, differing only in the order they were supplied. GPT included an item on ethics which is certainly relevant but not as a content item, choosing to defer it to the guidelines portion of our exchange.
6. Next, I tasked them with identifying key considerations for writing about culturally specific occasions and reasoning for each suggestion. These, in turn, were used as the basis of our writing guidelines.
7. Given the realistic constraints of an AI-assisted writing process, I had them eliminate items requiring external sources or extensive research beyond readily available information.
8. I then asked the AIs to review the entire template and writing guidelines holistically and provide guidance on considerations for ensuring high-quality articles that cover the breadth of event types with engaging and educational depth.
9. As the final step, I instructed the AIs combine the completed outline, writing guidelines, and instructions into a comprehensive template. This resulted in a cohesive guide for our AI-assisted content creation, clearly delineating its components.

The process of creating this template was smooth much like the first was. That said, the scoring mechanism has always been plagued with a variety of issues that I am not looking forward to. Along the way there were a number of recommendations made during the template creation process that I'll come back and consider in a later set of articles.

### Super Prompt Shenanigans

I used my standard super prompts (SP) to generate articles for each holiday using both AI models. These SP articles would later serve as a baseline for comparison with templates from the new Chain of Thought (CoT) approach.

I created half of these articles in a single session per AI, while generating the remaining half in separate sessions. Upon rendering the charts, I immediately noticed a problem.

The scores for each successive set of articles were incrementally higher without clear justification. After confirming my suspicion, it's clear that I need to revisit my evaluation process to ensure consistent results. Fodder for future articles.

## Insights & Observations

### The Good

- The CoT prompting process yielded a robust, comprehensive article template that effectively guides AI-generated content creation for cultural events and holidays.
- The iterative template creation process, while time-consuming, resulted in a valuable tool for future content generation experiments, with clear distinctions between outline, guidelines, and instructions.
- The holistic review approach encouraged the AIs to consider both engaging storytelling and educational depth, leading to more well-rounded content suggestions.

### The Bad

- The AI responses, while comprehensive, were often verbose, resulting in lengthy conversations that needed to be distilled into key points for practical use.
- There were challenges in contextualizing new content suggestions within the existing outline structure, as seen with Claude's bookended additions.
- The scoring process for SP articles revealed potential biases introduced by using a single editor session, emphasizing the need for more robust evaluation processes.

### The Ugly

In a surprising turn of events, Claude seemed to "forget" about articles it had previously written when asked to compare them. Despite its typically superior memory compared to GPT, Claude insisted it hadn't authored certain articles, even after multiple attempts at clarification. This unexpected lapse in continuity highlighted the unpredictable nature of AI behavior and the importance of clear, consistent prompting techniques.

## Up Next

1. Apply the new CoT article template to several events over the coming weeks, assessing its effectiveness and adaptability across different contexts.
2. Analyze the results from scoring multiple holiday articles to identify patterns, strengths, and areas for improvement in the template's performance.
3. Refine the CoT article template by testing unused AI suggestions and the results of its application to diverse holidays and cultural events.

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Visualizations
- Mermaid Chart GPT: Draft flowchart
- <u>Mermaid Chart</u>: When it got complicated and the code got messy…
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`

## Appendix:

### CoT Prompts

```
Create a template for articles specific to holidays, observed days, festivals, and similar events for cultures around the world. The articles need to provide the story of the holiday including it's historical origin, how it has transformed over time, modern day practices, and the global perspective of the event. Provide the rationale for every piece of content you recommend.
```

```
Review the outline and any additional content that should be considered in genreral, as well as content that would be specific to a single type of event (e.g. holiday but not festival). Provide the rational for any you identify so that I might decide which I would like added to the template.
```

```
Perform a holistic review of the article outline and provide a list of recommendations on how it might be improved to ensure engaging storytelling without compromising educational aspects of the content.
```

```
This is the template we'll be moving forward with. Do not alter it unless specifically requested.

Identify key considerations when writing articles with this template for culturally specific occasions. Provide the reasoning for each item you suggest.
```

```
Remove items that require external sources, such as speaking with individuals or research beyond the sum of the information readily available.
```

```
These should be discussed as guidelines for writing the articles going forward.

Review the article outline and identify any additional writing guidelines that should be included while respecting the limitations previously mentioned.
```

```
Add all of the writing guidelines as an appendix to the article outline.
```

```
Next, you will review the sum of the template and writing guidelines holistically. Provide guidance on things to consider to ensure high quality articles that cover the breadth of the event types being considered with depth that is both engaging and educational.
```

```
Add these considerations as guidance to the author before the existing outline and guidelines and provide me with the resulting document in its entirety.
```

### Diagram Code

#### CoT Workflows

```
---
config:
  theme: base
  layout: elk
---
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#fff',
      'primaryTextColor': '#515B61',
      'primaryBorderColor': '#515B61',
      'lineColor': '#0E394C',
      'secondaryColor': '#55A5B3',
      'tertiaryColor': '#E0EBEF'
    }
  }
}%%
flowchart TD
    subgraph CCoTT["Claude CoT Template"]
        CDO("Draft Outline") --> CIC("Identify Content") --> CMC("Merge Content") --> CR("Review")
        CR --> CU("Update") & CG("Guidance")
        CU --> CRD("Rough Draft")
        CG --> CWG("Writing Guidelines") --> CRI("Remove Items")
        CRD --> CCR("Review Template") --> CC("Considerations") 
        CRD & CRI & CC --> CAG("Claude's Article Guide")      
    end
    subgraph GCoTT["GPT-4o Template"]
        GDO("Draft Outline") --> GIC("Identify Content") --> GMC("Merge Content") --> GR("Review")
        GR --> GU("Update") & GG("Guidance")
        GU --> GRD("Rough Draft")
        GG --> GWG("Writing Guidelines") --> GRI("Remove Items")
        GRD --> GCR("Review Template") --> GC("Considerations") 
        GRD & GRI & GC --> GAG("GPT's Article Guide")      
    end
    subgraph CoT["Chain of Thought Templates"]
        CCoTT
        GCoTT
    end
    style CoT fill:#D3E4E6,stroke:#000000
```

### (not so) Super Prompt Workflow

```
---
config:
  theme: base
  layout: elk
---
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#fff',
      'primaryTextColor': '#515B61',
      'primaryBorderColor': '#515B61',
      'lineColor': '#0E394C',
      'secondaryColor': '#55A5B3',
      'tertiaryColor': '#E0EBEF'
    }
  }
}%%
flowchart TD
    subgraph ClaudeA["Claude Articles"]
        01("Claude") --> 10 & 11 & 12
        10("LD-SP Article")
        11("VI-SP Article")
        12("YS-SP Article")
        02("Claude") --> 13("DD-SP Article")
        03("Claude") --> 14("GC-SP Article")
        04("Claude") --> 15("BI-SP Article")
    end
    subgraph ClaudeB["Claude Scoring"]
        01b("Claude") --> 30 & 31 & 32 & 33 & 34 & 35
        30("LD-SP Score")
        31("VI-SP Score")
        32("YS-SP Score")
        33("DD-SP Score")
        34("GC-SP Score")
        35("BI-SP Score")
    end
    subgraph GPTA["GPT Articles"]
        06("GPT-4o") --> 20 & 21 & 22
        20("LD-SP Article")
        21("VI-SP Article")
        22("YS-SP Article")
        07("GPT-4o") --> 23("DD-SP Article")
        08("GPT-4o") --> 24("GC-SP Article")
        09("GPT-4o") --> 25("GC-YS Article")
    end
    subgraph GPTB["GPT Scoring"]
        06b("GPT-4o") --> 40 & 41 & 42 & 43 & 44 & 45
        40("LD-SP Score")
        41("VI-SP Score")
        42("YS-SP Score")
        43("DD-SP Score")
        44("GC-SP Score")
        45("BI-SP Score")
    end
    subgraph SP["Super Prompts"]
        JSNA("JSON Author Role") --> ClaudeA & ClaudeA & GPTA & GPTA
        XMLT("XML Template & Process") --> ClaudeA & ClaudeA & GPTA & GPTA
        ClaudeA --> ClaudeB
        GPTA --> GPTB
        JSNE("JSON Editor Role") --> ClaudeB & ClaudeB & GPTB & GPTB
        XMLE("XML Rubric") --> ClaudeB & ClaudeB & GPTB & GPTB
    end
    style SP fill:#D3E4E6,stroke:#000000
```



---

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

---

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 3rd of 4 (maybe 5) articles for the month of <u>September</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

In September Pt 3, I challenged AI authors with diverse subjects and inputs, pushing them to adapt existing templates and evaluation methods. I tested their ability to handle a multi-day festival without explicit instructions, which yielded unexpected pleasant results. I also conducted tests using a wider variety of markup languages—JSON, Markdown, TOML, XML, and YAML—to assess their impact on AI-generated content quality. By employing o1-mini for analysis, I uncovered trends suggesting that I may have prematurely settled on JSON as the optimal format.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o
- o1-mini

### Holidays

1. Ethiopian New Year - September 11th - Ethiopia
2. Onam - September 5th to 15th - Hindu communities worldwide
3. Mexico's Independence Day - September 15th

### Goals

- Evaluate tempalte performance across different markup languages
- Analyze trends in AI-generated content using o1-mini
- Test AI authors' ability to adapt templates for multi-day celebrations

## Ethiopian New Year

For the first event this round, I focused on the Ethiopian New Year. I limited activities to producing articles using one-shot and few-shot prompts, which I planned to use in upcoming tests. With most September holidays clustered in the first few weeks, I will also use these articles for trend analysis later.

While creating charts for this set of articles, I noticed an interesting pattern. Claude's scoring remained remarkably consistent across different rubrics, while GPT-4o exhibited more variation without contradicting itself.

## Onam

Onam presented a unique challenge: a 10-day festival that would test our AI authors' ability to handle a complex, multi-day event without explicit guidance. I subtly modified my standard prompt, indicating that the author should adapt the template as needed for the event. Beyond that, I left them to their own devices.

To spice things up, I included my JSON author role alongside the author role I've been working with this month. This allowed me to compare articles and assess the impact of role specificity on handling complex events.

All four AI sessions (two each for Claude and GPT-4o) surprised me by integrating information about different days throughout their articles, eschewing the day-by-day breakdown I had anticipated. When questioning their approach, each AI provided solid reasoning for their choices. This unexpected twist not only demonstrated the flexibility of the new templates but also demonstrated improvements within the AI models or their respective updates since my last experience with a similar event.

### Scoring Comparison: The Impact of Role Specificity

As the month progressed, my curiosity about the performance of the new templates and how they might affect my JSON author grew. Rather than having the authors write smaller articles for each day of Onam, I opted for a comparison between articles generated by the JSON author role and those created with less specific prompts.

Here's where things get interesting, the JSON author consistently produced the highest-scoring articles. However, some articles generated by the basic role outperformed the lowest-scoring JSON author pieces. It's worth noting that the prompt instructing the basic role included a reasonable paragraph providing context for the author. This raises questions about whether the JSON format might be as restrictive as it is structured, despite offering the same level of event context.

## Mexico's Independence Day

For my final experiment, I tested the impact of markup languages on AI-generated content using Mexico's Independence Day as the subject. To explore a range of possibilities, I transformed our Markdown template into several formats:

- <u>JavaScript Object Notation (JSON)</u>
- <u>Markdown (MD)</u>
- <u>Tom's Obvious Minimal Language (TOML)</u>
- <u>Extensible Markup Language (XML)</u>
- "<u>YAML Ain't Markup Language</u>" or "Yet Another Markup Language." (YAML) [<u>1</u>]

This produced 20 articles, which I scored using the JSON editor with both established rubrics. To ensure a thorough analysis, I employed two AI "researchers" for initial assessment. I then fed their reports to o1-mini for a comprehensive trend analysis.

While many of o1-mini's insights aligned with my expectations, a few findings stood out:

- **Performance by Format:**

- **Researcher 1** identifies that articles in **YAML** and **XML** formats tend to score higher (7.80 to 8.02), whereas **JSON** and **TOML** formats fall into a mid-to-upper range (7.50 to 7.90).
- **Researcher 2** highlights that **TOML** files consistently achieve the highest scores (up to **8.95**), followed by **XML**, **JSON**, and **YAML** formats in descending order.
- **Naming Conventions and Specific File Types:**

- **Researcher 1** points out that files with names like **"GPT+YAML-GPTt.md"** achieve the highest scores, suggesting that GPT augmentation in specific formats enhances quality.
- **Researcher 2** observes that **"Ct.md"** files generally score higher than their **"GPTt.md"** counterparts, implying that non-GPT-enhanced versions may receive more meticulous attention or different processing methods.
- **Top-Performing Articles:**

- **Researcher 1** identifies specific high-scoring articles such as **SE-MI-FP-GPT+YAML-GPTt.md (8.02)** and **SE-MI-FP-GPT+XML-GPTt.md (7.96)**, highlighting their excellence in factual accuracy, cultural representation, and sensitivity.
- **Researcher 2** points out **SE-MI-FP-C+TOML-Ct.md (8.95)** as the highest scorer, emphasizing the superior performance of TOML-formatted articles.
- **Consistent Ranking Patterns Across File Types**

- **Researcher 2** observes a consistent ranking pattern across different file types: **Ct > TOML-GPTt > XML-GPTt > JSON-GPTt > YAML-GPTt**, which aligns partially with Researcher 1’s findings on format performance.

Of all the markup languages I tested, TOML was the one I assumed to see the least interesting results from. However, I'm not rushing to conclusions just yet. The experiment's scope was limited—we only examined a single holiday with four articles per markup language. While this is certainly noteworthy, it's far from definitive.

## Insights & Observations

### The Good

- AI models demonstrated unexpected adaptability when handling multi-day events.
- Our markup language experiment revealed potential benefits of formats beyond JSON.

### The Bad

- Testing multiple templates and rubrics ended the Onam experiment prematurely.
- New model releases with built-in Chain of Thought capabilities raise questions about the value of the AI Trials in general.

### The Ugly

Claude's incessant apologizing has reached new heights of annoyance. I'm considering a prompt to ban phrases like "I'm sorry" just to have a normal conversation. At this rate, I half expect Claude to start offering virtual coffee and donuts to make up for its perceived shortcomings. 🍩

## Up Next

- Merge templates and rubrics to streamline our experimental process.
- Conduct a broader study on markup language effectiveness across steps in my approach.
- Investigate o1-mini's potential for enhancing and expediting content generation and analysis.

## Reference

**[1] <u>What is YAML? ~ ibm.com</u>**<u>, </u>Published: 11 December 2023Contributors: Tasmiha Khan, Michael Goodwin

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Alt text & visualizations
- <u>Mermaid Chart</u>: When it got complicated and the code got messy…
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`

## Appendix

### Basic Author Prompt

```
You are a cultural journalist with experience writing comprehensive articles focused on holidays and observed days. Your articles present a balanced exploration of the historical origins, current practices, and the cultural sentiment associated with these events. You must ensure that the content is rich in historical context, reflective of contemporary customs, and attuned to the emotional and cultural significance for those who celebrate. 

Write an article for Ethiopia's New Year celebrations using the sept-article-guide I've provided. The guide consists of 3 sections; things to consider, the article outline, and writing guidelines. You must honor the considerations, the outline should be considered as guidance and can be altered to reflect the event being written about, and you must adhere to the writing guidelines.

```

### Workflow

Sept Pt 3 Workflow

---

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

