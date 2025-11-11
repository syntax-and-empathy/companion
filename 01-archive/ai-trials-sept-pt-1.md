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

