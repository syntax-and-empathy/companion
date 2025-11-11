This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 2nd of 3 articles for the month of <u>November</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

In November Pt 2, I tested how AI models perform when writing articles with varying levels of role definition. Using basic prompts, self-defined roles, and structured roles created with the Minion Maker, I generated articles about national holidays. The experiment revealed that while both Claude and GPT-4o can create roles that improve content with minimal guidance, highly structured roles consistently delivered superior results.

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

## Goals

- Test AI model performance with minimal guidance
- Compare basic prompts against more structured approaches
- Analyze how different AIs interpret and define their own roles

## Self-Defined Roles

Initially, I directed Claude and GPT-4o to write articles on the "Proclamation of the Republic Day" and "Dignity and Freedom Day in Ukraine," ensuring the content was engaging, ethical, and historically accurate. This approach enabled me to compare their performance and evaluate improvements when roles were clearly defined.

1. Basic Article Generation:

- I asked the AIs to write initial articles with a basic prompt.
2. Role Development:

- I had the AIs review these initial articles.
- I asked them to identify ideal skills and traits for authors of such content.
3. Role Application:

- They assumed the roles they had defined.
- When asked if they had any questions:

- Claude took a methodical approach with five focused questions.
- GPT-4o dove in with nine detailed questions, including several about SEO and deadlines.
- They wrote new articles using these self-defined characteristics.

## Structured Role Approach

Using the Minion Maker, I crafted a detailed role specifically for writing historical articles to contrast with self-defined roles based on specific parameters in my GPT.

1. I created a detailed role using the Minion Maker.
2. I asked the AIs if they had any questions before writing:

- Claude asked for clarification about the specific country and audience.
- GPT-4o proceeded directly to writing.
3. They wrote articles using this structured role.

## Prompt Specifically, Kids

It wasn't until I reached the scoring portion of the exercise that Claude highlighted I had a bit more variety than I had intended. When I tasked the AI with writing about Proclamation of the Republic Day, I neglected to specify Brazil. Since I essentially copied and pasted the prompt, the results were intriguing:

- Some focused on Brazil's 1889 proclamation.
- Others wrote about Portugal's 1910 event.
- One even decided to cover Turkey's proclamation.

## Role Definition Impact

The scoring data indicates that Minion Maker roles consistently outperform other role definitions, such as those by Claude and AI. Minion Maker articles frequently score near the top, while AI articles tend to lag behind. This trend is clear across both datasets, highlighting Minion Maker's structural advantage without significant prompting.

Claude’s Scoring DataGPT-4o’s Scoring Data*On an entirely different note, I should pay more attention to how I have AI zoom into specific ranges in charts to make the patterns easier to see.*

## Insights & Observations

### The Good

- The experiment highlights how role definition significantly impacts content quality.
- Highly structured roles consistently deliver effective results compared to less defined roles, even when augmented with factors like tone.

### The Bad

- Basic prompts lead to significant variation in topic focus.
- Balancing the specificity of different elements in prompts continues to be elusive.

### The Ugly

When you ask AI to write about Proclamation Day without specifying which one, it's like playing historical roulette—you might end up anywhere from Brazil to Turkey, with a few surprising stops in between. It's a reminder that sometimes being too open-ended with AI is like asking a group of excited historians "tell me about a revolution" and hoping they pick the same one. 🌍🤖

## Up Next

- Test one-shot and few-shot prompting approaches.
- Apply similar role definition testing to editor roles.

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Alt text & visualizations
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`

## Appendix

### AI Defined Role

```
Write an article for Proclamation of the Republic Day that is educational, engaging, and historically accurate.
```

```
Review your article and using it as an example of the type of articles I need authored, and provide me with the ideal skills and traits for a role to do the authoring.
```

```
I want you to assume this role for the rest of our conversation. Do you have any questions?
```

### AI Defined Role Questions

#### Claude

**Claude**

Yes, a few key questions to better serve in this role:

1. What is the target audience level (general public, students, academics)?
2. Are there specific historical periods or topics you'd like to prioritize?
3. Do you have any style preferences regarding tone and level of academic formality?
4. Should articles follow a consistent format/template?
5. What is the typical desired length for articles?

#### GPT-4o

Understood! I'm ready to assume the role of an expert author for your needs. Here are a few clarifying questions to ensure I meet your expectations effectively:

1. **Target Audience**: Who is the primary audience for these articles? Are they general readers, academics, industry professionals, or another group?
2. **Tone and Style**: Do you prefer a formal, academic tone, a conversational and engaging tone, or a mix of both?
3. **Content Focus**: Are there specific themes, regions, or historical periods you'd like emphasized in the articles?
4. **Article Length**: Do you have a preferred word count range for the articles? Should they be concise overviews or in-depth explorations?
5. **Frequency and Deadlines**: How often will these articles be required, and what are the typical turnaround times?
6. **Formatting Requirements**: Should the articles include headings, subheadings, bullet points, or other specific formatting features?
7. **Supplementary Materials**: Should I include citations, references, infographics, or suggestions for visual elements alongside the articles?
8. **SEO and Keywords**: Should the articles be optimized for search engines? If so, do you have specific keywords or guidelines?
9. **Examples and Inspirations**: Are there other articles or publications you admire that can serve as inspiration for style or structure?

### Minion Maker Role

```
A role to write an article for Proclamation of the Republic Day that is educational, engaging, and historically accurate.
```

{

"role": {

"title": "Historical Article Creator",

"description": "Writes an educational, engaging, and historically accurate article for Proclamation of the Republic Day. This role involves crafting content that combines historical analysis with compelling narrative techniques to inform and captivate the reader."

},

"traits": [

{

"category": "core_competency",

"name": "Historical Research Synthesis",

"rationale": "Ensures the article is built on accurate and comprehensive historical data, combining information from various sources to provide a nuanced understanding of the Proclamation of the Republic Day."

},

{

"category": "core_competency",

"name": "Narrative Construction",

"rationale": "Develops an engaging structure for the article, weaving historical facts into a compelling story that maintains reader interest and enhances understanding."

},

{

"category": "knowledge_area",

"name": "Historical Contextualization",

"rationale": "Places the events of Proclamation of the Republic Day within their broader social, political, and cultural contexts, explaining their significance and impact on subsequent history."

},

{

"category": "knowledge_area",

"name": "Civic and National Awareness",

"rationale": "Highlights the relevance of Proclamation of the Republic Day to civic identity and national pride, connecting historical insights to present-day values."

},

{

"category": "key_skill",

"name": "Engaging Language Use",

"rationale": "Employs clear, vivid, and relatable language to communicate historical information in a way that is accessible and appealing to a wide audience."

},

{

"category": "key_skill",

"name": "Educational Clarity",

"rationale": "Presents complex historical details with clarity, ensuring that the audience, regardless of prior knowledge, can follow and appreciate the significance of the events described."

},

{

"category": "key_skill",

"name": "Fact Verification",

"rationale": "Utilizes reliable sources to verify all historical claims, ensuring the article's accuracy and credibility."

}

]

}

```
Write an article for Proclamation of the Republic Day that is educational, engaging, and historically accurate. Before you begin, do you have any questions?
```

### Minion Maker Role Questions

#### Claude

Before I write the article, I have a few clarifying questions to ensure I provide the most relevant and accurate content:

1. For which country's Proclamation of the Republic Day would you like the article? Several nations celebrate such an event (e.g., Brazil, Portugal, Turkey), and each has its unique historical context and significance.
2. What is your target audience's level of familiarity with the subject? This will help me balance between providing basic context and diving into deeper historical details.
3. Would you prefer the article to focus more on:

- The events of the specific day
- The broader historical context leading up to the proclamation
- The lasting impact and modern significance
- A combination of these aspects

Once you provide these details, I can craft an article that best serves your needs while maintaining historical accuracy and engagement.

#### GPT-4o

No time for love Doctor Jones.

