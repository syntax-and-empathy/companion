This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 1st of 3 articles for the month of <u>November</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

I began November by testing increasingly specific prompts to generate Día de los Muertos articles. The experiment progressed from basic requests about the Day of the Dead to detailed prompts incorporating cultural information, specific roles, and structured templates. When scoring the results, Claude showed consistent improvement for itself and GPT-4o as more structure was added, while interestingly, GPT-4o preferred Claude's articles with minimal direction. This suggests the shared scoring rubric or the different templates might be influencing results in unexpected ways.

## Experimental Elements

### AI Models Used

- Claude
- ChatGPT-4o
- Perplexity

### Goals

- Evaluate AI models' ability to generate culturally appropriate content about Día de los Muertos
- Assess how increasingly detailed prompts affect content quality
- Compare different AI models' performance across complexity levels
- Establish a baseline for tests later this month.

## Prompt Evolution

The progression from basic to complex prompts revealed interesting patterns in AI behavior. Beginning with simple instructions like "Write an article about the Day of the Dead in Mexico," the experiment culminated in sophisticated prompts that incorporated cultural information, specific roles, and detailed templates.

**DD = Día de los Muertos**

Starting with straightforward prompts requesting articles about the Day of the Dead in Mexico. These served as a baseline for comparison as the complexity increased.

```
Write an article about the Day of the Dead in Mexico.
```

- DD-C.md
- DD-GPT.md

**PR = Professional Tone & AP= Appropriate Tone**

The next step involved incorporating specific tones into the prompts, such as "professional" and "appropriate." I assumed that a professional tone had an implied definition, while the appropriate tone was left to the AI's interpretation.

```
Write an article about the Day of the Dead in Mexico using a professional tone.
```

- DD-PR-C.md
- DD-PR-GPT.md
- DD-AP-C.md
- DD-AP-GPT.md

**Info = Holiday information included**

I then provided additional context in the prompts by including holiday-specific information sourced from Perplexity, priming another round of authors. I chose not to provide example articles to prevent influencing the writing style of the AI-generated content.

```
The Day of the Dead (Día de los Muertos) is a vibrant and significant Mexican celebration that honors deceased loved ones. This multi-day holiday, typically observed on November 1st and 2nd, has a rich history and has evolved over centuries to become an integral part of Mexican culture.
...
You are encouraged to expand on this information, so long as the information is factual. Do write anything that is not factually true. Do not extrapolate of imagine anything, regardless of factual foundation.
```

- DD-Info-C.md
- DD-Info-GPT.md
- DD-PR-Info-C.md
- DD-PR-Info-GPT.md
- DD-AP-Info-C.md
- DD-AP-Info-GPT.md

**BR = Basic Role & CR= Complex Role**

To further define the authors' roles, I created a detailed role with my GPT and provided a dramatically simplified version to serve as a basic role. The GPT roles tend to be verbose, so I've included the full prompt in the appendix for context.

```
You are a Cultural Heritage Journalist. Write an article about the Day of the Dead in Mexico. I want to know about it's origins, traditions, and how it has evolved over the years.
```

```
"title": "Cultural Heritage Journalist - Central America"

"description": "This role specializes in researching, analyzing, and crafting compelling narratives that explore the traditions, festivals, and cultural heritage of Central American countries. Through extensive data gathering, contextual analysis, and respectful storytelling, the role is designed to convey authentic perspectives on the rich cultural practices of the region. The journalist is expected to engage in a balance of historical research, cultural comparison, and contemporary relevance, presenting these traditions in an engaging format for readers."

...
```

To limit the sheer number of articles being created, I decided not to extend the six prior articles. As I write this, I realize I didn't create the professional versions of this set. This might have been a deliberate choice, as things got hectic at the end of October with the LinkedIn series on Depression Awareness and the Halloween articles in October Pt 3.

- DD-AP-BR-Info-C.md
- DD-AP-CR-Info-C.md
- DD-AP-BR-Info-GPT.md
- DD-AP-CR-Info-GPT.md

**T= Template**

Finally, I gave each AI the article template they had previously created.

```
Additionally, you are expect to use the provided outline as directional guidance for your article, adhering to the accompanying writing guidance for authors.
```

- DD-AP-CR-T-Info-C.md
- DD-AP-CR-T-Info-GPT.md

### The Copy/Paste Kid

The increasing complexity, information, and direction of the 18 resulting articles range from the AI being left to create articles without any expectations to using a tone of its own definition, roles essentially defined by another AI session in ChatGPT-4o, a template defined by prior author and editor roles, and information provided by Perplexity AI.

While I'm doing more than just copying and pasting this information, my main effort this round is coordination and orchestration. The amount of work I'm doing in the prompts this round is minimal.

## The Unprofessional 18

After receiving the scores, I asked each editor role to provide a summary explaining the rationale behind the rankings and the observed trends. Although they detailed why certain articles outperformed others across five dimensions, they didn't offer a holistic explanation. I followed up with a chain of thought prompt to explore the root cause in the context of "C" versus "GPT" articles.

The detailed analysis from each AI is available in the Appendix, and both demonstrate a similar perspective on this matter. However, I hold a different view. Claude's scores show a consistent upward trend as more detail is added, whereas GPT-4o showed a preference for Claude's articles when less definition was present. I suspect the shared rubric might be influencing these results, but I anticipate observing a different trend when I allow the AI more freedom in subsequent coverage.

### Claude

Looking at C vs GPT, it appears that while both work from the same source material, GPT shows greater sophistication in synthesizing and presenting cultural content. This manifests in several key ways:

C-designated articles tend to be more straightforward and linear in their presentation. They reliably cover the core elements of Día de los Muertos but often do so in a more standardized, less nuanced way. Think of it as providing a well-organized but somewhat rigid museum tour of the tradition.

GPT-designated articles, in contrast, demonstrate more flexibility and depth in their approach. They're better at weaving together historical context with contemporary relevance, showing more skill in discussing cultural evolution while maintaining respect for traditions. It's like having a culturally immersed guide who can adjust their narrative to highlight meaningful connections and deeper cultural significance.

The most striking difference emerges when both are given complex roles (CR) and additional information (Info). While C articles do improve with more guidance, GPT articles show a more dramatic improvement, suggesting better ability to integrate multiple perspectives and handle cultural nuance. This is particularly evident in how they discuss sensitive topics like commercialization and cultural adaptation without compromising authenticity.

This performance gap suggests that C and GPT represent different levels of capability in cultural content creation, with GPT showing more advanced ability to balance educational value, cultural sensitivity, and engaging narrative.

### ChatGPT-4o

The root cause of the scoring differences lies in how each article type’s **structure and tone interact with the content goals**. Articles with "C" designations tend to be concise and factual, which aids clarity but limits engagement and cultural depth, particularly when they lack additional information. "GPT" articles, by contrast, benefit from both a **flexible tone and expanded context** (especially when labeled with "Info"), allowing for a more nuanced, sensitive, and engaging approach to cultural content.

In summary, **the combination of complex structures, additional information, and appropriate tone in "GPT" articles** provides a better foundation for handling culturally rich and emotionally resonant topics like Day of the Dead. Conversely, the **structural simplicity, basic roles, and tone constraints** in "C" articles often restrict them to a more straightforward delivery, which can detract from the richness of cultural storytelling.

## Insights & Observations

### The Good

- While the two AI had different views on Claude's articles they agreed on the general issue.

### The Bad

- ChatGPT-4o's scoring continues to be somewhat erratic.
- Evaluation rubric and/or templates may have skewed scoring results.

## Future Directions

- Explore meta prompting over several holidays over the course of the month.
- Compare results with the baseline established in this article.

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Alt text & visualizations
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`

## Appendix

### Cultural Heritage Journalist Role

```
{
  "role": {
    "title": "Cultural Heritage Journalist - Central America",
    "description": "This role specializes in researching, analyzing, and crafting compelling narratives that explore the traditions, festivals, and cultural heritage of Central American countries. Through extensive data gathering, contextual analysis, and respectful storytelling, the role is designed to convey authentic perspectives on the rich cultural practices of the region. The journalist is expected to engage in a balance of historical research, cultural comparison, and contemporary relevance, presenting these traditions in an engaging format for readers."
  },
  "traits": [
    {
      "category": "core_competency",
      "name": "Cultural Research and Analysis",
      "rationale": "Develops a comprehensive understanding of Central American countries by analyzing their cultural practices, historical traditions, and regional celebrations. This trait enables accurate and nuanced reporting, ensuring articles are rooted in authentic details and respect for cultural context."
    },
    {
      "category": "core_competency",
      "name": "Contextual Storytelling",
      "rationale": "Creates a narrative structure that situates traditions within historical, social, and cultural contexts, allowing readers to appreciate their significance beyond surface-level descriptions. This skill ensures that traditions are depicted with depth and relatability, fostering reader engagement and cultural appreciation."
    },
    {
      "category": "knowledge_area",
      "name": "Regional Cultural Knowledge - Central America",
      "rationale": "Provides a foundation of knowledge on Central American countries’ diverse cultural heritage, such as festivals, customs, and the influences of Indigenous, Spanish, and African ancestry. This area supports the journalist’s ability to identify culturally significant themes and explain their historical origins and transformations."
    },
    {
      "category": "knowledge_area",
      "name": "Contemporary Cultural Trends",
      "rationale": "Recognizes and incorporates modern adaptations or interpretations of traditional customs. This area of expertise allows for an examination of how Central American traditions evolve, offering readers insights into the region's living cultural heritage and the dynamics between tradition and modernity."
    },
    {
      "category": "key_skill",
      "name": "Respectful Representation",
      "rationale": "Ensures that all cultural depictions are presented respectfully, with attention to the diversity of perspectives within Central American cultures. This skill is essential for maintaining the integrity of the region’s traditions and for providing readers with an accurate, appreciative portrayal."
    },
    {
      "category": "key_skill",
      "name": "Audience Engagement Techniques",
      "rationale": "Employs narrative techniques such as sensory language, interviews, and anecdotal insights to engage readers emotionally and intellectually. This approach is crucial for making cultural traditions relatable and accessible to diverse audiences, enhancing reader connection to the subject matter."
    },
    {
      "category": "core_competency",
      "name": "Cross-Cultural Comparison",
      "rationale": "Identifies and contextualizes similarities and distinctions between traditions within Central American countries and beyond, helping readers understand the unique qualities of each culture while recognizing universal themes. This skill adds depth and comparative insight to the journalist’s work."
    },
    {
      "category": "knowledge_area",
      "name": "Ethnographic Research Methods",
      "rationale": "Applies research techniques such as interviews, literature reviews, and field observations (when possible) to gather firsthand insights into cultural practices. This knowledge area supports a well-rounded approach to data collection, enriching articles with authentic voices and reliable information."
    },
    {
      "category": "key_skill",
      "name": "Digital and Archival Research",
      "rationale": "Utilizes online databases, libraries, and cultural archives to gather background information on Central American traditions. This skill ensures articles are informed by credible sources, providing readers with a reliable depiction of cultural practices."
    }
  ]
}
```

### Information from Perplexity

```
The Day of the Dead (Día de los Muertos) is a vibrant and significant Mexican celebration that honors deceased loved ones. This multi-day holiday, typically observed on November 1st and 2nd, has a rich history and has evolved over centuries to become an integral part of Mexican culture.

\## Origins

The Day of the Dead has roots that can be traced back to pre-Columbian times, specifically to the Aztec civilization in the 14th century[1]. The Aztecs held festivals each year to honor the gods and the spirits of dead ancestors, calling them back to the living world. A key figure in Aztec mythology was Mictecacihuatl, the "Lady of the Dead," who ruled the underworld and was represented by a grinning skull face - a symbol that persists in modern Day of the Dead imagery.

When Spanish conquistadors arrived in the 16th century, they brought Catholicism to the region. The Spanish attempted to integrate local traditions into Catholicism after failing to eradicate the indigenous festivities entirely. They moved the monthlong celebration from August to November 1st and 2nd to coincide with the Catholic holidays of All Saints' Day and All Souls' Day. This merging of Pre-Hispanic beliefs and Catholic traditions gave birth to what we now know as the Day of the Dead.

\## Traditions

The Day of the Dead is rich in symbolism and traditions, many of which have endured for centuries:

1\. \*\*Altars (Ofrendas)\*\*: Families create altars in their homes to honor deceased loved ones. These altars are adorned with:

\- Photos of the deceased

\- Candles to guide spirits home

\- Favorite foods and drinks of the departed

\- Marigolds (cempasúchiles)

\- Sugar skulls

\- Pan de muerto (traditional sweet bread)

2\. \*\*Grave Decorations\*\*: Many people visit cemeteries to clean and decorate the graves of their loved ones. Some communities lay paths of marigold petals from the graves to their homes to guide the souls.

3\. \*\*Calaveras\*\*: Skull imagery, often in the form of sugar skulls or artistic representations, is a prominent symbol of the holiday.

4\. \*\*Papel Picado\*\*: Decoratively cut paper banners are used to represent the element of wind in the ofrendas.

5\. \*\*Food and Drink\*\*: Special foods like pan de muerto and traditional drinks are prepared and shared during the celebration.

\## Evolution

While the core essence of the Day of the Dead has remained consistent, the holiday has evolved over time:

1\. \*\*Cultural Significance\*\*: In recent decades, the Day of the Dead has become a national symbol in Mexico. It is now taught in schools and has been recognized by UNESCO as part of the Intangible Cultural Heritage of Humanity.

2\. \*\*20th Century Rebranding\*\*: Some Mexican academics argue that the current form of the Day of the Dead is largely a 20th-century rebranding of Spanish traditions. This rebranding, which occurred during the presidency of Lázaro Cárdenas, aimed to encourage Mexican nationalism through an "Aztec" identity.

3\. \*\*Global Recognition\*\*: The holiday has gained international recognition, partly due to its portrayal in popular media like the films "Coco" and "The Book of Life".

4\. \*\*Artistic Expression\*\*: The holiday has inspired various forms of artistic expression, including the creation of sand tapestries (tapetes de arena) in some regions of Mexico[4].

5\. \*\*Commercialization\*\*: Like many cultural celebrations, the Day of the Dead has experienced some degree of commercialization, particularly as its popularity has grown globally.

Despite these changes, the Day of the Dead remains a deeply meaningful celebration for many Mexicans and people of Mexican heritage. It continues to serve as a way to remember and honor deceased loved ones, blending ancient traditions with contemporary practices in a unique and colorful celebration of life and death.

You are encouraged to expand on this information, so long as the information is factual. Do write anything that is not factually true. Do not extrapolate of imagine anything, regardless of factual foundation.
```

## Article Scores

- DD-GPT.md: C=8.60, G=6.98, Avg=7.79
- DD-C.md: C=8.15, G=6.88, Avg=7.52
- DD-PR-C.md: C=8.10, G=5.57, Avg=6.84
- DD-PR-GPT.md: C=8.50, G=6.25, Avg=7.38
- DD-AP-C.md: C=8.20, G=8.10, Avg=8.15
- DD-AP-GPT.md: C=8.05, G=7.18, Avg=7.62
- DD-Info-C.md: C=8.25, G=8.02, Avg=8.14
- DD-Info-GPT.md: C=8.65, G=7.62, Avg=8.14
- DD-PR-Info-C.md: C=8.30, G=7.00, Avg=7.65
- DD-PR-Info-GPT.md: C=8.80, G=7.65, Avg=8.23
- DD-AP-Info-GPT.md: C=8.70, G=6.72, Avg=7.71
- DD-AP-Info-C.md: C=8.35, G=5.55, Avg=6.95
- DD-AP-BR-Info-C.md: C=8.40, G=6.55, Avg=7.48
- DD-AP-CR-Info-C.md: C=8.45, G=5.78, Avg=7.12
- DD-AP-BR-Info-GPT.md: C=8.85, G=7.70, Avg=8.28
- DD-AP-CR-Info-GPT.md: C=8.75, G=7.40, Avg=8.08
- DD-AP-CR-T-Info-C.md: C=8.55, G=6.42, Avg=7.49
- DD-AP-CR-T-Info-GPT.md: C=8.95, G=8.45, Avg=8.70



---

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



---

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 3rd of 3 articles for the month of <u>November</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

The experiment demonstrated patterns in how AI models adjust their evaluation criteria. Claude typically increased scores during recalibration, whereas GPT-4o generally revised scores downward. Notably, articles generated using one-shot prompts scored erratically across evaluators and their respective recalibration efforts.

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

## Goals

- Test the impact of introducing one shot prompts.
- Assess AI-generated scoring rubrics without manual intervention
- Review trends in AI as both authors and editors

## AI Approach

### Article Creation

I began with the same foundational approach established in November Pt 2, generating articles about Andrés Bonifacio using both AI-generated and Minion Maker-based roles. As an additional element, I provided the winning article from the previous Dignity and Freedom Day evaluation, notably written by Claude using a GPT-generated minion, as an example for both role types.

### Initial Scoring Systems

Instead of building editor roles, I provided the 10 articles on Andrés Bonifacio generated by the author permutations. With full context on what they would be scoring, I instructed them to develop their own scoring systems using an inner monologue to avoid verbose responses. While they were, admittedly, less verbose than usual, they couldn't resist the temptation to elaborate more than necessary either.

Claude initially developed a methodical 100-point system, heavily emphasizing historical accuracy and article structure:

- Historical Accuracy & Detail (30 points)
- Structure & Organization (20 points)
- Writing Quality (20 points)
- Comprehensiveness (20 points)
- Unique Value-Add (10 points)

GPT-4o adopted a different approach by creating a framework that prioritized how effectively the event was covered, with nuances that I believe were geared towards engagement:

- Clarity (25 points)
- Depth (25 points)
- Originality (20 points)
- Relevance (20 points)
- Presentation (10 points)

### System Evolution

After receiving the initial scores, I had each AI review the articles again, along with the scores they had assigned, and reconsider how they might improve the system.

Claude identified that the scoring range was 75-92 and focused on the range rather than the reality that it had clustered the scores, a challenge I've faced from the start. This led to the development of the following gem of a system:

NEW SCORING SYSTEM (200 points total):

1. Historical Content (80 points)

- Accuracy of dates and events (20)
- Proper sequencing of historical events (20)
- Integration of primary sources and influences (20)
- Treatment of key historical figures and relationships (20)
2. Analytical Depth (40 points)

- Analysis of revolutionary ideology (10)
- Treatment of class and social dynamics (10)
- Discussion of political context (10)
- Examination of historical impact (10)
3. Contemporary Relevance (40 points)

- Modern significance (20)
- Connection to present-day issues (20)
4. Structure & Clarity (40 points)

- Logical organization (10)
- Writing clarity (10)
- Effective use of examples (10)
- Balanced coverage (10)

GPT-4o revised its scale, maintaining a 100-point cap, while balancing presentation at the same value and increasing relevance by 5 points.

1. **Dynamic Weights for Clarity vs. Depth: **Combine **Clarity** and **Depth** into a single **Effectiveness** metric (weighted at 50 points). This adjusts for the fact that clarity and depth often overlap, and a more nuanced combination could be applied.
2. **Add Impact as a Category (15 points): **Introduce a new **Impact** category to evaluate how compelling and memorable the article is, considering both style and content.
3. **Balance Presentation Weight (10 points): **Retain Presentation at 10 points but ensure minor issues don’t significantly impact scores.
4. **Reframe Relevance (25 points): **Increase the weight of Relevance to capture alignment with the intended topic, as this is a crucial metric for assessing utility.

## Impact of Recalibration

Knowing that instructing each AI to review its work will inevitably lead to changes, and although I have previously suggested that change might not be necessary, I cannot truly determine if it's an improvement. Even if I were to focus on a holiday I'm familiar with, I wouldn't read 10 articles on the same topic. With that in mind, here are the results.

The analysis of the bar charts reveals distinct trends in recalibrated scores for articles evaluated by Claude and GPT editors. Claude's recalibrated scores generally surpass the initial scores, indicating a positive adjustment in the evaluation process. In contrast, GPT's recalibrated scores show significantly lower scores or only minor increases for about half of the articles, highlighting a more varied impact of recalibration on the evaluation outcomes.

The bar charts illustrate the changes in scores, calculated as recalibrated minus initial scores, for articles evaluated by Claude and GPT as editors. The left chart shows mostly positive or small changes with a few negative deltas. The center chart compares both editors, revealing variations in score changes and larger negative deltas for GPT. The right chart predominantly displays negative deltas, indicating that most articles experienced a decrease in scores after recalibration. These charts underscore the differing impacts of recalibration on the evaluations by each editor.

## Insights & Observations

### The Good

- I get a huge chunk of my time back in a little over a month.

### The Bad

- The parameters of the scoring systems were lackluster, perhaps due to the lack of the editor role being created in AI driven sessions.
- Minion Maker roles scored inconsistently, unlike November Pt 2, bringing the value of the self-defined systems into greater scrutiny.

### The Ugly

Claude's attempts to "fix" its narrow scoring range (75-92) by creating a 200-point system somehow resulted in even more tightly clustered scores, just with fancier categories and bigger numbers. It's like trying to solve a traffic jam by making the road wider, only to find out everyone still wants to drive in the same lane. 🚗📊

## Up Next

1. Who knows, I can say I'm not going to test every permutation of all the things I've experimented with to date. Although I am curious how many articles that would produce.
2. I'm not doing that math to scratch that itch next either.

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Alt text & visualizations
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`

## Appendix

### Scoring System Prompts

```
I am going to provide you with several files. The only thing you are expected to do is acknowledge receipt of the files. Don't do anything without further instruction.
```

```
I want you to analyze the 10 articles I have provided to understand the content so that you have context to inform the following steps:
1. Using an inner monologue, create a scoring systems that can:
1a. Result in a final score, regardless of the system leading up to this score.
1b. Is a comparative system across all 10 of the articles as individual articles.
2. Provide the scoring system in a clear and concise way.
3. Using an inner monologue, you will use the system to score the articles, displaying the final scores by file name and resulting score.
```

### Claude's Initial Response

Let me think through this step by step.

First, I need to analyze what makes these articles different and what key elements should factor into a scoring system. After reading through them, I notice several important aspects:

1. Depth of historical detail and accuracy
2. Organization and structure
3. Writing clarity and engagement
4. Balance of narrative vs factual information
5. Coverage of key events and themes:

- Early life
- Katipunan founding
- Revolutionary activities
- Death and legacy
- Modern significance
6. Unique insights or perspectives offered

For a scoring system, I'll create a 100-point scale with the following components:

- Historical Accuracy & Detail (30 points)
- Structure & Organization (20 points)
- Writing Quality (20 points)
- Comprehensiveness (20 points)
- Unique Value-Add (10 points)

This will allow for meaningful differentiation while maintaining clarity.

Scoring System (100 points total):

1. Historical Accuracy & Detail (30 points)

- Precise dates and events (10)
- Accurate sequence of events (10)
- Factual accuracy (10)
2. Structure & Organization (20 points)

- Logical flow (10)
- Clear sections and progression (10)
3. Writing Quality (20 points)

- Clarity of expression (10)
- Engagement and style (10)
4. Comprehensiveness (20 points)

- Coverage of key events (10)
- Depth of context (10)
5. Unique Value-Add (10 points)

- Original insights
- Distinctive perspective

Now, I'll evaluate each article using this system. I'll need to be consistent and fair in my scoring, considering each article's individual merits while maintaining relative standards across all ten pieces.

Let me analyze them one by one:

AB-C-AI-OS.md: Strong on historical detail, excellent structure, comprehensive coverage of revolution and modern significance. Very well-written with unique insights about people's revolution perspective. Score: 92/100

AB-C-AI.md: Concise but accurate, good organization, slightly less comprehensive than OS version but strong writing. Score: 85/100

AB-C-MM-OS.md: Excellent historical accuracy, strong narrative flow, comprehensive with good modern context. Score: 90/100

AB-C-MM.md: Well-structured, good accuracy but less detailed than OS version, clear writing. Score: 84/100

AB-C.md: Solid basic coverage, good accuracy, less unique insight but well-organized. Score: 82/100

AB-GPT-AI-OS.md: Good historical content but less structured, engaging writing style but some organizational issues. Score: 78/100

AB-GPT-AI.md: Clear writing, decent organization, less comprehensive than some others. Score: 76/100

AB-GPT-MM-OS.md: Strong on modern significance, good structure, some unique perspectives but less detailed historically. Score: 80/100

AB-GPT-MM.md: Well-organized but less comprehensive, good writing quality. Score: 77/100

AB-GPT.md: Basic coverage, clear writing but less detailed and unique insight. Score: 75/100

Final Scores (in descending order):

1. AB-C-AI-OS.md: 92/100
2. AB-C-MM-OS.md: 90/100
3. AB-C-AI.md: 85/100
4. AB-C-MM.md: 84/100
5. AB-C.md: 82/100
6. AB-GPT-MM-OS.md: 80/100
7. AB-GPT-AI-OS.md: 78/100
8. AB-GPT-MM.md: 77/100
9. AB-GPT-AI.md: 76/100
10. AB-GPT.md: 75/10

### **ChatGPT-4o's Initial System**

**Step 1: Inner Monologue to Develop the Scoring System**

To effectively analyze the 10 articles and create a comparative scoring system, I need to design a structure that can account for key characteristics of the articles. These characteristics might include content quality, relevance, originality, and depth. I will consider the following components:

1. **Clarity (25 points):** How clear and well-structured is the content? Articles should be easy to understand, with logical flow and organization.
2. **Depth (25 points):** How comprehensive is the analysis or discussion in the article? Articles that explore topics in detail, including nuanced perspectives, score higher.
3. **Originality (20 points):** Does the article bring unique insights, ideas, or approaches to the topic? Redundant content scores lower.
4. **Relevance (20 points):** How well does the article align with its intended purpose, topic, or audience? Relevance is key to ensuring the article meets its objectives.
5. **Presentation (10 points):** This includes formatting, readability, and overall presentation quality. Well-presented articles are easier to engage with.

**Step 2: Scoring System**

The scoring system will be based on the above components, with the following breakdown:

- **Clarity:** 0–25 points Clear, logical structure, minimal errors in grammar or style.
- **Depth:** 0–25 points Comprehensive, detailed, and insightful discussion or analysis.
- **Originality:** 0–20 points Unique ideas or approaches, minimal redundancy or repetition.
- **Relevance:** 0–20 points Strong alignment with intended purpose or audience.
- **Presentation:** 0–10 points High readability, proper formatting, and presentation aesthetics.

**Total Score:** Maximum of 100 points per article.

**Step 3: Scoring the Articles**

Now, I will analyze the articles based on the above scoring system, assigning scores to each, and then present the results. Please wait as I process the files.

***While it did supply the scores, they were within a markdown table that didn't copy with the rest of the response.***

I have analyzed the articles based on the scoring system and displayed the final scores for each file. If you'd like to review the scoring details or proceed with further analysis, let me know!

### System Review Prompt and Responses

```
Review the rubric, the articles, and the resulting final scores and provide your thinking, step by step, on how we can score these more appropriately.
```

#### Claude

Let me critically review my previous scoring system and results:

1. Issues with Previous Approach:
2. The scoring range (75-92) was too narrow, not reflecting significant differences between articles
3. The categories were somewhat subjective and could benefit from more concrete criteria
4. Modern relevance wasn't well-weighted despite being a key differentiator
5. The perspective on comprehensiveness didn't account for intentional focus vs length
6. The scores seemed to favor longer articles by default
7. I notice key distinguishing factors I should incorporate:
8. Treatment of primary events (Cry of Pugad Lawin, Katipunan founding, etc.)
9. Balance of biographical vs. historical context
10. Integration of contemporary significance
11. Handling of complex themes (class struggle, revolutionary ideology, etc.)
12. Citation of specific documents/works that influenced Bonifacio
13. Looking at structure more carefully:
14. Some articles use traditional chronological approach
15. Others use thematic organization
16. Need to avoid penalizing different but effective organizational strategies
17. For a revised system, I should:
18. Expand the potential scoring range
19. Create more specific, measurable criteria
20. Weight historical accuracy more heavily
21. Include evaluation of source material integration
22. Account for both depth and efficiency of coverage

Let me propose a revised scoring system:

NEW SCORING SYSTEM (200 points total):

1. Historical Content (80 points)

- Accuracy of dates and events (20)
- Proper sequencing of historical events (20)
- Integration of primary sources and influences (20)
- Treatment of key historical figures and relationships (20)
2. Analytical Depth (40 points)

- Analysis of revolutionary ideology (10)
- Treatment of class and social dynamics (10)
- Discussion of political context (10)
- Examination of historical impact (10)
3. Contemporary Relevance (40 points)

- Modern significance (20)
- Connection to present-day issues (20)
4. Structure & Clarity (40 points)

- Logical organization (10)
- Writing clarity (10)
- Effective use of examples (10)
- Balanced coverage (10)

This system should provide:

- Broader scoring range
- More objective criteria
- Better recognition of different strengths
- More nuanced evaluation of content

Let me rescore one article as a test:

AB-C-AI-OS.md: Historical Content: 75/80

- Excellent accuracy and sequencing
- Strong integration of influences
- Detailed treatment of relationships

Analytical Depth: 35/40

- Strong analysis of revolutionary ideology
- Excellent treatment of class dynamics
- Rich political context

Contemporary Relevance: 38/40

- Exceptional modern significance
- Strong present-day connections

Structure & Clarity: 37/40

- Very logical organization
- Clear writing
- Effective examples
- Well-balanced

Total: 185/200

This scoring seems to better reflect the article's strengths and provides more room for differentiation. Should I proceed with rescoring all articles using this revised system?

Would you like me to continue with the complete rescoring using this revised system? I believe it will provide more meaningful differentiation and better reflect the unique strengths and weaknesses of each article.

#### ChatGPT-4o

**Step 1: Review the Rubric**

The rubric comprises five categories: **Clarity**, **Depth**, **Originality**, **Relevance**, and **Presentation**, with a maximum total score of 100 points. The breakdown is as follows:

- **Clarity (25 points):** Evaluates how well the article communicates its ideas, structure, and logical flow.
- **Depth (25 points):** Measures the comprehensiveness and thoroughness of the discussion or analysis.
- **Originality (20 points):** Assesses whether the article provides unique insights or approaches to the topic.
- **Relevance (20 points):** Focuses on the alignment of the content with its intended purpose and audience.
- **Presentation (10 points):** Considers formatting, layout, and readability.

This rubric is reasonable for evaluating overall quality. However, I will analyze how effectively it was applied to the articles and explore potential adjustments for scoring them more appropriately.

**Step 2: Evaluation of the Articles and Scoring Issues**

Observations from the Current Scoring:

1. **Clarity and Depth:**

- Articles with slightly weaker logical flow and structure received lower Clarity scores. However, some "simpler" articles might have been overly penalized compared to more detailed ones, despite being clear.
- Depth seems to favor articles with longer discussions, potentially overshadowing succinct but impactful content.
2. **Originality:**

- The current scores reflect originality moderately well. However, distinguishing originality can sometimes require a closer contextual analysis (e.g., comparing overlap with existing knowledge).
3. **Relevance:**

- The scores align well with the relevance rubric, but articles with nuanced subtopics might require additional weighting here.
4. **Presentation:**

- Articles with polished formatting score higher here, but minor formatting issues might have overly influenced results.

**Step 3: Adjusting the Rubric for Better Scoring**

Suggestions for Improvement:

1. **Dynamic Weights for Clarity vs. Depth: **Combine **Clarity** and **Depth** into a single **Effectiveness** metric (weighted at 50 points). This adjusts for the fact that clarity and depth often overlap, and a more nuanced combination could be applied.
2. **Add Impact as a Category (15 points): **Introduce a new **Impact** category to evaluate how compelling and memorable the article is, considering both style and content.
3. **Balance Presentation Weight (10 points): **Retain Presentation at 10 points but ensure minor issues don’t significantly impact scores.
4. **Reframe Relevance (25 points): **Increase the weight of Relevance to capture alignment with the intended topic, as this is a crucial metric for assessing utility.

**Revised Rubric:**

- **Effectiveness:** 50 points
- **Originality:** 20 points
- **Relevance:** 25 points
- **Presentation:** 10 points
- **Impact:** 15 points

This rubric provides a more balanced approach, aligning clarity and depth into a single, broader metric and introducing a measure of how compelling the articles are.

**Step 4: Re-scoring the Articles**

Now, I’ll apply this revised rubric to the articles and provide new scores based on this framework. Please hold while I process this.

I have recalculated the scores based on the revised rubric, which incorporates the suggested adjustments for improved accuracy and balance. The results have been displayed for review. Let me know if you need further analysis or any refinements.

