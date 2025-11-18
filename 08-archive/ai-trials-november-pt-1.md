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

