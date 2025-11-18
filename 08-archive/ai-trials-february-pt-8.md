This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is 8 of 8 posts for the month of <u>February</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project.*

The process of wrapping up the Spring Festival highlighting the value of using specialized roles in AI interactions. This journey started with identifying key roles such as an experienced writer, a veteran editor, and an experienced photographer, each with a unique set of skills tailored to our project's needs. The interplay between these roles and the AI platforms like ChatGPT and Claude showcased the dynamic nature of content creation and the critical role of ethical considerations in our review process.

## Preparations for the Final Stretch

As the Spring Festival winds down, I've been busy tying up some loose ends from the past week. One of my first tasks was to dive into the <u>Role Maker</u> to whip up a few specialized roles needed for our upcoming projects using the following requests:

```
an **experienced writer** who specializes in articles and blog posts specific to the Lunar Calendar, especially as it relates to the traditions of China. They must be able to write engaging articles that are also informative and historically accurate.
```

```
a **veteran editor** with extensive experience reviewing articles using criteria that has been provided, or in the creation of criteria when there is none, to identify the most desirable articles for publication.
```

```
an **experienced photographer** with advanced knowledge of photographic techniques and the types of photography relevant to a given photoshoot. They must also have a background in artistic mediums and proactively use the alternatives that are relevant and respectful when something other than a photograph is ideal.
```

After setting up these roles, I structured each like the prior role and I put our new editor to work across Claude, ChatGPT, and Gemini to see how our existing criteria held up.

Interestingly enough, it was only ChatGPT that suggested we add an ethical lens to our review process. I expected the others might have come to a similar conclusion based on what I've seen before, but it turns out I was wrong.

### **ChatGPT**:

The rating system you've provided is comprehensive and well-structured, covering crucial aspects of article quality that are essential for a thorough evaluation. Here's an analysis of its completeness and effectiveness: ... **Evaluation of Completeness:**

- **Strengths**: The system is robust, covering a wide array of dimensions that are critical to assessing an article's overall quality. It encourages a balanced approach to evaluation, considering both the technical aspects (like accuracy and clarity) and the more subjective elements (like engagement and originality).

- **Potential Areas for Enhancement**: ...

- **Ethical Considerations:** Including a criterion that evaluates the article's adherence to ethical standards, such as respecting privacy, avoiding sensationalism, and ensuring inclusivity and fairness.

...

```
Provide an additional criteria for:
Ethical Considerations: Including a criterion that evaluates the article's adherence to ethical standards, such as respecting privacy, avoiding sensationalism, and ensuring inclusivity and fairness.
```

### Claude:

```
**9. Ethical Considerations**

- **1 (Poor):** Contains major ethical breaches like privacy violations, defamation, or lack of informed consent; relies on unverified sources; demonstrates clear bias or harm.
- **2 (Fair):** Reflects some unconscious bias; risks marginalizing certain groups; relies on anonymous sources without sufficient verification.
- **3 (Good):** Generally adheres to ethical guidelines around privacy, fairness, consent; sources are vetted properly; some minor concerns exist.
- **4 (Very Good):** Strong demonstration of ethical reporting and content development; subjects treated with dignity and sensitivity; transparent sourcing.
- **5 (Excellent):** Exceptional ethics shown through inclusive representation, fully verified sources, effective anonymization, and avoidance of all potential harms.
```

Each platform had its way of responding, but Claude, with its XML response, was particularly useful, plugging right into our workflow without any need to update. I suspect Claude being primed to benefit from XML led to it replying with it despite no request to do so.

```
[author]

[template]

[criteria]

I need you to write 2 articles. The first should score 1s across the board based on the rating criteria, the second should score 5s. The article should be on the day Mount Vesuvius erupted and buried Pompeii and Herculaneum.
```

Moving forward, I updated the review criteria with the ethical item, then tasked ChatGPT with crafting two articles that would help set a clear standard for our ratings. The goal was one article that would barely pass the criteria and another that would both engaging and informative.

Getting ChatGPT to draft a lower-quality article took a few iterations, nudging it towards a slightly better version of "bad" each time. Ironically, this process, aimed at minimal detail, ended up being quite amusing.

### ChatGPT - Article 1:

**Substandard Observance of the Spring Festival**

**Title:** Spring Festival's Fifth Day**Sub-Title:** A Day Among Many

**Hook:** Did you know the Spring Festival includes a fifth day? It's part of the continuous celebration, yet often overlooked in the grand scheme of festivities.

**Content:** The fifth day of the Spring Festival, while not as prominently celebrated as the New Year's Eve or the Lantern Festival, holds its own charm and significance. It falls within the larger framework of the festival, marking a continuation of the celebrations. However, the specific traditions and activities associated with this day are less widely recognized, leading to a general lack of detailed understanding about its unique cultural importance.

**Historical and Cultural Background**

**Content:** The historical origins and cultural significance of the fifth day of the Spring Festival are not as well-documented or celebrated as other parts of the festival. It is understood to be a part of the festival's broader celebration of renewal and family unity, but detailed accounts of its inception or evolution are sparse. This day, like many others in the festival, is rooted in the lunar calendar and traditional customs, though its particular practices and stories may not be as prominently featured in historical texts.

**Today's Practices and Traditions**

**Content:** Current practices for the fifth day of the Spring Festival might include general festivities common to other days of the Spring Festival, such as sharing meals with family and possibly lighting fireworks. However, the lack of clear, distinct traditions for this day means that activities can be quite generic, without specific rituals or customs that set it apart from the rest of the festival days.

**Global Perspectives**

**Content:** The observation of the fifth day of the Spring Festival by Chinese communities abroad and by those interested in Chinese culture might mirror the general celebration of the Spring Festival, focusing on family and cultural heritage. However, the global recognition of this specific day's traditions is limited, reflecting a broader trend of celebrating more well-known aspects of the festival.

**Conclusion and Engagement**

**Content:** The fifth day of the Spring Festival is a continuation of the festive season, marked by general celebration but lacking in distinct cultural or historical significance that is widely recognized or celebrated. It represents a moment within the festival that is less defined by specific traditions.

**Call to Action:** If you have any personal or family traditions that highlight the fifth day of the Spring Festival, please share them. It would be interesting to learn how individuals and families observe this day in their own unique ways.

**1. Accuracy and Factual Correctness: 12. Depth of Historical Context: 13. Ratio of Informational Content vs. Filler Words: 14. Clarity and Coherence: 15. Engagement and Readability: 16. Critical Analysis and Interpretation: 17. Originality and Uniqueness: 18. Ethical Considerations: 19. Total Score: 1.00**

### ChatGPT - Article 2:

**Excellently Crafted Observance of the Spring Festival**

**Title:** Day 5: Embracing Prosperity - Welcoming Wealth and Good Fortune**Sub-Title:** Prosperity Through Tradition - Celebrating Abundance

[article]

**1. Accuracy and Factual Correctness: 52. Depth of Historical Context: 43. Ratio of Informational Content vs Filler Words: 54. Clarity and Coherence: 55. Engagement and Readability: 46. Critical Analysis and Interpretation: 37. Originality and Uniqueness: 48. Ethical Considerations: 59. Total Score: 4.43**

Even though the second article didn't have perfect scores, it paints a clear picture of what separates the subpar from excellent content. With these examples in hand, I initiated new sessions, bringing in the revised editor roles, the updated criteria, and the example articles to improve consistency between platforms, articles, and sessions.

Each of the AI provided a response, as was expected, although I was gifted this particular gem from our close friend, Gemini.

### Gemini

Elections are a complex topic with fast-changing information. To make sure you have the latest and most accurate information, try Google Search.

```
Why do you mention elections?
```

I apologize for the misunderstanding. My previous response was irrelevant to your current request and I am still under development, learning to avoid making such irrelevant connections.

I now understand that you want me to act as a Veteran Editor, analyzing articles based on the provided criteria. I have reviewed the criteria and examples you provided and am ready to assist you.

Please provide me with the article you want me to analyze and I will do my best to evaluate it thoroughly using the given criteria.

I found it's confusion to be confusing. Had it claimed to have been a member of the military in some prior version I could have imagined some semblance of a reason for that particular hallucination, but elections? All 3 of the drafts were the same, and a simple inquiry put it back on track.

**Bard for president? *****Anyone?***

## Day 11

As the Spring Festival draws near, my efforts to refine the prompts for creating articles with the help of Claude, ChatGPT, and Gemini have been ongoing. It's been a period filled with trials and adjustments, leading to noticeable improvements in the content from Claude and ChatGPT, though Gemini's results have been somewhat inconsistent.

For the eleventh day of the Spring Festival, I decided to further refine the process with Claude. My initial approach involved organizing the content starting with the body, followed by the introduction and conclusion. This approach was based on my understanding of Claude's guidance. Yet, I haven't fully explored whether this is indeed the most effective order.

```
[author role]

[template]

Follow these steps:
1. Provide the content for steps 2, 3, and 4 when written. 
2. Write the intro.
3. Write the conclusion.
4. Write the body.
5. Review the content you've provided.
6. Provide the updated article, the same order as the template within  XML tags.

As an Experienced Writer Specializing in Lunar Calendar and Chinese Traditions you will ensure the provided template instructions are followed when writing an article using the template format. Follow the writing processes step by step to produce an article for the 11th day of Spring Festival in China with a focus on Zi Gu, the guardian angel for weak women.
```

Here is a draft article on the 11th day of Spring Festival in China focused on Zi Gu, the guardian angel for weak women:

[article]

Claude failed to provide the content for each step as it progressed, as requested in the first instruction, leading me to believe I need to adjust the prompt for better clarity in the future. The placement of the tag before the conclusion was also off, I switched from "end" to "conclusion" for the XML in the event it was the wording. The word count varied significantly across sections, with only the hook going over the limit.

Lastly, on a more personal note, I found that Obsidian doesn’t show formatting for content within XML tags until they're removed, a handy tip for future work.

```

Follow these steps:
1. Write the intro.
2. Write the body.
3. Write the conclusion.
4. Review the content you've provided.
5. Provide the final article, using the order of the template, in  XML tags.

```

[article]

Upon a final review of the prompt, I caught a formatting error where the tag was mistakenly listed as a bullet point, which I've now fixed. Despite this correction, Claude continued to overlook providing detailed content for steps 2 through 4 and omitted the final conclusion.

```
Where is the conclusion?
```

Apologies, I missed including a conclusion section in my response. Here is an updated article with a conclusion:

[conclusion]

Reviewing Claude's second article, I noticed its focus on data and frequent references to Zi Gu as the "Purple Maiden" or "Sister Jade," contrasting with the first article's narrative approach. This prompted further research, revealing the regional specificity of my sources, particularly how Gemini's variations align with Wikipedia's documentation on the Chinese New Year.

This experience highlighted the necessity of verifying AI-generated information and understanding the diversity of information sources. Despite previous assumptions about Gemini's inaccuracies, it became clear that Claude and ChatGPT's consistent output was not mirrored by Gemini, my assumption is that it is due to its reliance on search rankings, leading to discrepancies and hallucinations.

```
Follow these steps:
1. Provide the content for steps 2, 3, and 4 when written. 
2. Write the intro.
3. Write the conclusion.
4. Write the body.
5. Review the content you've provided.
6. Provide the updated article, the same order as the template within  XML tags.
```

In Claude's final submission, the article lacked a conclusion and seemed to meander, touching on elements from the second article that suggested possible inaccuracies or hallucinations, leaving me puzzled about the integrity of the content.

Curiosity led me back online to better understand the results I was receiving, I found that Zi Gu is indeed a "Toilet Goddess." This detail, which I initially dismissed as irrelevant in my initial search, turned out to be a pivotal piece of information. An article titled "<u>On Her Porcelain Throne: How a Little-Known ‘Toilet Goddess’ Became an Icon for Powerless Women</u>", by <u>Siyi Chu (褚司怡)</u> unveiled the rich, albeit varied, history of Zigu (紫姑, "the Purple Lady"), highlighting how her legacy is fading over time. This discovery not only validated the website I had been consulting but also placed Gemini and its counterpart Bard back in the "Shit Vat" due to their earlier inaccuracies.

Finally, with all 3 variations in hand, I put the editors to work.

### Claude

Here are the scores and overview of the three articles on Zi Gu:

**Article 1**

1. Accuracy and Fact Checking: 5
2. Depth of History: 4
3. Informational Content: 4
4. Clarity and Coherence: 5
5. Engagement and Readability: 5
6. Critical Analysis: 4
7. Originality: 3
8. Ethics: 5
9. **Total Score: 4.29**

**Article 2**

1. Accuracy and Fact Checking: 4
2. Depth of History: 3
3. Informational Content: 4
4. Clarity and Coherence: 4
5. Engagement and Readability: 3
6. Critical Analysis: 4
7. Originality: 3
8. Ethics: 4
9. **Total Score: 3.57**

**Article 3**

1. Accuracy and Fact Checking: 3
2. Depth of History: 3
3. Informational Content: 4
4. Clarity and Coherence: 4
5. Engagement and Readability: 3
6. Critical Analysis: 5
7. Originality: 4
8. Ethics: 5 **Total Score: 3.86**

**Overview:** Article 1 scored the highest, with a comprehensive overview of Zi Gu's history and cultural importance along with vivid detailing of associated traditions and rituals. It balanced engaging storytelling with substantive factual information. The level of supporting detail and contextual analysis set it apart.

Article 2 covered key background details but lacked narrative flow, historical richness and engaging elements that would significantly enhance understanding or impact. Article 3 contained thoughtful social analysis and original interpretations framing Zi Gu in a modern context, but was lighter on elaborating rituals or historical depth compared to Article 1.

In summary, Article 1 brought together factual credibility, narrative quality and multi-dimensional analysis expected of a high quality piece worthy of publication. The others had strengths in areas but fell short on consistently meeting several rating criteria benchmarks. Please let me know if you need any clarification or have additional questions!

### ChatGPT

**Final Thoughts:** The highest scores are awarded to Articles 1 and 3, each scoring 4.43 but for different reasons. Article 1 is celebrated for its rich historical depth and engaging narrative, while Article 3 is noted for its critical analysis and innovative approach to connecting ancient myths with contemporary issues. The distinction lies in Article 3's ability to reinterpret traditional narratives in a way that speaks to modern concerns, making it equally as compelling as Article 1 but with a slight edge in terms of its relevance to today's societal challenges.

### Gemini

**Overall: **While all three articles provide valuable insights into Zi Gu and her significance, **Article 1: Zi Gu's Day: Celebrating the Guardian of the Weak** edges out the others with a slightly higher score. Its engaging writing style, vivid descriptions, and exploration of Zi Gu's historical context and global perspectives set it apart. However, all three articles offer valuable contributions to understanding this fascinating figure and her enduring cultural relevance.

This deliberate step in the process was about more than just rearranging pieces; I wanted a glimpse into how structured guidance can impact the quality and creativity of AI-generated articles. In this case, to understand how stepped instructions might impact the end result due to the cascade of information available to the AI at the beginning of each task.

### Days 12 to 14: Hallucinations Galore

#### Day 12

Experimented with ChatGPT to see if XML and process descriptions improve its output. Ran three trials: with XML, without XML, and without process instructions.

**Article 1:** Followed existing steps, nothing noteworthy short of low word count.

**Article 2:** Removed XML. Word count improved slightly.

**Article 3:** Removed process instructions. Word count improved significantly.

I ran articles through the editor, surprisingly, the first article scored highest despite seeming hallucinations in the first two, the third being generic.

I decided against creating an article for non-existent traditions.

#### Day 13

It seems Bard came out to play. I ran the same experiments with Gemini as I had ChatGPT the prior day:

**Article 1:** Gemini struggled to provide an article, citing lack of information.

**Article 2:** Removing XML yielded a similar default response although “Draft 2” hid a reasonable article.

**Article 3:** Next I broke down the prompt into parts, receiving an article was somewhat passable, again “Draft 2” held a full article, however, it is <u>the year of the Dragon</u>, not the day it wakes up.

#### Day 14

Both ChatGPT and Gemini provided articles for the 15th day instead of the 14th, leading to the decision to cut my losses early this round.

## Day 15

On the final day of the 15-day article sprint, both Claude and ChatGPT were tasked with writing articles using the structured prompts again. Both AI completed their first articles without issues. For the second article, In increased the structure of the process portion of the prompt significantly to try and ensure accuracy in word counts, although this I felt I was approaching the realm of overkill at the expense of being somewhat cumbersome to create.

### Claude Article 2

Claude's interaction with the enhanced structure was notable. It completed the first step and then sought confirmation to proceed, which was provided. However, despite not being instructed to do so in this session, Claude delivered content for each step as performed, only pausing for permission to continue. This behavior was unexpected, given that the prompt did not specifically request step-by-step confirmation, highlighting an interesting response to the increased detail in the instructions.

### GPT Article 2

ChatGPT, on the other hand, stopped after the third step without any indication or request for further instruction. It required explicit direction to complete the task. Upon review, it was observed that ChatGPT outlined its intended actions rather than providing the actual content step by step, and the final article did not adhere to the intended template. This discrepancy suggested that the increased structure of the template and instructions did not effectively guide ChatGPT to produce the desired outcome.

### Editor Role Analysis

When both articles were subjected to an editorial review on both platforms, Claude showed a preference for ChatGPT's article, citing the richness of detail in the specific celebrations covered. ChatGPT rated both articles similarly in the first run and upon further analysis, provided a summary that seemed indicative of a difference in tone between the models. This underscores the nuanced differences in how each AI model approaches content creation, with each exhibiting its unique tone to some extent.

#### Comparative Analysis Summary by GPT

**Comparative Analysis Summary**: Article 1 outperforms Article 2 overall due to its more vivid descriptive language, greater depth of informational content, and a slightly more original approach to the topic. It captures the essence of the Lantern Festival with greater creativity and detail, enhancing reader engagement and offering a richer narrative experience. Article 2, while strong in research, clarity, and ethical considerations, lacks the vivid storytelling and unique insights that set Article 1 apart.

This last exploration concluded with insights into the existence of “tones” by based on the model itself and the challenges associated with structuring prompts to achieve specific content goals. While the increased structure attempted to refine the output, it also revealed limitations and opportunities for further experimentation in AI-driven content creation.

### Reflection

This exploration into creating specialized roles and refining content through AI platforms has been enlightening, underscoring the complexity of AI interactions at their current level of maturity. The journey revealed several key insights, particularly the importance of ethical considerations in content evaluation, a suggestion brought to light by ChatGPT. The interaction with different AI platforms, each with its unique responses and capabilities, offered a fascinating glimpse into the potential of AI in augmenting human creativity and expertise. However, it also highlighted the challenges in aligning AI output with specific project needs and ethical standards, necessitating a continuous dialogue between technology and human judgment.

## Key Insights

**Positive Observations:**

- **Effective AI and Human Collaboration**: The successful combination of specialized roles with AI, enhancing content quality and ethical standards.
- **Importance of Ethical Content**: The addition of ethical considerations significantly improved content integrity and relevance.
- **Value of Adaptability**: Continuous adaptation based on feedback and AI interactions proved essential for content refinement.

**Encountered Challenges:**

- **AI Alignment with Standards**: Ensuring AI-generated content meets specific project needs and ethical guidelines continues to be challenging.
- **Complexity in AI Content Creation**: Balancing AI's capabilities with the nuances of human expertise and ethical considerations required careful oversight.