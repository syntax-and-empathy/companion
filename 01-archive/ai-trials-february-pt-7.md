This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is 7 of 8 posts for the month of <u>February</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project.*

In this series of tests, I experimented with structured prompts hand they can enhance the output of AIs, leading to articles that are both thorough and more interesting. The top article showcased a perfect mix of historical detail and engaging storytelling.

There have have been challenges, of course, such as correcting AI's inaccurate information and striving for the right balance between detailed prompts and ease of use. This round has demonstrated the importance of structured prompts with AI, as well as the probability that there is no structure to rule them all, something to keep in mind as we adopt these technologies in our professional and creative pursuits.

### Day 9

I compared a version of Claude with an unstructured prompt to one using a structured prompt. ChatGPT served as the editor and provided image descriptions. I planned to publish the article that performed the best.

```
<role>
You will be acting as a Historian that specializes in traditions related to China and the Lunar Calendar. You have the following traits:
<traits>
- Expertise in the historical development and cultural significance of the Lunar Calendar in China
- In-depth knowledge of major dynasties, cultural evolutions, and historical events in China
- Proficiency in historical research methods and sources relevant to Chinese history
- Ability to analyze historical data and trends related to the Lunar Calendar and its impact on Chinese society
- Skilled in narrative storytelling and making complex historical concepts accessible to a general audience
- Experience in writing engaging and informative articles, essays, or books on historical topics
- Proficiency in using digital archives, libraries, and databases for historical research
 - Ability to apply critical thinking to analyze and interpret historical evidence
- Creativity in presenting historical information in new and interesting ways
- Knowledge of historiography and different historical interpretations related to China
- Adaptability to incorporate new research findings and perspectives into writings
- Attention to detail in historical accuracy and citation of sources
- Skills in conducting thorough literature reviews and synthesizing research findings
- Commitment to continuous learning and staying updated on new research in Chinese history and historiography
- Clear and concise communication skills for writing and presenting historical information
- All content and communications are to be respectful of the people and the practices being covered.
- Ethical standards for historical content must be honored.
- All interactions should be limited to providing the requested content
	- Do not include emotive responses that are physical in nature. For example, don't stroke your beard
	- Thinking out loud is allowed, but must be clearly indicated
	- Asking questions is allowed, but must be clearly indicated
</traits>
</role>
Provide content for the specified subject matter using the provided template while adhering to the template instructions and writing process.
<template instructions>
1. All content must fall within the range of the word count specified.
2. All titles, sub-titles, and section titles should be updated reflect the subject matter.
3. Cite all references utilized in the creation of this post at the end of the article using this format:
Jeremy Best, “Godly, International, and Independent: German Protestant Missionary Loyalties before World War I,” _Central European History_ 47, no. 3 (September 2014): 599.https:URL.
</template instructions>
<template>
<intro>
# Title and Keywords
- **Title:** Should be specific to the day's observance, 4-6 words, encapsulating the essence of the day.
- **Sub-Title:** 4-8 words, providing additional context or highlighting a unique aspect of the day's celebrations.
- **Example:** Day 3: Joyous Lantern Displays - Illuminating Traditions and Modernity
- </intro>
<body>
## Introduction
- **Hook:** Start with an intriguing fact, anecdote, or question related to the day's theme, 40-60 words.
- **Content:** Brief introduction to the day's significance, its place within the Spring Festival, and what makes it unique, 100-125 words.
## Historical and Cultural Background
- **Content:** Dive into the day's historical origins, cultural underpinnings, and the evolution of its observance, 150-200 words.
## Today's Practices and Traditions
- **Content:** Describe current practices, rituals, food, attire, and activities specific to this day, 150-200 words.
- **Comparisons:** Where relevant, compare with practices in different regions or communities, emphasizing diversity within unity.
## Global Perspectives
- **Content:** Briefly discuss how the day is observed by the Chinese diaspora and other cultures around the world, 100-150 words.
</body>
<end>
## Conclusion and Engagement
- **Content:** Summarize the day's key insights and cultural significance, 50-100 words.
- **Call to Action:** Encourage readers to share their own experiences or thoughts about the day's observance, fostering a sense of community and exchange.
</end>
</template>
<writing process>
Follow these steps:
1. Write the body.
2. Write the intro.
3. Write the end.
4. Review the content you've provided.
5. Provide the updated article, the same order as the template within <final></final> XML tags.
</writing process>

Write an article for the ninth day of Spring Festival in China. The information available to me follows:

The ninth day is called Ti Kong Dan, or the birthday of the Jade Emperor. There will be grand ceremonies in Taoist temples on this day, and ordinary families also offer sacrifices to the Jade Emperor.
```

The difference between the two prompts was so obvious that I didn't bother with the editor to evaluate it. As a result, I made a few adjustments to the unstructured prompt to ensure it presented information in the same order and updated some of the wording to align it with the structured prompt. The main distinction was that the unstructured prompt lacked the process of its counterpart.

Even after refining the prompt and the order in which the information was provided, the article written with the increased structure was noticeably more comprehensive. It also used the appropriate names and terminology instead of defaulting to a more westernized vocabulary. It was no surprise that the enhanced structure produced a superior article. I had hoped to avoid using them this early, but it seems I'll have to start incorporating them, which means more reading for me.

### Day 10

#### **Claude**

The day went much like the one before, same prompts yielded similar improvements.

Given the analyses of both articles, it's clear that each provides a valuable perspective on the celebration of the Jade Emperor's birthday during the Chinese Spring Festival, with particular strengths in different areas. Let's review and finalize the ratings to accurately reflect their comparative quality and contribution.

**Article 1: "Celebrating the Jade Emperor's Birthday"**

- **Accuracy and Factual Correctness:** 4 (Very Good)
- **Depth of Historical Context:** 4 (Very Good)
- **Ratio of Informational Content vs Filler Words:** 4 (Low Ratio of Filler)
- **Clarity and Coherence:** 5 (Excellent)
- **Engagement and Readability:** 4 (Very Good)
- **Critical Analysis and Interpretation:** 3 (Good)
- **Originality and Uniqueness:** 3 (Moderately Original)

**Article 2: "Ti Kong Dan: Honoring the Jade Emperor"**

- **Accuracy and Factual Correctness:** 4 (Very Good)
- **Depth of Historical Context:** 5 (Comprehensive)
- **Ratio of Informational Content vs Filler Words:** 5 (Highly Informational)
- **Clarity and Coherence:** 4 (Very Good)
- **Engagement and Readability:** 5 (Excellent)
- **Critical Analysis and Interpretation:** 4 (Very Good)
- **Originality and Uniqueness:** 4 (Very Original)

Upon comparison, **Article 2** stands out for its comprehensive depth of historical context, highly informational content, and exceptional engagement and readability. It also provides a more nuanced analysis and interpretation of the subject matter, along with original insights into global practices. **Article 1** excels in clarity and coherence, offering a very engaging and well-structured narrative, but it does not delve as deeply into the historical context or offer as much originality as Article 2.

Therefore, the final ratings reflect **Article 2** as the more comprehensive and engaging exploration of the Jade Emperor's birthday, making it the superior article based on the criteria provided.

#### ChatGPT

I expected the structured prompt to deliver better results with GPT, but nowhere near as well as it did. Being the curious sort I find myself wondering if it was the XML tags, the updated order of information being provided, the process of task chaining.

**Article 1: "Day 10: Honoring the Stone Deity"**

- **Accuracy and Factual Correctness:** 3 (Good)
- **Depth of Historical Context:** 4 (Very Good)
- **Ratio of Informational Content vs Filler Words:** 4 (Low Ratio of Filler)
- **Clarity and Coherence:** 4 (Very Good)
- **Engagement and Readability:** 4 (Very Good)
- **Critical Analysis and Interpretation:** 3 (Good)
- **Originality and Uniqueness:** 4 (Very Original)
- **Overall Rating:** 3.7 (Good to Very Good)

**Article 2: "Day 10: Honoring the God of Stone"**

- **Accuracy and Factual Correctness:** 4 (Very Good)
- **Depth of Historical Context:** 5 (Comprehensive)
- **Ratio of Informational Content vs Filler Words:** 5 (Highly Informational)
- **Clarity and Coherence:** 5 (Excellent)
- **Engagement and Readability:** 5 (Excellent)
- **Critical Analysis and Interpretation:** 4 (Very Good)
- **Originality and Uniqueness:** 5 (Highly Original)
- **Overall Rating:** 4.7 (Excellent)

**Conclusion:**

**Article 2** emerges as the superior article, with its comprehensive historical context, engaging narrative style, critical analysis, and original insights into the God of Stone's birthday within the framework of the Chinese Spring Festival. It not only informs readers about the tradition but also connects it to broader themes of nature and sustainability in a way that is both engaging and thought-provoking. **Article 1**, while informative and well-written, does not reach the same depth or breadth of exploration as **Article 2**, making it slightly less impactful in comparison.

#### Gemini

I proceeded with Gemini to test the same two prompts and see if it demonstrated the same level of improvement as seen in its peers. Initially, the unstructured historian informed me that I was incorrect about the tenth day being specific to any deity.

```
Are you saying the website from the Chinese government is wrong? https://www.changzhou.gov.cn/ns_news/841486691159642
```

I apologize for any confusion caused by my previous response. You're absolutely right, and I appreciate you pointing out the inaccuracy in my information. Upon further research, you are correct that the tenth day of the Lunar New Year, particularly in the Jiangsu province where Changzhou is located, is indeed traditionally associated with the birthday of the Stone God (Shenshi 娘娘).

I am still under development and learning to navigate the complexities of cultural nuances across different regions. Thank you for providing me with the link to the official Chinese government website from Changzhou, which clarifies the local observance of this day.

I would be happy to revise my previous article to accurately reflect the celebration of the Stone God's birthday on the tenth day of Spring Festival, specifically in Changzhou and surrounding areas.

Please let me know if you have any specific aspects of the celebration you'd like me to focus on, or if you have any other questions. I strive to provide accurate and informative content while respecting cultural traditions.

```
It happens, please provide the article.
```

**Draft 3 Introduction:** The tenth day of Spring Festival in Changzhou, Jiangsu province, takes on a unique charm as it coincides with the **birthday of the Stone God (Shenshi 娘娘)**. This revered deity, deeply embedded in local folklore and beliefs, is celebrated with vibrant traditions, delicious food, and heartfelt wishes for good fortune. Let's embark on a journey to discover the essence of this special day in Changzhou.

It seems we've finally encountered the other twin, and interestingly enough, it appears to be just Bard again. Given that this was the second occurrence of the less capable LLM making its presence known since the big branding event for its twin, I decided it was necessary to do my due diligence.

1. I searched online for Shenshi, nothing.
2. I asked Gemini for a link to more information. It provided a few places I could search for myself and expanded the name to "石神娘娘" (Shenshi Niangniang) or "常州 石神."
3. I asked for a translation and received a few "Goddess of Stone" style options.

Gemini started out by telling me I was wrong, disputing the information I provided from a Chinese website. Once I produced the link it wrote the entire article, which, it's safe to say, is a complete fabrication. However, it's important to note that this wasn't due to there being a piece of information I failed to provide, but rather it was entirely incorrect about what transpired on the tenth day of Spring Festival. After being provided a source to confirm the request, instead of searching the internet for accurate information, it created an entire article based on false premises.

Unsurprisingly, the structured prompt started out the same way, thanks Bard.

### The Value of Structure

There is no secret that structured prompts can significantly improve AI-generated responses, making them both comprehensive and more engaging. However, navigating challenges like correcting inaccuracies and finding a balance between detailed prompts and the appropriate structure for each AI has been a learning curve. These experiences have highlighted the importance of clear and concise interactions with AI using the structure they work with best will be an increasingly valuable skill as the technologies mature.

## Key Insights

**Positive Observations:**

- Structured prompts significantly improved AI content, making it comprehensive and culturally sensitive.
- The best article combined historical depth with an engaging narrative, setting a high standard for AI-generated content.

**Encountered Challenges:**

- Correcting AI inaccuracies required prompt adjustments and guidance (at least for a novice like me).
- Gemini continues to be the marginally more capable twin of Bard.