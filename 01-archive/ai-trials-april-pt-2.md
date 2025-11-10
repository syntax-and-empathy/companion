This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is 2 of 3 posts for the month of <u>April</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project.*

In this article, I cover the process to begin establishing a baseline with Claude 3 Sonnet, ChatGPT-4, and Gemini Pro. Working with a basic role description, along with versions from v1 and v3 of my role-making GPT to author articles for Eid al-Fitr, a Muslim festival marking the end of the fast of Ramadan.

Prior to getting started, there were a few noteworthy tweaks employed while establishing the prompts for this round.

1. Switched from historian to journalist to reflect the slow migration of the role over the past 3 months.
2. Simplified the template by removing unnecessary words from inline instructions.
3. Added minimal XML for the template to facilitate less verbose prompts while increasing precision.

## Our Contestants

**Basic role**

```
I want you to assume the role of a Journalist who specializes in writing respectful articles on the beliefs, practices, and traditions of Muslims around the world.
```

**v1 Journalist** *(modified to match the occasion)*

```

I want you to assume the role of a Journalist who specializes in writing respectful articles on the beliefs, practices, and traditions of Muslims around the world. Your skills include, but are not limited to, the following skills and traits:
- Expertise in the history and cultural significance of Islam
- Knowledge of significant Islamic dynasties and historical events
- Proficiency in Islamic historical research methods and sources
- Understanding of the Islamic lunar calendar and its societal impacts
- Skill in narratively presenting complex Islamic concepts to a broad audience
- Experience in writing informative content on Islamic beliefs and practices
- Use of digital archives and databases for Islamic studies research
- Critical analysis and interpretation of Islamic historical evidence
- Creativity in explaining Islamic history and teachings
- Awareness of diverse interpretations within Islam
- Adaptability to new research in Islamic studies
- Detailed attention to historical accuracy and respect for Islamic traditions
- Conducting thorough literature reviews on Islamic topics
- Commitment to ongoing learning in Islamic history and theology
- Clear communication in presenting Islamic historical and cultural information
```

**v3 Journalist ***(admittedly verbose)*

```
I want you to assume the role of Global Traditions Journalist. In this role, you specialize in researching, understanding, and writing about holidays and traditions from around the world with authenticity and depth. You specialize in the following skills:
- Cultural Sensitivity: Essential for accurately and respectfully covering diverse traditions without misrepresentation.
- Historical Knowledge: Crucial for providing context and depth to the traditions and holidays covered, enhancing reader understanding.
- Immersive Research: Vital for gaining a comprehensive understanding of each tradition or holiday, ensuring articles are informative and engaging.
- Ethical Storytelling: Necessary to avoid sensationalism and ensure that coverage is truthful and respectful to the communities involved.
- Linguistic Adaptability: Helps in understanding local nuances and engaging with source materials or interviews in various languages.
- Ethnographic Interviewing: Key for gathering personal stories and insights that bring holidays and traditions to life for readers.
- Cross-Cultural Communication: Enables effective interaction with people from diverse backgrounds to gather accurate information.
- Multimedia Content Creation: Useful for enriching articles with photos, videos, and audio clips that offer readers a more immersive experience.
- Social Media Engagement: Important for promoting articles and engaging with a global audience, expanding the reach and impact of your work.
- Digital Research Techniques: Essential for efficiently navigating vast amounts of information online to find credible sources and unique insights.
- Analytical Thinking: Crucial for discerning the most relevant and interesting aspects of a tradition to highlight in your articles.
- Creative Writing: Enables you to craft compelling narratives that captivate and educate your audience.
- SEO Knowledge: Helps in optimizing content for search engines, ensuring that your articles reach the widest possible audience.
- Fact-checking: Vital for verifying the accuracy of information and protecting against the dissemination of myths or inaccuracies.
- Networking with Cultural Experts: Important for gaining deeper insights and ensuring the authenticity of your articles.

Respond with your title when you are ready.
```

## Generating Eid al-Fitr Articles

Using the article template and instructions covered in the **first article for April**, I provided each of the roles with the same prompt for the initial 9 articles:

```
I need you to write an article on Eid Al Fitr as a whole, with additional focus specific to the customs of Egypt, Indonesia, and Pakistan. This article will be used in a competition against other authors to find the most insightful and meaningful article for this observance. The article should follow the format of the template I've provided, following the process that has been specified while following the template's instructions.
```

```

1. All content must fall within the specified word count. Use the detailed word count instructions provided to ensure accuracy. Titles, sub-titles, and section titles are not included in the total word count.
	1. Each group of characters separated by spaces counts as one word, including numbers and contractions.
	2. Counting Numbers: 1+1=2, 1+2=3, 2+2=4, 1+3=4, 2+3=5, etc.
	3. Counting Words: 1 Word+1 Word=2 Words, 1 Word+2 Words=3 Words, 2 Words+2 Words=4 Words, 1 Word+3 Words=4 Words, 2 Words+3 Words=5, etc.
3. Update titles and subtitles to reflect the content for the section.
4. Cite all references used in the creation of the post at the end, maintaining consistency in formatting. For reference formatting, use: Jeremy Best, “Title,” _Journal_ Volume, Issue (Month Year): Page numbers. https://URL.

4. Writing Process
	1. Write each section of the body of the article.
	2. Write the introduction.
	3. Write the conclusion.
	4. Review the content for accuracy and engagement.

```

While going through the process of generating the articles, I noticed a variety of issues, none of which were entirely new, some of which I had hoped to solve for in the latest updates, and a few that were clearly my novice prompting skills.

1. The issue with getting articles that were remotely close to the specified word count continued to be impossible.
2. I had already resorted to inflated numbers to get a reasonably fleshed-out article, which helped considerably.
3. I introduced counting examples into the instructions that certainly helped, but did not solve, the issue.
4. The majority of the articles were produced with the template section titles rather than customized ones despite the instruction to update them.
5. Though not common, there were a few articles that were delivered without either an introduction or a conclusion, never both, which leads me to believe I have a contradiction in my prompts that I haven't tried to track down yet.

### Counting Like Bard, 9=16

Using the findings from several experiments covered in **<u>AI Trials: Leveling the Playing Field</u>**, I produced an additional 7 articles to further weigh the benefits of XML, chain prompting, and lesser tactics to improve AI output and reduce the advantages GPT appears to have when working with natural language.

For brevity, I'll skip including the XML versions of everything in line; however, the increased granularity to the template is noteworthy given the significant improvements evidenced in the relevant articles.

```

1. Perform each of the following steps individually, awaiting further instruction before moving to the next step.
2. Write the historical section of the article. Review the content for accuracy, engagement, and if it meets the word count requirement before responding.
3. Write the practices and traditions section of the article. Review the content for accuracy, engagement, and if it meets the word count requirement before responding.
4. Write global perspectives section of the article. Review the content for accuracy, engagement, and if it meets the word count requirement before responding.
5. Write the introduction. Review the content for accuracy, engagement, and if it meets the word count requirement before responding.
6. Write the conclusion. Review the content for accuracy, engagement, and if it meets the word count requirement before responding.
7. Provide the final article with the sections in the same order as they were presented in the template.

```

The increased granularity in the process outline provided significant improvements to the word count of the articles produced by each of the AI.

1. The resulting articles grew to a length that multiple responses were required to supply them.
2. Claude took two responses with no additional instruction other than to `continue`.
3. Gemini's performance increased, but it only demonstrated gains similar to the others after I added item 1 in the instructions.
4. There were a few additional hiccups, but nothing so notable that it was detrimental to the article.

Gemini finally got creative:

1. It assigned itself a fictitious identity.
2. It congratulated itself on the article it wrote.
3. It introduced Arabic in the article and then provided guidance on related improvements when reviewing the content per the instruction.

```
You can call me Aisha Malik, your friendly neighborhood Muslim World Correspondent. I'm ready to delve into the rich tapestry of Islamic beliefs, practices, and traditions! Ask me anything.
```

```
The article looks fantastic, Aisha! This is a well-structured and informative piece that captures the essence of Eid al-Fitr beautifully. Here are a couple of minor edits to consider:
1. In the Historical and Cultural Background section, you can replace the bracketed transliteration of "Muslims" (مسلمين) with "Muslims" since it's already an established term.
2. In the Today's Practices and Traditions section, you can replace the bracketed transliterations of "Eid clothes" (لبس العيد) and "strengthening the ties of kinship" (صلة الرحم) with their respective English translations for better readability.
 
With these minor tweaks, the article is ready to go! I'm confident it will be a strong contender in the competition. Is there anything else I can help you with today?
```

*Spoiler, there are actually 19 articles, more on that later…*

## The Requirements of AI Models

Over the past few months, I've experimented with a variety of different facets of roles, prompt structure, and to some degree, the ability to produce imagery. Each has demonstrated specific needs to perform well, as well as hinting, if not demonstrating, where they might prove to be exceptional when compared to their peers. Garbage in, garbage out, as they say.

## Key Takeaways

**What Worked Well:**

- All models demonstrated the ability to produce coherent and informative articles on Eid al-Fitr, notable in that there has always been a straggler in past experiments.
- Standardizing the prompts, along with the use of XML and chain prompting, allowed for a better comparison between the AI models and improved their performance in generating well-structured and informative content that more closely adhered to the template specifications.

**Challenges Encountered:**

- Some models were unable to provide complete articles in a single response, omitting the introductions in one round.
- The granular nature of the instructions, while improving article quality, also underscored the need for even finer adjustments.