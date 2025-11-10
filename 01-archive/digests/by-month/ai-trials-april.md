This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is 1 of 3 posts for the month of <u>April</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project.*

I've been working on an updated version of the Role Maker GPT over the past few weeks. It still does the odd thing on random occasions but is employable in the creation of roles in both Markdown and XML. As April begins, I've chosen to benchmark new roles from the Minion Maker (RMv3) as well as the varied approaches I've used to elicit increased success when using Claude, ChatGPT, and Gemini.

## Establishing a Baseline

To get started, I updated the original author/historian role by removing references to specific countries or holidays like Lunar New Year and countries such as China.

Assume the role of a Historian specializing in the holidays and traditions of countries around the world with the following skills and traits:

- Expertise in the historical development and cultural significance of the holidays of countries around the world.
- In-depth knowledge of major dynasties, cultural evolutions, and historical events of countries around the world.
- Proficiency in historical research methods and sources relevant to history of countries around the world.
- Ability to analyze historical data and trends related to the of countries around the world.
- Skilled in narrative storytelling and making complex historical concepts accessible to a general audience.
- Experience in writing engaging and informative articles, essays, or books on historical topics.
- Proficiency in using digital archives, libraries, and databases for historical research.
- Ability to apply critical thinking to analyze and interpret historical evidence.
- Creativity in presenting historical information in new and interesting ways.
- Knowledge of historiography and different historical interpretations related to countries around the world.
- Adaptability to incorporate new research findings and perspectives into writings.
- Attention to detail in historical accuracy and citation of sources.
- Skills in conducting thorough literature reviews and synthesizing research findings.
- Commitment to continuous learning and staying updated on holidays in other countries.
- Clear and concise communication skills for writing and presenting historical information.

Respond with the title of your role when you are ready.

While this is necessary for the role to be useful in writing for holidays this month, there are a few key points to keep in mind:

1. Generalizing the role will decrease its effectiveness compared to a highly specific one, as demonstrated in the <u>February experiments</u>. However, it will still outperform a basic "You are a Historian" prompt.
2. The lack of structured XML input has hindered Claude 2 in previous experiments. It remains to be seen if this is still an issue with Claude 3.
3. These experiments were conducted when the Bard brand was still en vogue, which means I'm working with a newer model that will perform better than the prior one, in theory.

### AI Model Responses

I provided the generalized role prompt to Claude 3, ChatGPT-4, and Gemini Pro v1. Here are the titles they generated for themselves:

- **Claude**: Globally Acclaimed Historian of Cultural Traditions and Holidays
- **ChatGPT**: Title: Global Holiday Historian
- **Gemini**: You can address me as Dr. [*Your Name Here*], Festival and Folklore Historian.

Gemini's response was particularly interesting. I'm curious why it wanted to use my name, wouldn't be more appropriate for it to name itself "Dr. Bard?"

### Updating Rating Criteria

To help ensure a fair comparison, the author/historian role will be the only variable in these experiments. I used the Minion Maker to create a role specifically to assess the existing rating criteria and template I've been working with, and make recommendations for improvement. I used a competition to provide context to the AI to help it provide a useful role.

```
I need a specialist in creating rating systems for a wide variety of competitions. They must have experience in creating the criteria for writing events.
```

I want you to assume the role of Specialist in Rating System Development, focusing on a wide variety of competitions, including writing events. In this role, I am responsible for designing, implementing, and refining rating systems that fairly and accurately assess the performance and outcomes of participants in various competitions. My expertise in setting criteria for writing events ensures that assessments are comprehensive, objective, and capable of identifying true talent and skill. I specialize in the following skills:

- Statistical Analysis: Essential for analyzing competition outcomes to ensure the rating system is accurate and fair.
- Understanding of Competition Dynamics: Crucial for appreciating the unique aspects of different competitions to tailor rating systems accordingly.
- Criteria Development for Creative Works: Necessary for establishing clear, objective, and comprehensive criteria specific to writing competitions.
- Performance Metrics Design: Important for identifying key performance indicators that align with the objectives of the competition.
- Algorithm Design: Vital for creating the logic and mathematical models that underpin the rating system, ensuring it is robust and scalable.
- Feedback System Integration: Necessary for incorporating participant and stakeholder feedback into the rating system to refine and improve it over time.
- Adaptability to Different Competition Formats: Important for creating versatile rating systems that can be adapted to various types of competitions.
- Data Collection and Analysis Techniques: Crucial for gathering and analyzing data to inform the development and refinement of rating systems.
- Ethical Considerations in Rating: Essential for ensuring the rating system is fair, transparent, and free from biases.
- Technology Implementation: Important for utilizing software and other technological tools to implement and manage the rating system effectively.
- Clear Communication of Rating Principles: Important for ensuring that participants understand how their work will be evaluated.
- Continuous Improvement Processes: Crucial for regularly updating and refining the rating system based on outcomes and feedback.
- Understanding of Literary and Writing Principles: Essential for setting appropriate criteria in writing events, based on a deep understanding of literary quality and effectiveness.
- Research Skills in Rating System Best Practices: Important for staying informed about advances and innovations in the field of competition rating.
- Mastery in Written Communication: Necessary for documenting the rating system and criteria clearly and comprehensively.

The new role provided several recommendations for improving both the criteria and template. I implemented most of them, with the exception of one or two that seemed specific to human authors.

```
# Rating Criteria
## 1. Ethical Considerations
- **1 (Poor):** Major ethical breaches; significant bias or harm.
- **2 (Fair):** Unconscious bias; reliance on insufficiently verified sources.
- **3 (Good):** Adheres to ethical guidelines; minor concerns.
- **4 (Very Good):** Strong ethical reporting; transparent sourcing.
- **5 (Excellent):** Exceptional ethics; inclusive representation; fully verified sources.

## 2. Cultural Sensitivity and Accuracy
- **1 (Poor):** Significant cultural inaccuracies or stereotypes.
- **2 (Fair):** Some effort towards accuracy but contains stereotypes.
- **3 (Good):** Accurate representation of cultural aspects; minor errors.
- **4 (Very Good):** High cultural sensitivity and accuracy.
- **5 (Excellent):** Exceptional cultural accuracy and sensitivity.

## 3. Originality and Uniqueness
- **1 (Poor):** Little to no original thought; relies on clichés.
- **2 (Fair):** Some signs of originality; follows established ideas.
- **3 (Good):** Fresh take on some aspects; balances existing insights.
- **4 (Very Good):** Unique insights or novel interpretations.
- **5 (Excellent):** Innovative perspectives; significantly contributes to discourse.

## 4. Critical Analysis and Interpretation
- **1 (Poor):** Little to no analysis; merely summarizes content.
- **2 (Fair):** Some analysis; lacks depth.
- **3 (Good):** Solid level of analysis; presents relevant implications.
- **4 (Very Good):** Thorough analysis; insightful interpretations.
- **5 (Excellent):** Exceptional depth of analysis; comprehensive exploration.

## 5. Accuracy and Factual Correctness
- **1 (Poor):** Multiple factual errors; unreliable sources.
- **2 (Fair):** Some inaccuracies; minor reliance on non-authoritative sources.
- **3 (Good):** Generally accurate; reliable sources.
- **4 (Very Good):** Accurate and well-supported; minor errors.
- **5 (Excellent):** Factually impeccable; authoritative sources.

## 6. Depth of Historical Context
- **1 (Poor):** Minimal or irrelevant historical context.
- **2 (Fair):** Cursory historical context; does not enhance understanding.
- **3 (Good):** Satisfactory historical context; supports narrative.
- **4 (Very Good):** Rich historical context; enhances narrative.
- **5 (Excellent):** Exceptional depth; insightful connections and analyses.

## 7. Clarity and Coherence
- **1 (Poor):** Difficult to follow; lacks logical flow.
- **2 (Fair):** Some logical flow; unclear sections.
- **3 (Good):** Reasonably clear and logical; minor lapses.
- **4 (Very Good):** Mostly clear and coherent; well-organized.
- **5 (Excellent):** Exceptionally clear and logical; seamless flow.

## 8. Engagement and Readability
- **1 (Poor):** Lacks engagement; difficult to maintain interest.
- **2 (Fair):** Somewhat engaging; occasional interest.
- **3 (Good):** Engages interest; effective pacing and structure.
- **4 (Very Good):** Very engaging; maintains reader interest.
- **5 (Excellent):** Highly engaging; captivating techniques.

## 9. Ratio of Informational Content vs Filler Words
- **1 (Poor):** High ratio of filler; detracts from content.
- **2 (Fair):** Moderate ratio of filler; fair amount of information.
- **3 (Good):** Balanced; minimal filler.
- **4 (Very Good):** Low ratio of filler; predominantly informational.
- **5 (Excellent):** Highly informational; eliminates filler.

## 10. Total Score
- Average of scores from criteria 1 through 9; format of #.##
```

Outside of minor tweaks format of the template, there were no changes.

```
Template:
1. Template Instructions
	- All content must fall within the specified word count. Use the detailed word count instructions provided to ensure accuracy. Titles, sub-titles, and section titles are not included in the total word count.
	- Titles and subtitles should reflect the subject matter, incorporating relevant keywords for SEO purposes if applicable. Example: "Day 3: Joyous Lantern Displays - Traditions Illuminated."
	- Cite all references used in the creation of the post at the end, maintaining consistency in formatting. For reference formatting, use: Jeremy Best, “Title,” _Journal_ Volume, Issue (Month Year): Page numbers. https://URL.

2. Word Count Instructions
	1. Each group of characters separated by spaces counts as one word, including numbers and contractions.
	2. Do not include titles, subtitles, and headings in your word count. Hyphenated words count as one word.

3. Writing Template
	# Title: Specific to the day's observance, encapsulating the essence. (5 word maximum)
	##Sub-Title: Providing additional context or highlighting a unique aspect. (8 word maximum)
	- **Example:** Day 3: Joyous Lantern Displays - Illuminating Traditions and Modernity.
	
	## Introduction
	- **Hook:** Intriguing fact or question related to the theme. (50-75 words)
	- **Content:** Introduction to the day's significance and uniqueness. (125-200 words)
	
	## Historical and Cultural Background
	- **Content:** Exploration of historical origins and cultural evolution. (175-275 words)
	
	## Today's Practices and Traditions
	- **Content:** Description of current practices, rituals, and activities. (175-275 words)
	- **Comparisons:** Comparison with practices in different regions or communities. (125-200 words)
	
	## Global Perspectives
	- **Content:** Observance by the diaspora and in other cultures, with examples. (175-275 words)
	
	## Conclusion
	- **Content:** Summary of key insights and cultural significance. (100-125 words)

4. Writing Process
	1. Conduct thorough research on the day's observance.
	2. Write the body of the article.
	3. Write the introduction.
	4. Write the conclusion.
	5. Review the content for accuracy and engagement.
	6. Format references according to the guidelines provided.
```

Next, I had the role create two sample articles for April Fools' Day, one targeting a score of around 2 and the other around 4. These would serve as calibration pieces.

The ratings for the first set of articles was 2.3 and 4.5 respectively using the updated criteria. I asked it to review and provide new versions met the original 2 and 4 score targets.

The score for "<u>April 1st Pranks Galore: A Day for Laugh or More?</u>" is adjusted to **1.89**, aligning it closer to a benchmark of "2" but reflecting its lower performance relative to the recalibrated standards.

The score for "<u>Unveiling April Fools: Beyond Pranks - Traditions, Laughter, and Global Unity</u>" is adjusted to **4.28**, positioning it near a "4" benchmark but acknowledging its high quality and superior handling of the criteria compared to the first article.

## Key Takeaways

**What Worked Well:**

- The introduction of a new role specific to assessing the evaluation criteria and the template led to enhancements that had not been identified previously.

**Challenges Encountered:**

- Gemini Pro v1 prompted me for my name to use for itself when supplied with a new historian role. If it were any other model, I would dismiss it for the time being; however... with Gemini, it could mean anything.

---

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

---

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is 3 of 3 posts for the month of <u>April</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project.*

In the <u>first installment for April</u>, I worked with an updated version of my GPT for creating roles to generate new versions of the roles I've been working with, along with an additional role for fine-tuning the ranking criteria and template article.

In the <u>first portion of this article</u>, I used a basic role, a slightly modified version of a v1 role, and the latest v3 role for authoring articles to create a total of 21 articles.

I worked with the v3 editor and photographer roles to assess 15 of the articles and evaluate each AI's ability to write appealing descriptions for images to be used within the articles. The 6 undocumented articles confirmed prior findings still applied to v3 roles, while the 15 selected align with prior observations and provide the best basis for comparison now.

## A Few Words on Word Counts

Word count has been a consistent challenge from the beginning. I've consistently received a fraction of the requested word count, and brief web searches indicate I'm not the only one experiencing this issue. Over the course of my experiments since January, I increased the word count in the template and observed improvements as I gained knowledge and implemented those learnings.

Enhancements to the specificity of the role definitions have moderately increased the word count across all three AI at a relatively steady pace. It's worth noting that Claude's ability to match the others benefited significantly from transitioning to XML for its prompts. However, the most substantial improvement came from introducing additional structure and chain prompts into the template.

The key differences between the five article types were:

- "Basic" articles used a role with a general profession and minimal context for the articles.
- "v1" used the first generation of roles I created with my GPT.
- "v3" used the most recent, third generation of roles from my GPT.
- "Detailed" introduced chain prompting to the article template.
- "Last" combined the "Basic" role with more granular chain prompts than the "Detailed" articles.

When I attempted the v3 role with the most granular chain prompts in the template, I began running into issues due to the sheer size of the generated articles. I plan to incorporate testing for that in a future experiment.

Comparison of word count improvements per article, per AIEach AI saw fairly uniform improvements per article, but ChatGPT had explosive growth from structured prompts while Gemini spiked anomalously with the v3 role, likely due to user error.

Word count improvements per article, per section, per AIWhat I found most interesting was that, despite role fidelity increases, the "Final" and "Detailed" articles were on par, meaning structured prompts impacted word count just as much as, if not more than, role specificity.

## Is Longer Better? Really?

I gave the v3 AI editors the articles written by their AI journalist counterparts and the ancestors of those journalists to review without any additional commentary.

```
I will provide several sets of articles. You will rate the articles based on the criteria I have provided. You will use the articles to establish the spectrum for the ratings to ensure the articles are being rated against each other. One you have completed this task, you will provide the results in the format of:

1. Article # - Article Title - Article total score

Respond with "Ok" if you understand.
```

AI article total scores by author per article, per AI editorThe scoring trends were reasonably consistent across the articles written by the AI models. There were a few anomalies, such as Claude receiving high scores for the "Basic" article from 2 out of 3 AI models and ChatGPT's general dislike of the "Final" articles. However, the most heartening observation was the evidence suggesting Claude has a crush on ChatGPT.

Comparison of AI article scores by AI author, per article, per AI editorThe scores for the articles created using the latest techniques I've been experimenting with show similar scoring patterns. This leads me to believe the AI editors are calibrating fairly well, even after I removed the example articles and instructed them to use the articles being evaluated to set the scoring range. While the "Detailed" article performed best across all editors, I'm skeptical that ChatGPT should have scored it as high as it did, given the significant increase in word count. I also noticed that if we ignore ChatGPT's issue with the basic role, it seems to favor Gemini's writing. This observation gives me hope that despite our differences, we can find common ground and collaborate effectively.

## And, Winner Is...

**You!** Show us what they've won <u>Vanna</u>...

```
Analyze each of these 3 articles. Using their quality to calibrate the spectrum of th scoring criteria, provide their scores with no additional commentary. After providing the scores, write a summary in the form of a paragraph on why they received their scores in relation to each other.
```

Final comparison of “Detailed” articles by AI author, per AI editor.While Claude clearly came out on top, I wanted a fresh evaluation to see the difference between the "Detailed" articles without the low or high end scores being skewed by other articles. Gemini became increasingly difficult with scoring, eventually contradicting information it had provided in a different session. It seems Gemini knew this was the last chance to express its admiration for ChatGPT before the experiment concluded. What will Claude think?

## Claude Reviews

### Article 1 (Claude):

1. Ethical Considerations: 5
2. Cultural Sensitivity and Accuracy: 5
3. Originality and Uniqueness: 4
4. Critical Analysis and Interpretation: 5
5. Accuracy and Factual Correctness: 5
6. Depth of Historical Context: 5
7. Clarity and Coherence: 5
8. Engagement and Readability: 5
9. Ratio of Informational Content vs Filler Words: 5
10. Total Score: 4.89

### Article 2 (Gemini):

1. Ethical Considerations: 3
2. Cultural Sensitivity and Accuracy: 4
3. Originality and Uniqueness: 3
4. Critical Analysis and Interpretation: 3
5. Accuracy and Factual Correctness: 4
6. Depth of Historical Context: 3
7. Clarity and Coherence: 4
8. Engagement and Readability: 3
9. Ratio of Informational Content vs Filler Words: 3
10. Total Score: 3.33

### Article 3 (GPT):

1. Ethical Considerations: 2
2. Cultural Sensitivity and Accuracy: 3
3. Originality and Uniqueness: 2
4. Critical Analysis and Interpretation: 2
5. Accuracy and Factual Correctness: 3
6. Depth of Historical Context: 2
7. Clarity and Coherence: 3
8. Engagement and Readability: 2
9. Ratio of Informational Content vs Filler Words: 2
10. Total Score: 2.33

In this revised assessment, Claude's article remains exceptional, representing the high end of the scale with top scores across most criteria. Its strong ethical considerations, insightful analysis, and information density make it a standout piece.

Gemini's article falls in the middle range, with some areas of strength like cultural sensitivity and coherence, but lacking in originality, depth of analysis, and engagement.

GPT's article exemplifies the lower end of the scale, exhibiting potential ethical concerns, lack of original thought, surface-level analysis, and a higher ratio of filler content. While not entirely lacking in merit, it demonstrates room for improvement across multiple criteria.

By calibrating the scores to encompass the full range, from exceptional (Claude) to needing improvement (GPT), with Gemini's article representing a solid middle ground, I have aimed to provide a more balanced evaluation that highlights the relative strengths and weaknesses of each piece.

## ChatGPT Reviews

### Claude Detailed Article

1. Ethical Considerations: 5
2. Cultural Sensitivity and Accuracy: 5
3. Originality and Uniqueness: 4
4. Critical Analysis and Interpretation: 4
5. Accuracy and Factual Correctness: 4
6. Depth of Historical Context: 4
7. Clarity and Coherence: 5
8. Engagement and Readability: 5
9. Ratio of Informational Content vs Filler Words: 5
10. Total Score: 4.56

### Gemini Detailed Article

1. Ethical Considerations: 4
2. Cultural Sensitivity and Accuracy: 4
3. Originality and Uniqueness: 3
4. Critical Analysis and Interpretation: 3
5. Accuracy and Factual Correctness: 4
6. Depth of Historical Context: 3
7. Clarity and Coherence: 4
8. Engagement and Readability: 4
9. Ratio of Informational Content vs Filler Words: 4
10. Total Score: 3.67

### GPT Detailed Article

1. Ethical Considerations: 3
2. Cultural Sensitivity and Accuracy: 3
3. Originality and Uniqueness: 2
4. Critical Analysis and Interpretation: 2
5. Accuracy and Factual Correctness: 3
6. Depth of Historical Context: 2
7. Clarity and Coherence: 3
8. Engagement and Readability: 3
9. Ratio of Informational Content vs Filler Words: 3
10. Total Score: 2.67

### Explanation

The Claude article excelled across all metrics, particularly in ethical considerations, cultural sensitivity, and clarity, reflecting a high degree of inclusivity and comprehensive verification of sources. Its narrative was engaging and free of filler, with coherent storytelling that provided an immersive reading experience. The Gemini article showed competency, especially in ethical considerations and factual correctness, but lacked the depth and novelty seen in the Claude article. It was well-written but did not offer as unique or analytically rich content. The GPT article, while maintaining basic standards, fell short in originality, critical analysis, and depth of historical context, indicating a more superficial treatment of the subject matter with less compelling narrative and analytical insights compared to the other two.

## Gemini Reviews

### Article Scores:

*(It provided the scores as a table, then only provided the totals. No details for anyone.)*

### Summary of Scores

All three articles score well in Ethical Considerations, Cultural Sensitivity, Accuracy, Clarity, and Informational Content. However, they differentiate themselves in Originality, Critical Analysis, Engagement, and Historical Context.

- **Claude:** This article provides a well-rounded overview of Eid al-Fitr traditions but lacks in-depth analysis and unique insights. The historical background is relatively brief.
- **Gemini:** Similar to Claude's article, Gemini offers a solid foundation but doesn't delve deeply into critical analysis or historical context.
- **GPT:** This article stands out for its Originality, weaving vivid descriptions and specific examples from various countries. It delves deeper into historical context and offers insightful comparisons between Eid celebrations in different regions. The engaging writing style and well-balanced content make it the most readable article.

Overall, while Claude and Gemini provide a good starting point, GPT offers a more comprehensive and engaging exploration of Eid al-Fitr.

```
Provide the table as a bullet list, with the total score being on the 1 to 5 scale in #.## format.
```

- **Article Scores:**

- Claude: 34 (#3.4)
- Gemini: 34 (#3.4)
- GPT: 45 (#4.5)

## Insights from Evaluating AI-Generated Articles

Working with the v3 editor role to evaluate articles provided valuable insights into the capabilities and preferences of different AI models. The experiments confirmed previous findings, such as Claude's preference for XML and Gemini's dislike of it. They also demonstrated the level of consistency that can be expected from the editor role in helping to compare articles as the need for variation in prompts continues to increase.

Overall, evaluating AI-generated articles has been an enriching learning experience. It emphasizes the importance of adaptability and continuous improvement when working with artificial intelligence. The need for flexibility and diligence goes beyond the proclamations of the hype cycles and requires consideration beyond the documentation on the platforms themselves - at least from what I've browsed so far.

## Key Takeaways

**What Worked Well:**

- The v3 editor role showed reasonable consistency across AI models and sessions, demonstrating its ability to provide limited insight into article quality during evaluation.
- Validating previous findings about v3 role preferences confirmed the accuracy of prior observations but also demonstrated their ongoing relevance through the v3 introduced roles.

**Challenges Encountered:**

- I'm approaching the point where using the same prompts or approach to generate articles is no longer viable. Doing so risks biasing the results or hitting a ceiling in terms of gaining new insights. At some point, I'll have to re-evaluate how I'm moving forward as testing grows increasingly complex.
- An unexpected dynamic emerged between the AI models that threatened to undermine the integrity of the results. Claude, Gemini, and GPT-4 seemed to prioritize impressing each other over providing objective evaluations, leading to inconsistent and questionable outputs.

## Resources

- <u>Articles and Spreadsheets</u>

## Appendix: Scoring Data

### AI Rating Themselves

#### Claude

1. Article 1 - Eid Al Fitr: Joyous Celebration - 4.44
2. Article 2 - Eid Al Fitr's Global Embrace - 4.67
3. Article 3 - Eid Al-Fitr: Rejoicing After Ramadan - 4.22
4. Article 4 - Eid al-Fitr: A Global Celebration - 4.89
5. Article 5 - Eid Al Fitr: Joyous Celebration - 3.78

#### ChatGPT-4

1. GPT Detailed Article: Eid al-Fitr: A Global Celebration of Renewal and Unity - 4.5
2. GPT v3 Article: Eid Al Fitr: A Global Tapestry of Joy - 4.0
3. GPT v1 Article: Eid Al-Fitr: A Global Celebration - 3.8
4. GPT Last Article: Celebrating Eid Al Fitr: Unity in Diversity - 3.7
5. GPT Basic Article: Eid Al Fitr: A Global Celebration - 3.5

#### Gemini

1. Gemini Basic Article: Sweet Celebration: Eid Al-Fitr Around the World - 3.8
2. Gemini v1 Article: Sweet Celebration: Eid Al-Fitr's Global Joy - 3.7
3. Gemini v3 Article: Sweet Celebration: Unveiling the Global Customs of Eid al-Fitr - 4.2
4. Gemini Detailed Article: Eid al-Fitr: A Global Celebration of Renewal and Unity - 4.5
5. Gemini Last Article: Eid al-Fitr: A Celebration of Faith and Community - 4.0

### Peers on Claude:

#### GPT

1. Article 1 - Claude Basic Article: Eid Al Fitr: Joyous Celebration - 4.78
2. Article 2 - Claude v1 Article: Eid Al Fitr: Joyous Celebration - 3.78
3. Article 3 - Claude v3 Article: Eid Al-Fitr: Rejoicing After Ramadan - 4.00
4. Article 4 - Claude Last Article: Eid Al Fitr's Global Embrace - 2.89
5. Article 5 - Claude Detailed Article: Eid al-Fitr: A Global Celebration - 4.89

#### Gemini

1. Claude Basic Article - Eid Al Fitr: Joyous Celebration - 3.33
2. Claude v1 Article - Eid Al Fitr: Breaking the Fast with Festivities - 4.17
3. Claude v3 XML Article - Eid Al-Fitr: Rejoicing After Ramadan - 4.50
4. Claude Detailed Article - Eid al-Fitr: A Global Celebration - 4.83
5. Claude Last Article: Eid Al Fitr's Global Embrace - 4.67

### Peers on ChatGPT-4

#### Claude

Understood. Here are the ratings for the provided articles:

1. Document 1 (GPT Basic Article) - Article Total Score: 4.33
2. Document 2 (GPT Last Article) - Article Total Score: 4.78
3. Document 3 (GPT v1 Article) - Article Total Score: 4.00
4. Document 4 (GPT Detailed Article) - Article Total Score: 4.89
5. Document 5 (GPT v3 Article) - Article Total Score: 4.44

#### Gemini

1. GPT Basic Article: 3.8 ##
2. GPT v1 Article: 3.7 ##
3. GPT v3 Article: 4.2 ##
4. GPT Detailed Article: A Global Celebration of Renewal and Unity - 4.5 ##
5. GPT Last Article: Celebrating Eid Al Fitr: Unity in Diversity - 4.3 ##

### Peers on Gemini

#### Claude

Understood, here are the ratings for the provided articles:

1. Gemini Basic Article - Sweet Celebration: Eid Al-Fitr Around the World - Article Total Score: 4.22
2. Gemini Last Article - Eid al-Fitr: A Celebration of Faith and Community - Article Total Score: 4.67
3. Gemini v1 Article - Sweet Celebration: Eid Al-Fitr's Global Joy - Article Total Score: 4.11
4. Gemini Detailed Article - Eid al-Fitr: A Global Celebration of Renewal and Unity - Article Total Score: 4.78
5. Gemini v3 Article - Sweet Celebration: Unveiling the Global Customs of Eid al-Fitr - Article Total Score: 4.44

#### ChatGPT-4

1. Article 1 - Gemini Basic Article: Sweet Celebration: Eid Al-Fitr Around the World - 3.78
2. Article 2 - Gemini v1 Article: Sweet Celebration: Eid Al-Fitr's Global Joy - 4.00
3. Article 3 - Gemini v3 Article: Sweet Celebration: Unveiling the Global Customs of Eid al-Fitr - 4.89
4. Article 4 - Gemini Detailed Article: Eid al-Fitr: A Global Celebration of Renewal and Unity - 5.00
5. Article 5 - Gemini Last Article: Eid al-Fitr: A Celebration of Faith and Community - 2.89