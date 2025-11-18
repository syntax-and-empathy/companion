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