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