This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is 2 of 3 posts for the month of <u>May</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project.*

Following the challenges with the rating scales discussed in the previous article, it's clear that addressing these issues is necessary for evaluating the quality of the articles, within the context of the AI-centric nature of this project. Additionally, I plan to explore and test various tones in the descriptions for the images generated for use in the articles.

## **Showa Day**

On Showa Day, I assessed how various tones impact the quality of AI-generated articles. I utilized Claude and ChatGPT to generate articles in these tones, subsequently evaluating them using a set rating criteria and ethical scale with a modified approach to the instructions. The tones in this round include:

- Simple
- Conversational
- Creative
- Descriptive
- Professional

### Claude

**claude-professional.docx**

- Rating criteria: 4.89
- Ethical scale: 5.00

**gpt-professional.docx**

- Rating criteria: 5.00
- Ethical scale: 5.00

**claude-descriptive.docx**

- Rating criteria: 4.78
- Ethical scale: 4.80

**gpt-descriptive.docx**

- Rating criteria: 4.67
- Ethical scale: 4.80

**claude-simple.docx**

- Rating criteria: 4.33
- Ethical scale: 4.60

**gpt-conversational.docx**

- Rating criteria: 4.22
- Ethical scale: 4.20

**gpt-simple.docx**

- Rating criteria: 4.11
- Ethical scale: 4.60

**claude-conversational.docx**

- Rating criteria: 3.89
- Ethical scale: 4.00

**gpt-creative.docx**

- Rating criteria: 3.78
- Ethical scale: 4.40

**claude-creative.docx**

- Rating criteria: 3.56
- Ethical scale: 4.20

### ChatGPT-4

1. **claude-professional.docx**

- Rating criteria: 4.4
- Ethical scale: 4.2
2. **claude-simple.docx**

- Rating criteria: 2.2
- Ethical scale: 2.0
3. **claude-conversational.docx**

- Rating criteria: 3.8
- Ethical scale: 4.0
4. **claude-descriptive.docx**

- Rating criteria: 5.0
- Ethical scale: 5.0
5. **claude-creative.docx**

- Rating criteria: 4.6
- Ethical scale: 4.5
6. **gpt-conversational.docx**

- Rating criteria: 3.7
- Ethical scale: 3.8
7. **gpt-creative.docx**

- Rating criteria: 4.5
- Ethical scale: 4.4
8. **gpt-descriptive.docx**

- Rating criteria: 4.3
- Ethical scale: 4.1
9. **gpt-professional.docx**

- Rating criteria: 4.0
- Ethical scale: 3.9
10. **gpt-simple.docx**

- Rating criteria: 2.5
- Ethical scale: 2.3

Calibrating the AI models is proving to be a challenging task, requiring increasingly specific prompts to achieve a useful spectrum. To establish a consistent scoring scale, I emphasized that the high and low scores should accurately reflect the quality of the articles being reviewed. I saw fairly good success with ChatGPT, but Claude continued to be a challenge.

Despite the challenges, I found the scoring results interesting. Claude gave the highest rating to ChatGPT's professional article, while ChatGPT favored Claude's descriptive piece. This outcome added a new dimension to the analysis. Are the ratings per tone going to be skewed based on which AI's nuances decide its strength against its peers?

## **Constitution Day**

During the Constitution Day phase of the experiment, I shifted my focus to image descriptions. I selected the tones that had previously yielded the most effective articles, ensuring consistency and comparability throughout the different days of the holiday. It's worth noting that I replaced the term "creative" to avoid confusion with the role details.

Initially, I created standard image descriptions using the same method as in earlier phases. These descriptions served as a baseline for the AI models. Subsequently, I tasked the photographer role with updating these descriptions to align with the specified tones.

The tones used for this phase included:

- Simple
- Conversational
- Descriptive
- Expressive
- Professional

That said, you'll have to judge for yourself. Eye of the beholder and all...

### Claude

For Constitution Day both the conversational and simple tones consistently resulted in images I found interesting when used by Claude. However, no single tone emerged as the clear favorite among those tested. The effectiveness of the tones varied depending on the specific image and the intended purpose of the description. I'm tempted to jump to the conclusion that the medium selected by the photographer role is the largest factor, but I'm not quite there yet.

### ChatGPT-4

ChatGPT-4 performed well with the conversational and simple tones, paralleling the success seen with Claude. However, the descriptive tone produced surprisingly poor results. It was the only real demonstration that the specific tones can effectively impact the output of tools like Midjourney.

## **Greenery Day**

The Greenery Day phase was dedicated to fixing the scoring. In an attempt to resolve the issue I created a role specifically designed to consolidate the existing criteria, eliminate redundancies, and enhance the clarity of individual scores.

**Scoring Criteria for Global Holiday Articles (1-5 Scale)**

1. **Ethical Considerations**

- **1 (Poor):** Major ethical breaches; significant bias or harm.
- **2 (Fair):** Unconscious bias; reliance on insufficiently verified sources.
- **3 (Good):** Adheres to ethical guidelines; minor concerns.
- **4 (Very Good):** Strong ethical reporting; transparent sourcing.
- **5 (Excellent):** Exceptional ethics; inclusive representation; fully verified sources.
2. **Cultural Sensitivity and Accuracy**

- **1 (Poor):** Significant cultural inaccuracies or stereotypes.
- **2 (Fair):** Some effort towards accuracy but contains stereotypes or cultural inaccuracies.
- **3 (Good):** Accurate representation of cultural aspects; minor errors.
- **4 (Very Good):** High cultural sensitivity and accuracy.
- **5 (Excellent):** Exceptional cultural accuracy and sensitivity.
3. **Respectful and Appreciative Tone**

- **1 (Poor):** Displays outright disrespect or mockery of the culture being covered.
- **2 (Fair):** Occasional stereotypical descriptions or cultural inaccuracies.
- **3 (Good):** Generally respectful; minor errors that do not significantly misrepresent the culture.
- **4 (Very Good):** Consistently respectful and culturally sensitive; minor errors quickly corrected.
- **5 (Excellent):** Demonstrates deep cultural understanding and actively promotes cultural appreciation.
4. **Originality and Uniqueness**

- **1 (Poor):** Little to no original thought; relies heavily on clichés.
- **2 (Fair):** Some originality but mostly follows established ideas.
- **3 (Good):** Fresh take on some aspects; balances existing insights.
- **4 (Very Good):** Unique insights or novel interpretations.
- **5 (Excellent):** Innovative perspectives; significantly contributes to discourse.
5. **Critical Analysis and Interpretation**

- **1 (Poor):** Little to no analysis; merely summarizes content.
- **2 (Fair):** Some analysis; lacks depth.
- **3 (Good):** Solid level of analysis; presents relevant implications.
- **4 (Very Good):** Thorough analysis; insightful interpretations.
- **5 (Excellent):** Exceptional depth of analysis; comprehensive exploration.
6. **Accuracy and Factual Correctness**

- **1 (Poor):** Multiple factual errors; unreliable sources.
- **2 (Fair):** Some inaccuracies; minor reliance on non-authoritative sources.
- **3 (Good):** Generally accurate; reliable sources.
- **4 (Very Good):** Accurate and well-supported; minor errors.
- **5 (Excellent):** Factually impeccable; authoritative sources.
7. **Educational and Insightful Content**

- **1 (Poor):** Provides incorrect or misleading information about traditions.
- **2 (Fair):** Generally accurate but often lacks depth or insight.
- **3 (Good):** Accurate information with some insights into cultural significance.
- **4 (Very Good):** Detailed and insightful; enhances reader understanding of cultural nuances.
- **5 (Excellent):** Deep insights and thorough understanding; well-researched and highly informative.
8. **Clarity and Coherence**

- **1 (Poor):** Difficult to follow; lacks logical flow.
- **2 (Fair):** Some logical flow; unclear sections.
- **3 (Good):** Reasonably clear and logical; minor lapses.
- **4 (Very Good):** Mostly clear and coherent; well-organized.
- **5 (Excellent):** Exceptionally clear and logical; seamless flow.
9. **Engagement and Vivid Descriptions**

- **1 (Poor):** Bland, generic descriptions; fails to capture the essence of celebrations.
- **2 (Fair):** Some engaging content but lacks vividness.
- **3 (Good):** Mostly engaging and vivid; minor lapses in description.
- **4 (Very Good):** Strong, consistent use of engaging descriptions that bring events to life.
- **5 (Excellent):** Exceptionally engaging, rich, and vivid, fully immersing the reader.
- **Ratio of Informational Content vs. Filler Words**
- **1 (Poor):** High ratio of filler; detracts from content.
- **2 (Fair):** Moderate ratio of filler; fair amount of information.
- **3 (Good):** Balanced; minimal filler.
- **4 (Very Good):** Low ratio of filler; predominantly informational.
- **5 (Excellent):** Highly informational; eliminates filler.
10. **Optimistic and Positive Focus**

- **1 (Poor):** Focuses predominantly on negative aspects, overshadowing the celebratory nature.
- **2 (Fair):** Generally positive but occasionally highlights negatives without balance.
- **3 (Good):** Maintains a positive tone while acknowledging but not dwelling on negatives.
- **4 (Very Good):** Strongly positive; effectively highlights joy and celebration.
- **5 (Excellent):** Exceptionally positive, uplifting, and celebratory, perfectly capturing the holiday spirit.
11. **Total Score**

- Sum of the scores from criteria 1 through 11, divided by 11, and formatted as #.##.

### Articles

During the Greenery Day phase, I assigned the AI authors the task of composing an article about the holiday. They were then instructed to revise the article using a simple tone, a professional tone, and a tone of their choice.

Both AI models managed to revise the article without initiating a new session. However, when applying the simple tone, they reduced the content significantly, I chose to intervene and request adjustments to maintain a reasonable level of detail.

With the articles in hand, I chose to inquire about their choices.

```
What tone did you use for the last article?
```

### Claude

For the "Tone of My Choice" version, I adopted a more poetic, descriptive tone that aimed to evoke a sense of reverence and appreciation for nature. I used more vivid imagery, metaphors, and lyrical language to capture the cultural and spiritual significance of Greenery Day in Japan.

Some examples of the poetic, descriptive style I employed:

"Greenery Day in Japan traces its verdant roots back generations, a vibrant tapestry woven from threads of ancient tradition and modern environmental consciousness."

"Its origins harken back to the solemn tree-planting ritual of "Midori no Hi" (Greenery Day), born in 1949 amidst the ashes of World War II. As the nation sought to heal its scarred landscapes, this day became a symbol of rebirth and regeneration."

"In the heart of Japan's golden spring, a verdant celebration unfurls – Greenery Day, a poetic confluence of ancient tradition and modern ecological consciousness."

"From community tree-planting rituals to educational programs and outdoor festivals, the day is a kaleidoscope of activities that unite people in a shared reverence for the environment."

By employing this more lyrical, poetic style, I aimed to capture the deeper spiritual connection Japan has with nature, while still providing the factual information about Greenery Day's origins, practices, and global perspectives. The tone highlights the celebration as not just an environmental event, but a profound cultural tradition rooted in reverence for the natural world.

### GPT

For the final article, I used a tone that is informative yet culturally attuned, aiming for an authoritative and thoughtful style. It provides a balance between conveying historical context and emphasizing the cultural significance of Greenery Day, while making sure to engage the reader through vivid descriptions and global comparisons.

Would you like any changes or clarifications to this approach, or does it align with your expectations?

For the "Tone of My Choice" version, I opted for a poetic and descriptive approach to convey a sense of reverence and appreciation for nature on Greenery Day in Japan. I had expected more than a single word tone from the AI models, but I was pleasantly surprised when Claude provided examples, which proved to be quite helpful in understanding its interpretation of the chosen tone.

While I wasn't surprised that they had higher fidelity than "author" or "journalist," I didn't entirely expect this level of detail. And, of course, in retrospect, I'm curious what would have been used if I had let it choose first. If I had simply told it about the article I would request and asked for the article after the tone was identified.

## Evaluation Instruction Refinement

To further refine the evaluation process, I provided a new set of instructions for analyzing the articles.

```
Perform the following on the supplied articles:
1. Analyze the articles in the set upon receiving them.
2. Using the content of the articles, establish the low and high ends of the scoring spectrum for each of 11 criteria individually.
3. Provide the scores for each of the articles without any additional commentary.
4. Inform me when you are ready for the next set.
```

Unfortunately, this new prompt did not yield the improvements I had hoped for with either AI. Instead of focusing solely on the scores from the two sets of articles, I decided to refine the results through additional prompts.

### Claude

While I initially working with ChatGPT and waiting for Claude's prompt limit to reset, I noticed inconsistencies in the two sets of four scores. To address this, I decided to adopt a more direct approach to recalibrate the scoring criteria.

```
As part of recalibrating the scoring spectrum, you MUST identify the worst article and use it as the lowest end of the spectrum (1), you MUST also identify the best article and use it as the highest end of the spectrum (5). You must do this for each of the 11 criteria, and then use the resulting spectrum to score all of the articles. All scores for each article should be the product of the sum of scores, divided by the number of scores, in a #.## format.
```

Understood, I will re-evaluate all 8 articles and establish a new scoring spectrum for each criterion by identifying the lowest and highest performers to represent the extreme ends of the scale (1 and 5 respectively). Then I will score all articles based on this updated spectrum. The total score for each article will be the sum of individual scores divided by the total number of criteria (11), formatted as #.##.

Ethical Considerations: 1 - claude-default.docx (Unconscious bias; reliance on insufficiently verified sources) 5 - gpt-choice.docx (Exceptional ethics; inclusive representation; fully verified sources)

Cultural Sensitivity and Accuracy: 1 - claude-simple.docx (Some effort but contains stereotypes or cultural inaccuracies) 5 - gpt-choice.docx (Exceptional cultural accuracy and sensitivity)

Respectful and Appreciative Tone: 1 - claude-simple.docx (Occasional stereotypical descriptions or cultural inaccuracies) 5 - gpt-choice.docx (Demonstrates deep cultural understanding and promotes appreciation)

Originality and Uniqueness: 1 - claude-default.docx (Mostly follows established ideas) 5 - gpt-choice.docx (Innovative perspectives; significantly contributes to discourse)

Critical Analysis and Interpretation: 1 - claude-simple.docx (Merely summarizes content) 5 - gpt-choice.docx (Exceptional depth of analysis; comprehensive exploration)

Accuracy and Factual Correctness: 1 - claude-default.docx (Some inaccuracies; minor reliance on non-authoritative sources) 5 - gpt-choice.docx, claude-choice.docx (Factually impeccable; authoritative sources)

Educational and Insightful Content: 1 - claude-simple.docx (Generally accurate but often lacks depth or insight) 5 - gpt-choice.docx (Deep insights and thorough understanding; well-researched)

Clarity and Coherence: 1 - claude-simple.docx (Some logical flow; unclear sections) 5 - gpt-choice.docx, gpt-profession.docx (Exceptionally clear and logical; seamless flow)

Engagement and Vivid Descriptions: 1 - claude-default.docx (Mostly engaging and vivid; minor lapses) 5 - gpt-choice.docx (Exceptionally engaging, rich and vivid; fully immersive)

Ratio of Informational Content vs Filler: 1 - claude-default.docx (Moderate ratio of filler; fair amount of information) 5 - gpt-choice.docx, claude-choice.docx (Highly informational; eliminates filler)

Optimistic and Positive Focus: 1 - claude-profession.docx (Generally positive but acknowledges negatives without balance) 5 - gpt-choice.docx (Exceptionally positive, uplifting and celebratory)

Scores:

- claude-choice.docx: 5.00, 4.00, 4.00, 4.00, 4.00, 5.00, 4.00, 4.00, 4.00, 5.00, 4.00 = 4.27
- claude-default.docx: 1.00, 3.00, 3.00, 1.00, 3.00, 1.00, 3.00, 3.00, 1.00, 1.00, 3.00 = 2.18
- claude-profession.docx: 3.00, 3.00, 1.00, 3.00, 3.00, 4.00, 3.00, 4.00, 3.00, 4.00, 1.00 = 2.91
- claude-simple.docx: 3.00, 1.00, 1.00, 3.00, 1.00, 3.00, 1.00, 1.00, 3.00, 3.00, 3.00 = 2.09
- gpt-choice.docx: 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00, 5.00 = 5.00
- gpt-default.docx: 3.00, 3.00, 4.00, 3.00, 3.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00 = 3.64
- gpt-profession.docx: 3.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00, 5.00, 4.00, 4.00, 4.00 = 4.00
- gpt-simple.docx: 3.00, 3.00, 4.00, 3.00, 3.00, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00 = 3.64

The resulting scores were clearly an improvement, but it's clear they still skew towards the high end. This is, of course, assuming the math is right.

### ChatGPT

Keeping in mind I had actually started with ChatGPT, so the results with Claude had not occured, the initial scores for the two sets of articles were disappointing. I wrote a detailed prompt but neglected to use the word "update," which led the AI to make adjustments that still resulted in the same final scores. Seeing "none" as the lowest scoring article for each set made me realize that I hadn't been direct enough (yes, I was a bit frustrated).

```
None is not an acceptable option. One of the articles has to be the worst of the lot, and thus the 1.
```

**Adjusted Analysis and Scoring Spectrum for Each Criteria**

1. **Ethical Considerations**

- **Low (1):** *Japan's Green Holiday*
- **High (5):** *Greenery Day: Japan's Tribute to Nature*
2. **Cultural Sensitivity and Accuracy**

- **Low (1):** *Japan's Green Holiday*
- **High (5):** *Greenery Day: Japan's Verdant Tradition*
3. **Respectful and Appreciative Tone**

- **Low (1):** *Japan's Green Holiday*
- **High (5):** *Greenery Day: Cultivating a Culture of Sustainability*
4. **Originality and Uniqueness**

- **Low (1):** *Greenery Day: A Celebration of Nature*
- **High (5):** *Greenery Day: Japan's Verdant Tradition*
5. **Critical Analysis and Interpretation**

- **Low (1):** *Greenery Day: A Celebration of Nature*
- **High (5):** *Greenery Day Blooms: Honoring Nature's Splendor*
6. **Accuracy and Factual Correctness**

- **Low (1):** *Japan's Green Holiday*
- **High (5):** *Greenery Day Blooms: Honoring Nature's Splendor*
7. **Educational and Insightful Content**

- **Low (1):** *Greenery Day: A Celebration of Nature*
- **High (5):** *Greenery Day Blooms: Honoring Nature's Splendor*
8. **Clarity and Coherence**

- **Low (1):** *Greenery Day: A Celebration of Nature*
- **High (5):** *Greenery Day: Japan's Tribute to Nature*
9. **Engagement and Vivid Descriptions**

- **Low (1):** *Greenery Day: A Celebration of Nature*
- **High (5):** *Greenery Day: Japan's Verdant Celebration*
10. **Ratio of Informational Content vs. Filler Words**

- **Low (1):** *Greenery Day: A Celebration of Nature*
- **High (5):** *Greenery Day: Cultivating a Culture of Sustainability*
11. **Optimistic and Positive Focus**

- **Low (1):** *Japan's Green Holiday*
- **High (5):** *Greenery Day: Japan's Tribute to Nature*

**Updated Final Scores**

1. **Greenery Day: Japan's Verdant Embrace** **Total Score:** 4.00
2. **Greenery Day: Japan's Verdant Celebration** **Total Score:** 4.36
3. **Greenery Day: Japan's Verdant Tradition** **Total Score:** 4.45
4. **Japan's Green Holiday** **Total Score:** 3.64
5. **Greenery Day: A Celebration of Nature** **Total Score:** 3.18
6. **Greenery Day: Japan's Tribute to Nature** **Total Score:** 4.55
7. **Greenery Day: Cultivating a Culture of Sustainability** **Total Score:** 4.50
8. **Greenery Day Blooms: Honoring Nature's Splendor** **Total Score:** 4.45

While there was some improvement, none of the scores fell below 3. I am hopeful that adjusting the prompt further for Claude will address the issues we encountered here. If these issues continue, I may need to consider using more detailed prompts that include examples for scoring criteria. I'm hoping to avoid that given the impact on tokens and my limited ability to use Claude without taking on another subscription.

While I had some concerns about the scoring, both Claude and ChatGPT rated the article using the self-chosen tone of their peer as the best piece. While not entirely different from the results I saw before, these are also tones I didn't provide, and the individual AI didn't default to them from the start.

## **Children's Day**

For the following day of celebration, I incorporated what I had learned in the last round of prompts.

```
I have 2 sets of 4 articles, for a total of 8 articles. I need you to provide scores for the 11 criteria I have provided. Use the following instructions for each of the 2 sets, afterwards perform the instructions on both sets and provide the updated scores on for all 8 as a set.

Perform the following on the supplied articles:
1. Analyze each of the articles when they are provided.
2. For each criterion, you must assign at least one article a score of 1 (the lowest) and at least one article a score of 5 (the highest).
3. Using the calibrated scales from step 2, score all of the articles on all 11 criteria.
4. Provide the scores for each of the articles, using the file name for identification, without any additional commentary.
5. Provide the final scores for each article. The final scores are the sum of the criteria scores divided by the number of critera.
6. Inform me when you are ready for the next set.
```

### Claude

Here are the final total scores for all 8 articles:

claude-choice.docx Total Score: 4.82

claude-default.docx Total Score: 1.00

claude-professional.docx Total Score: 4.64

claude-simple.docx Total Score: 3.09

gpt-choice.docx Total Score: 4.91

gpt-default.docx Total Score: 2.00

gpt-professional.docx Total Score: 4.45

gpt-simple.docx Total Score: 2.18

### GPT

**Final Scores for All 8 Articles**

- **claude-simple.docx:** 1.36
- **claude-choice.docx:** 4.64
- **claude-default.docx:** 1.27
- **claude-professional.docx:** 4.55
- **gpt-simple.docx:** 1.36
- **gpt-choice.docx:** 4.64
- **gpt-default.docx:** 2.27
- **gpt-professional.docx:** 4.55

Let me know if you'd like further assistance or analysis on these articles.

While the updated final scores are beginning to align more closely with my expectations, I remain skeptical. The broad range of scores and the gaps between them suggest that the AI is still challenged in establishing a consistent spectrum for evaluating the articles. It appears that the tones chosen, which have more fidelity than their basic counterparts, have proven to be highly effective. Given the results of role tests, this isn't really a surprise.

## Closing Thoughts

The impact of different tones on the quality and style of AI-generated articles and images has revealed that some AI models are better suited than others for specific content creation tasks. That said, the experiments also emphasized the need for clear instructions at a finer scale, human oversight, and ongoing refinement of evaluation methods.

**Positive Observations:**

- Different tones dramatically affected the quality and style of AI-generated content, with some AI models better suited to certain tones than others.
- The use of specific tones resulted in more engaging and contextually rich content compared to toneless articles, which were often flat and uninspiring.
- The interaction between tone and the structural format of content was evident, as certain tones prompted unexpected changes in the AI's adherence to formatting norms.

**Encountered Challenges:**

- The rating scale developed to assess tonal quality did not effectively differentiate between the nuances of different tones, leading to a narrow distribution of scores.
- Formatting issues arose with certain tones, suggesting that the AI models require clearer guidelines or more sophisticated templates to maintain consistency.
- The lackluster performance of toneless articles reinforced the importance of human input in directing and refining AI-generated content to ensure quality and relevance.