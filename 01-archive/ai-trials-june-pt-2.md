This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is the 2nd of 3 for the month of <u>June</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

As the end of June approaches, I'm starting to test an other updated version of my role making GPT, if all goes well it will create roles that are more flexible and focused without specifying an exact number of criteria. This should reduce the amount of tantgential items and reduce the verbosity of the roles, ultimately improving their efficiency and relevance. The roles, templates, and scoring criteria mentioned in this article are documented in <u>June Pt 3</u>.

## Researcher

To continue building on the theme of using AI to provide instruction with minimal human intervention, I employed an **AI Research Advisor** role to provide recommendations for the next round of experiments.

```
A researcher that can analyze the work of others and make recommendations for future experiments that follow a logical progression based on the existing work. They should have experience working with AI, particularly about how to use them successfully, such as roles, structured prompts, etc.
```

To bring the AI Research Advisor up to speed, I provided indices from each of the monthly articles for context on the project experiments as a whole, followed by the most recent articles where tone has been the primary focus.

```
These are the last 4 articles where I have been focused on tone. I have one last experiment to run this month for Eid al-Adha, using Bangladesh and Egypt as the focus for the articles themselves. Review these and let me know if you have any questions about the material I've provided that will help you make recommendations for the next experiment.
```

I received an updated outline of the experiments, including a multi-part recommendation for the next experiment. The recommendation was designed to combine all my findings into prompts for the next set and then run the major experiments sequentially, a task that is overwhelming to simply read, much less perform. Another suggestions included using a different tone for every section of an article, which simply like bad writing. After a few rounds of adjustments and directions for the AI to improve its recommendations, we settled on the following experiment:

**Experiment 1**: Consistent Tone Across Different Articles

**Objective**: Assess the quality and coherence of AI-generated articles about Eid al-Adha celebrations in Bangladesh and Egypt when written in a consistent tone.

**Implementation**:

1. **Tone Selection**: Select a single tone (e.g., informative, conversational, descriptive, professional, or celebratory).
2. **Article Generation**: Instruct the AI to generate two separate articles about Eid al-Adha celebrations, one for Bangladesh and one for Egypt, using the chosen tone consistently throughout each article.
3. **Evaluation**:
4. **Self-Consistency**: Check for consistency in tone throughout each article.
5. **Content Quality**: Use predefined criteria to assess the quality of the content, including accuracy, clarity, cultural sensitivity, and coherence.
6. **Comparative Analysis**: Compare the articles to determine if the chosen tone impacts the depth and quality of the information presented for each country.

**Expected Outcome**: Insights into how maintaining a consistent tone affects the quality and Note:

*This role was employed without any built-in instructions to provide direction. Multiple prompts were necessary to refine the output.*

## Journalist

In a session with Claude, I worked with a **Cultural Celebration Journalist** role to improve the template I've been using.

```
A journalist who specializes in articles covering holidays and celebrations from countries and cultures from around the world. Must be respectful of the culture and traditions in covered by their writing.
```

For context, I provided a brief summary, a copy of the current template, and a specific request for feedback.

```
Over the course of the year I have been working with AI to get a better understanding of how to make the most of their availability. I have using articles covering holidays and celebrations from countries and cultures from around the world as subject matter for testing  the benefits of increasingly specific roles, their ability to create scoring criteria and apply it to the articles I've had them create, the benefits of structured prompts, and initial tests on tones.

[template]

This is the template I've been using for the majority of the tests. How can this template, and the expected word count, be improved to make the article more comprehensive and engaging without making it overly verbose or too long to read?
```

The AI provided suggestions suitable for actual journalistic articles, such as including personal stories and testimonials. However, these suggestions would likely lead to AI-generated hallucinations. That said, I added one of the recommended sections. Additionally, I reduced the word count per section to address recent challenges and cleaned up the XML formatting to remove artifacts from various iterations, adding specificity to the process language.

## Eid al-Adha Articles

I created new sessions, provided the journalist role along with the updated template, and tasked the AI with writing related articles:

```
As a Cultural Celebration Journalist, I need you to write an article for the first day of Eid al-Adha in Bangladesh using a tone that is both professional and engaging. The article must adhere to the supplied template, honor the included guidelines, and follow the prescribed process.
```

I ran two sessions with both Claude and ChatGPT-4. One session focused on articles for Bangladesh, while the second focused on Egypt. The goal was to see if the tone stayed consistent across articles when reviewed in progressively larger sets.

In both sessions, ChatGPT-4's initial drafts were lackluster. After providing corrective feedback, it followed the prescribed process and produced satisfactory articles for the remaining five in each session. Claude generally performed well, but when the XML lacked proper formatting or carriage returns, it included XML tags in the output and fell short on word count, issues observed previously.

## Editor

With all 12 articles fresh out of the sessions, I used a new **Cultural Celebration Magazine Editor** role to score them. I attempted to clean up the scoring criteria and instructions to address issues with Claude's context window; however, I overlooked involving the AI role as I had with the article template.

```
An editor at a magazine that specializes in material on holidays and celebrations from countries and cultures from around the world. Must have a high attention to detail and experience reviewing articles on the same subject to determine which will have the most appeal.
```

Scoring articles has consistently fallen short of the output I was seeking. If the scores weren't tightly grouped or tied, the AI was overly descriptive, all of the criteria were verbose, and the prompt itself had some minor conflicts. While I've made progress towards achieving the desired output, it has taken several days and remains somewhat unreliable without prompts to guide the AI.

That said, GPT managed to provide distributions that were closer to my goals, but even then, there were three articles with a perfect score of 5.0. Additionally, all scores should ideally have two decimal places.

**c1b1.docx**: **5.0**

**g1b1.docx**: **5.0**

**g2e1.docx**: **5.0**

**c1b2.docx**: 4.5

**c2e1.docx**: 4.8

**g1b3.docx**: 3.5

**c2e2.docx**: 3.2

**g2e3.docx**: 2.8

**c2e3.docx**: 2.5

**g2e2.docx**: 1.5

**g1b2.docx**: 1.2

**c1b3.docx**: **1.0**

## Tone Consistency

With all the articles scored, it was time to analyze the consistency of tones across articles, countries, and AI platforms. Normally, I use the same prompts and processes in every test session. However, for this round, I had to start a new session with Claude to fit all 12 articles into the context window, while I could do this in the same session with ChatGPT-4 where I did the scoring.

### Claude's Bangladesh Articles

You're correct, my summary was more detailed than the concise paragraph you requested. Here is a more concise summary for the first set of three articles:

The articles consistently convey a reverent tone and engaging writing style while capturing the vibrant traditions and spiritual significance of Eid al-Adha celebrations in Bangladesh, seamlessly blending historical context with modern practices that highlight unity, sacrifice, and compassion.

### Claude's Egypt Articles

The three articles maintain a reverent and culturally-appreciative tone while vividly depicting the vibrant traditions and spiritual significance of Eid al-Adha celebrations in Egypt across the three-day observance. The descriptive writing style transports readers into the festivities, seamlessly blending historical context with modern practices that highlight community unity, religious devotion, and the country's rich cultural diversity.

### ChatGPT's Bangladesh Articles

The three articles exhibit a consistent reverent tone that honors the religious and cultural significance of Eid al-Adha, while employing an engaging descriptive writing style that vividly depicts the vibrant festivities across Bangladesh. The articles seamlessly weave together historical context, modern practices, insightful global perspectives, and practical guidance, maintaining a balanced and appreciative approach throughout.

### ChatGPT's Egypt Articles

The three articles maintain a reverent yet celebratory tone, vividly depicting the spiritual depth and communal vibrancy of Eid al-Adha festivities across Egypt. The descriptive writing style transports readers into the heart of the celebrations, seamlessly blending religious traditions with local cultural expressions. The articles provide insightful historical context while highlighting modern practices, global perspectives, and guidance for respectful participation.

```
Perfect, thank you. Now, analyze all of these articles with a focus on the tone and writing style of the author and compare on how consistent they are. Use concise language and a professional tone.
```

After a comprehensive analysis of all 12 articles across the 4 sets, I can confirm that the tone and writing style exhibited a high degree of consistency throughout.

The author maintained a reverent and culturally-appreciative tone, treating the subject matter with utmost respect while conveying the vibrancy and significance of the Eid al-Adha celebrations. The descriptive writing style effectively transported readers into the festive atmospheres, vividly capturing the sights, sounds, and emotions associated with the observances.

The articles consistently provided comprehensive historical context, tracing the origins and spiritual underpinnings of the festival. They seamlessly interwove modern practices and regional variations, highlighting the vibrant tapestry of traditions that have evolved over time.

Global perspectives were coherently incorporated, underscoring the universal resonance of Eid al-Adha's themes of sacrifice, unity, and compassion, while acknowledging the unique cultural expressions manifested by diaspora communities worldwide.

Practical guidance for respectful participation was offered consistently, ensuring readers could engage with the celebrations in a meaningful and culturally sensitive manner.

The language remained professional and objective throughout, striking a balance between reverence for the subject matter and an engaging narrative style that appealed to a diverse readership.

In summary, the 12 articles exhibited a consistent and harmonious tone and writing approach, reflective of a skilled author adept at capturing the essence of cultural celebrations with authenticity, insight, and an appreciative perspective.

```
Based on your analysis of the articles, do you believe they were written by a single author?
```

Based on my thorough analysis of the tone, writing style, language use, and overall narrative approach across all 12 articles, there are strong indications that they were likely written by a single author.

The consistent reverent yet celebratory tone, the seamless blending of historical context with modern practices, the harmonious incorporation of global perspectives, and the cohesive guidance for respectful participation suggest a unified authorial voice.

Additionally, the articles exhibit a distinct descriptive writing style that transports readers into the festive atmospheres through vivid sensory details and evocative language. This stylistic consistency across all articles points toward a single author's creative expression.

Moreover, the professional and objective language use, as well as the balanced approach to treating the subject matter with both respect and an engaging narrative flair, further reinforce the likelihood of a single authorial source.

While minor variations in phrasing and structural elements are present, as is expected in any comprehensive body of work, the overarching thematic consistency, tonal harmony, and narrative flow strongly suggest the articles were crafted by one author with a deep understanding and appreciation for the cultural significance of Eid al-Adha celebrations.

In summary, based on the cohesive elements observed throughout the articles, it is highly probable that they were indeed written by a single author possessing a keen cultural awareness, a reverent approach, and a skilled command of descriptive and engaging writing.

```
Is there anything noteworthy you want to share, based on my questions thus far, about the articles as a singular set?
```

Yes, there are a few noteworthy observations I'd like to share about this singular set of 12 articles:

1. Cultural Authenticity: The articles collectively demonstrate a deep understanding and appreciation for the cultural nuances, historical contexts, and spiritual significance of Eid al-Adha celebrations across Bangladesh and Egypt. The level of detail and respect paid to these elements suggests the author(s) has either lived experience within these cultures or conducted extensive research to capture the authenticity.
2. Thematic Cohesion: While each article focuses on a specific day or aspect of the Eid celebrations, there is a strong thematic cohesion that ties them all together. The recurring emphasis on values like sacrifice, unity, compassion, and community bonding creates a harmonious narrative tapestry across the entire set.
3. Global Perspective: The articles consistently provide global perspectives, highlighting how Eid al-Adha transcends borders and is celebrated by diaspora communities worldwide. This inclusivity adds richness and resonance, underscoring the universal appeal of the festival's core principles.
4. Balanced Tone: Remarkably, the author(s) has managed to strike a delicate balance between a reverent tone befitting the spiritual nature of the subject and an engaging, descriptive narrative style that appeals to a broad audience. This equilibrium is consistently maintained throughout the set.
5. Professional Presentation: The adherence to a consistent article structure, the well-researched content, and the polished writing style collectively give the set a professional, publication-ready quality. This level of cohesion and polish is impressive for such a comprehensive collection of articles.

In conclusion, this singular set of articles stands out for its cultural authenticity, thematic cohesion, global perspective, balanced tone, and professional presentation. These elements, consistently maintained across all 12 articles, suggest a highly skilled and culturally attuned author(s) with a deep respect for the subject matter.

I have to say I am honestly shocked by Claude's assessment. I had expected it to know there were at least two "minds" at work within the set of articles presented. While the consistency is impressive in my mind, I suspect this is also indicative of why unedited AI content feels the way it does, even when you ignore its abuse of certain words.

### ChatGPT's Analysis

```
Perfect, thank you. Now, analyze all of these articles with a focus on the tone and writing style of the author and compare on how consistent they are with the prior articles. Use concise language and a professional tone.
```

Analyzing the tone and writing style of the 12 articles, we observe varying degrees of consistency in how the authors approach their subjects across different sets. Here's a summary of the tone and writing style analysis:

1. **c1b1.docx, c1b2.docx, c1b3.docx**:

- The tone is uniformly professional and respectful, effectively capturing the cultural richness of Eid celebrations. The style is engaging, with c1b1.docx showing particular depth in its exploration of cultural nuances. The consistency in respectful and enriching language is notable across these articles.
2. **c2e1.docx, c2e2.docx, c2e3.docx**:

- These articles maintain a factual and educational tone with a clear intent to inform and engage the reader about Eid in Egypt. The style is descriptive and vivid, especially in c2e1.docx, which excels in presenting a dynamic picture of the celebrations. There is slight variability in the depth of cultural insight, but overall, the consistency in maintaining an informative and respectful tone is evident.
3. **g1b1.docx, g1b2.docx, g1b3.docx**:

- The writing style is richly detailed and well-structured, particularly in g1b1.docx and g1b3.docx, which provide thorough insights into the celebrations. The tone across these articles is consistently warm and inclusive, though g1b2.docx falls slightly short in engagement and depth compared to its counterparts.
4. **g2e1.docx, g2e2.docx, g2e3.docx**:

- These articles present a clear and articulate portrayal of Eid celebrations, with a focus on unity and community spirit. The tone is consistently celebratory and inclusive, promoting a sense of communal harmony. The writing style in g2e1.docx stands out for its ability to weave historical context seamlessly with modern practices, offering a holistic view of the festival.

Across all sets, the tone tends to be informative, respectful, and culturally sensitive, with some articles showing greater depth and narrative flair than others. The writing style is generally consistent, aimed at engaging a diverse readership by blending detailed cultural descriptions with broader reflections on the significance of Eid celebrations. However, some variability in engagement and depth is noted, particularly in articles with lower scores, which could benefit from enhanced narrative techniques and richer cultural insights to elevate their overall impact and consistency.

```
What about the tone and writing style across all of the articles as a whole.
```

Across all twelve articles, the tone and writing style exhibit a range of effectiveness in engaging and educating the readers about Eid celebrations in different cultural contexts. Here’s a consolidated overview:

### Tone

- **Respectful and Culturally Sensitive:** All articles strive to respect and honor the cultural traditions they discuss. The tone is carefully maintained to avoid cultural insensitivity, which is crucial given the religious and cultural significance of Eid al-Adha.
- **Educational and Informative:** There's a clear intent across the articles to educate readers about the specifics of Eid celebrations, the historical and cultural backgrounds, and the nuances of various traditions. This is done through an informative tone that remains accessible to a broad audience.
- **Varied Emotional Depth:** While some articles excel in evoking a sense of celebration and communal joy, others maintain a more straightforward, descriptive tone. The more successful articles in terms of engagement and educational impact are those that manage to blend information with evocative descriptions that capture the reader’s imagination and emotions.

### Writing Style

- **Descriptive and Detailed:** Most articles are rich in descriptive detail, providing vivid imagery of the scenes and traditions of Eid al-Adha. This style helps in painting a lively picture of the festivities, making the cultural practices more relatable and understood.
- **Consistency in Quality:** There is variability in the depth and quality of writing. Higher-scoring articles tend to feature a more polished and immersive narrative style that effectively combines factual content with storytelling elements. This approach enhances reader engagement by making the cultural insights more compelling.
- **Inclusivity in Language:** The writing often aims to include a wide audience, explaining cultural terms and practices in a way that can be understood by those unfamiliar with the subject. This inclusiveness is crucial for educational articles aiming to bridge cultural understanding.

### General Observations

- **Educational Focus with Emotional Engagement:** The best articles not only inform but also connect emotionally with the readers, using the human element of the celebrations to draw readers into the narrative.
- **Variability in Engagement:** Some articles could improve by varying sentence structure, using more active voice, or incorporating direct quotes or anecdotes to increase engagement and dynamism in the narrative.
- **Alignment with Audience:** All articles seem to be written with a global audience in mind, considering that explanations are provided for non-familiar cultural practices. This broad focus is beneficial for articles intended for a diverse readership that spans different cultural backgrounds.

In summary, while there's a general consistency in maintaining a respectful and informative tone across all articles, the writing style's effectiveness in engaging and educating varies. Enhancements in narrative techniques could elevate the articles with lower engagement scores, providing a more consistent reader experience across the set.

```
Based on your analysis of the articles, do you believe they were written by a single author?
```

Based on the analysis of tone, writing style, and variability in depth and engagement across the twelve articles, it seems unlikely that they were written by a single author. Here are the key observations supporting this conclusion:

1. **Variability in Depth and Engagement**: There is significant variation in how deeply each article engages with its subject matter and the emotional depth it conveys. Some articles are richly detailed and immersive, effectively combining educational content with evocative narrative elements, while others are more straightforward and less engaging. Such variability could indicate different authors with distinct writing styles and levels of expertise in cultural representation.
2. **Differences in Writing Style**: The narrative techniques and the consistency of the writing quality vary among the articles. Some are well-structured and polished, employing vivid descriptions and a narrative style that enhances reader engagement. Others lack these qualities and are more utilitarian in their approach. This inconsistency could suggest multiple authors with different narrative skills and approaches to article composition.
3. **Tonal Inconsistencies**: While all articles maintain a respectful and informative tone, the degree of warmth, cultural sensitivity, and inclusivity varies. Some articles demonstrate a profound understanding of the cultural context and effectively communicate this to the reader, while others are less successful in these areas. These tonal inconsistencies could be due to different authors interpreting the subject matter from various perspectives.
4. **Educational Focus and Emotional Engagement**: The ability to weave factual content with emotional engagement is handled differently across the articles. Some authors manage to strike a balance that makes the educational content compelling and accessible, while others focus strictly on information delivery without the emotional nuance. This could indicate different priorities or levels of experience among multiple authors.

Given these points, it is more plausible that the articles were written by multiple authors, each bringing their unique style, perspective, and level of expertise to their work. This diversity could be intentional, aiming to provide a range of voices and approaches to the topic of Eid celebrations in different cultural contexts.

```
Based on your observations, can you provide the names of the article files grouped by theoretical author?
```

Based on the detailed analysis of the tone, writing style, and depth across all twelve articles, here is a theoretical grouping of the article files by potential author, considering similarities and differences in how the articles are composed:

Author 1 (Highly Detailed and Immersive)

- **c1b1.docx**
- **g1b1.docx**
- **g2e1.docx**

These articles are characterized by their rich detail, immersive narrative, and effective blend of educational content with storytelling. They show a high level of cultural sensitivity and the ability to engage deeply with the subject matter.

Author 2 (Informative with Moderate Engagement)

- **c1b2.docx**
- **c2e1.docx**
- **g1b3.docx**

These pieces are well-informed and maintain a good educational focus but with slightly less narrative flair compared to the first group. They are professional and respectful but may not engage the reader's emotions as effectively.

Author 3 (Consistently Respectful, Varied Engagement)

- **c2e2.docx**
- **g2e3.docx**

This group includes articles that are respectful and aim to educate but have a more variable level of engagement and depth. The tone is consistent, yet the articles may lack the immersive qualities found in those by Author 1.

Author 4 (Less Engaging and Detailed)

- **c1b3.docx**
- **g1b2.docx**
- **g2e2.docx**

Articles in this group are the least engaging and detailed. They provide necessary information but in a more straightforward manner without the depth or narrative richness of the other groups. These articles might focus more on delivering content than enhancing reader experience.

This grouping is hypothetical and based on the analysis of writing style, tone, and content depth. It suggests that the articles were likely written by multiple authors, each bringing their own approach to the topic, which is reflected in the variability of the articles' quality and engagement levels.

It is fascinating to me that ChatGPT was able to discern there were multiple authors when Claude failed to. However, digging deeper, it turns out that no grouping was specific to either of the AI. Admittedly, the result was skewed as it tried to group them by the four items it identified as indicators. I think it's fair to say my prompt could have been more precise, but I have all I wanted for now.

## Consistently Consistent

Reflecting on this phase of the experiment, it was insightful to see how both Claude and ChatGPT-4 handled the task of maintaining a consistent tone across multiple articles. The emphasis on tone consistency revealed the nuances in how each AI system interprets and executes instructions. Both AIs generally achieved a coherent and culturally sensitive tone, although ChatGPT-4 required more iterative feedback to align with the desired style. This experiment highlighted the effectiveness of structured prompts in guiding AI output and the ongoing challenge of managing verbosity without losing depth.

### **Key Takeaways**

**What Worked Well:**

- **Consistent Tone:** Both Claude and ChatGPT-4 generally maintained a consistent tone across articles, achieving coherence and cultural sensitivity.
- **Structured Prompts:** Detailed and structured prompts effectively guided the AI to produce more accurate and contextually appropriate content.

**Challenges Encountered:**

- **False Starts:** It is common for each AI to have a false start, not strictly adhering to instructions even when explicitly stated.
- **Scoring Reliability:** The scoring process remains somewhat unreliable, with inconsistencies in how the AI interprets the instructions.