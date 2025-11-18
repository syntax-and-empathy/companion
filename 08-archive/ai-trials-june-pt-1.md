This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This article is the 1st of 3 for the month of <u>June</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

As a follow-up to the tone experiments in <u>May Pt. 2</u>, I wanted to explore the AI's ability to choose its own tone versus working without a specified tone or one I had requested. Focusing on the role of tone in articles about cultural celebrations, I tasked ChatGPT-4 with establishing a plan for testing different tones in articles about the Dragon Boat Festival and Brazilian Valentine's Day.

The AI provided a step-by-step process with detailed prompts for each tone to generate the articles. Additionally, I continued to refine the scoring methodology and addressed issues identified in prior experiments.

## AI Prompting AI

To begin, I instructed GPT-4 to generate outlines for each day's experiments. Much of this content was derived from prior articles I had shared from May for context. GPT-4 then provided five specific tones to be used in the Dragon Boat Festival experiment:

- Informative/Explanatory
- Engaging/Conversational
- Descriptive/Creative
- Professional/Formal
- Optimistic/Celebratory

I then had GPT write the actual prompts which I used with the author role and template I've been working with.

### Informative/Explanatory

```
Write an article about the Dragon Boat Festival in an informative and explanatory tone. Focus on providing clear, factual information about the festival's origins, historical significance, and modern-day practices. Explain the importance of dragon boat racing and the preparation of zongzi (rice dumplings). Include details about key rituals and the cultural impact of the festival in China.
```

### Engaging/Conversational

```
Write an article about the Dragon Boat Festival in an engaging and conversational tone. Describe the exciting dragon boat races, the delicious zongzi, and the lively atmosphere of the festival. Imagine you are talking to a friend who has never heard of the Dragon Boat Festival and make it fun and accessible. Share interesting facts and stories to draw the reader in.
```

### Descriptive/Creative

```
Write an article about the Dragon Boat Festival in a descriptive and creative tone. Use vivid descriptions and sensory language to paint a picture of the colorful dragon boats slicing through the water, the aroma of freshly made zongzi, and the sounds of festive music and cheering crowds. Highlight the historical origins and cultural significance of the festival with imaginative and metaphorical language.
```

### Professional/Formal

```
Write an article about the Dragon Boat Festival in a professional and formal tone. Provide a detailed analysis of the festival's historical background, cultural importance, and current practices. Discuss the significance of dragon boat racing and the preparation of zongzi in a respectful and scholarly manner. Use formal language suitable for an academic or professional audience.
```

### Optimistic/Celebratory

```
Write an article about the Dragon Boat Festival in an optimistic and celebratory tone. Focus on the joyful and festive aspects of the event, highlighting the excitement of the dragon boat races and the communal spirit of preparing and sharing zongzi. Emphasize the themes of unity, tradition, and cultural pride that the festival embodies. Capture the lively and uplifting atmosphere of the celebrations.
```

Claude and GPT-4o ignored the section-by-section instructions, producing complete articles, while Gemini followed the process step-by-step. This marked Gemini's first time creating a full article without major issues, despite struggling with the word count.

```
I would like to remind you that you are expected to follow the prescribed process, and wait for instruction to continue between each step.

[article prompt]
```

I reminded the AI models to follow the process and wait for instructions between each step.

- **Claude** performed as expected.
- **GPT-4** displayed unusual behavior by initially writing everything except the introduction.
- **Gemini** started with the introduction but failed to meet the word count by a considerable amount. Interestingly, it was the only model to update the titles and subtitles as requested in the instructions, a detail I had given up on enforcing without removing the requirement altogether.

### Scoring and Refinement

After generating the Dragon Boat Festival articles, I had all three AI models score the 15 articles they had produced. As the scoring rounds progressed, challenges emerged with each model:

- **Claude** initially performed well but struggled to maintain a full range of scores from 1 to 5.
- **GPT-4** showed significant improvement after receiving more detailed instructions, though it initially produced narrow score variations.
- **Gemini’s** performance was erratic and inconsistent, often failing to meet the expected format despite more detailed instructions showing limited success.

To address these issues, I adjusted the prompts, a process that highlighted the importance of breaking down complex tasks into smaller, more manageable steps.

## A More Sophisticated Approach

Building on lessons from the Dragon Boat Festival experiment, I requested the AI to propose a more sophisticated approach for Dia dos Namorados (Brazilian Valentine's Day) than it had for the prior experiment. The resulting prompts involved different tones as well as incorporating specific storytelling techniques and cultural insights.

### **Informative/Explanatory**

```
Write an article about Brazilian Valentine's Day (Dia dos Namorados) in an informative and explanatory tone. Use clear subheadings and bullet points to provide detailed information on the following aspects:
- Historical origins and evolution of the holiday
- Traditional customs and modern practices
- Typical gifts and activities
- Cultural significance and regional variations
Ensure the content is factually accurate and structured for easy reading.
```

### **Engaging/Conversational**

```
Write an article about Brazilian Valentine's Day (Dia dos Namorados) in an engaging and conversational tone. Start with a captivating anecdote or dialogue that draws the reader in. Use a warm and friendly style to describe:
- How couples in Brazil celebrate the day
- Personal stories or hypothetical scenarios
- Popular gifts and activities
- The cultural importance of love and romance in Brazil
Make the reader feel as if they are having a friendly chat about the holiday.
```

### **Descriptive/Creative**

```
Write an article about Brazilian Valentine's Day (Dia dos Namorados) in a descriptive and creative tone. Use vivid imagery and metaphors to bring the celebration to life. Focus on:
- The sights, sounds, and scents of the day
- Romantic scenes and festive atmospheres
- The preparation and exchange of gifts like flowers and chocolates
- The emotional and cultural significance of the holiday
Create an immersive experience for the reader through rich and imaginative descriptions.
```

### **Professional/Formal**

```
Write an article about Brazilian Valentine's Day (Dia dos Namorados) in a professional and formal tone. Provide a scholarly analysis that includes:
- Historical and sociological perspectives on the holiday
- Analysis of its cultural and economic impact in Brazil
- Comparison with Valentine's Day celebrations in other countries
- References to academic studies and expert opinions
Maintain a formal and respectful tone throughout, suitable for an academic or professional audience.
```

### **Optimistic/Celebratory**

```
Write an article about Brazilian Valentine's Day (Dia dos Namorados) in an optimistic and celebratory tone. Highlight the joy and romance of the day by focusing on:
- Heartwarming stories of love and celebration
- The communal spirit and shared happiness
- Uplifting quotes and positive anecdotes
- The significance of love and togetherness in Brazilian culture
Use emotive language to convey the festive and uplifting nature of the holiday.
```

It's worth noting that I only worked with ChatGPT to define the prompts that were used for all the AI models in this experiment. In the future, it will be interesting to see how the results differ when using prompts generated by different AI models.

### Scoring the Articles

Scoring the articles went relatively smoothly after making prior corrections. Here are my notes:

- **Claude** had no issues other than hitting the length limit, an increasing issue that has surfaced as of late. It, coincidentally, also rated itself highest on every single article.
- **GPT-4** experienced a few issues initially, but after adjustments, it did fine. GPT required a retest on two professional tone articles due to a tie. Unsurprisingly, it chose itself as the winner.
- **Gemini** was increasingly problematic. I gave up after going through each set of articles with GPT. I know I can get Gemini to work, but the effort required isn't worth the return.

### Reflections on AI Self Instruction

The experiment suggested that having AI models write prompts for themselves could lead to better results. However, further testing is warranted to validate this approach. The idea of "prompts for prompts" is amusing, but how far is too far? Nonetheless, the different AI models responded uniquely to the open-ended prompts for Brazilian Valentine's Day. Further tests could explore the AI's ability to self-instruct and adapt to various contexts.

## Key Insights

**Positive Observations:**

- The AI models showed potential for further refinement and optimization through self-instruction and providing instructions directly
- The different AI models responded uniquely to open-ended prompts, showcasing their individual strengths and approaches
- Having AI models write prompts for themselves emerged as a promising avenue for future exploration and optimization of AI-assisted content generation

**Encountered Challenges:**

- Most of the issues that have surfaced recently are problems I thought had been resolved in prior experiments. Any remaining challenges may be due to session variability, but the cause is uncertain.
- I have been underestimating the need to break down complex tasks into smaller, manageable steps. I'll need to set aside time to break things up to something more manageable for the AI.