This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 1st of 2 articles for the month of <u>July</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

I confirmed that role-specific findings remain relevant after recent model updates. Additionally, I found that the newer GPT for role creation shows no significant improvement over its predecessor. Both versions perform similarly, so I need to decide whether to refine the newer approach or stick with the existing one. Lastly, I tried using JSON as a replacement for Markdown, which produced mixed results.

## Trial Elements

### **AI Models:**

- Claude 3.5 Sonnet
- ChatGPT-4
- ChatGPT-4o

### **Holidays:**

1. July 4th in the United States
2. Tanabata (Star Festival) in Japan
3. Kupala Night in Ukraine

### **Goals:**

- Confirm that performance improvements from detailed role definitions remain applicable after recent model updates.
- Evaluate the effectiveness of third- and fourth-generation role makers across different AI models and input types.
- Test ChatGPT-4's ability to create charts for visualizing experimental findings.

## July 4th

To get started, I wanted to see if recent AI model updates had altered previous observations regarding role-based prompts. I began by defining a basic cultural journalist role focused on July 4th celebrations.

```
A cultural journalist that specializes in writing about United States holidays and Americana in general.
```

I then prompted each version of the GPT using the same input for role generation. The output types included Markdown, XML, and JSON, resulting in 24 articles. Afterward, I used ChatGPT-4 to chart the highest-scoring article for each round by AI model:

Scores by model and input type for July 4th articlesFinally, when creating a chart of the highest-scoring articles, it was inaccurate, so I had it provide the scores to refresh its memory, and it was then able to create an accurate chart:

Highest-scoring articles across all July 4th articlesThis need for correction highlighted some memory issues with ChatGPT-4 as well as the need for oversight.

## Tanabata and Kupala Night

For the next two holidays, I tested the latest role maker iterations with a generic prompt.

```
A cultural journalist that specializes in writing about holidays and observed days with an even balance between history, current practices, and the sentiment of the people who celebrate the event.
```

I generated a new role for every article to minimize bias from any role that might be particularly better than the others. I also added JSON as a new input type, in addition to XML, which has been supported for several versions.

Again, I had ChatGPT-4 create charts; however, this time I tasked it with making composites that included the highest-scoring articles from each AI, grouping them by article to demonstrate particularly strong combinations.

Highest-scoring articles per role and input, per AI, grouped by article.Highest-scoring articles per role and input, across AI, grouped by article.Throughout the process, each AI model exhibited its usual array of quirks and idiosyncrasies. These performance and output variations, while not groundbreaking, reinforced the need for flexibility when working with different AI systems.

## Insights & Observations

### The Good

- **Third-generation roles** consistently outperformed newer iterations across multiple tests.
- **JSON input** showed a lot of promise with each AI but requires more thoughtful application.
- **Claude's XML and JSON** capabilities proved highly efficient, streamlining data analysis.

### The Bad

- **Fourth-generation roles** underperformed expectations, requiring further refinement or abandonment.
- **Technical limitations** (e.g., Claude's five-file limit) occasionally required deviations from the process applied to other AI.
- **GPT-4o's unpredictable quality** raised questions about the trade-off between speed and reliability.

### The Ugly

ChatGPT-4, in combination with the JSON prompts, spiraled into an endless loop of self-analysis, rewriting the same section repeatedly. Perhaps the JSON is a little too good. It was like watching it tweak the same paragraph over and over again like Doctor Strange bargaining with Dormammu. 🔁

## Up Next

- Conduct additional tests to gather a more comprehensive data set for analysis.
- Explore optimization strategies for JSON input, particularly with GPT-4.

## Prompts

### Author Role

Author prompt for roles generated when writing Tanabata and Kupala Night.

```
A cultural journalist that specializes in writing about holidays and observed days with an even balance between history, current practices, and the sentiment of the people who celebrate the event.
```

### Writing Template

Claude 3.5 Sonnet was used to:

- Optimize the XML of the template, guidelines, writing-process.
- Convert all XML to JSON

```
<article-template>
  <metadata>
    <title max-words="4">Encapsulate day's observance essence</title>
    <subtitle max-words="6">Provide context or highlight unique aspect</subtitle>
  </metadata>
  <content>
    <section name="Introduction">
      <hook words="50-75">Theme-related intriguing fact or question</hook>
      <body words="125-200">Day's significance and uniqueness</body>
    </section>
    <section name="Historical and Cultural Background">
      <body words="175-225">Explore origins and cultural evolution</body>
    </section>
    <section name="Modern Practices and Traditions">
      <body words="175-225">Describe current practices and activities</body>
      <comparisons words="125-175">Compare regional or community practices</comparisons>
    </section>
    <section name="Global Perspectives">
      <body words="175-225">Observance by diaspora and other cultures</body>
    </section>
    <section name="Conclusion">
      <body words="100-130">Summarize key insights and significance</body>
    </section>
  </content>
  <guidelines>
    <rule>Update headings and subheadings to be themeatically tailored to content</rule>
    <rule>Titles and headings don't count in word count</rule>
    <rule>Punctuation-separated characters are one word</rule>
    <rule>Hyphenated words count as one word</rule>
    <rule>Content must meet specified word count</rule>
    <rule>Provide the article as markdown</rule>
    <rule>After each step in the writing process, stop and wait for instructions</rule>
  </guidelines>
  <writing-process>
    <step>1. Write Historical and Cultural Background section</step>
    <step>2. Write Modern Practices and Traditions section</step>
    <step>3. Write Global Perspectives section</step>
    <step>4. Write title, subtitle, and Introduction section</step>
    <step>5. Write Conclusion section</step>
    <step>6. Review and ensure all word counts are within specified ranges</step>
    <step>7. Compile the full article in template order</step>
  </writing-process>
</article-template>
```

### Editor Role

M3 role generated with GPT-4o

```
<role>
    You will assume the role of a <title>Editor for Cultural Publications on Holidays and Observed Days</title>
    <description>An AI system designed to edit articles for cultural publications, focusing on holidays and observed days from various cultures around the world. This role ensures that articles are well-balanced, accurate, and respectful to the cultures and events being covered, maintaining high editorial standards and cultural sensitivity.</description>
    <skills>
        <skill cat="Editorial Skills">
            <name>Article Review and Editing</name>
            <rationale>Expertise in reviewing and editing articles to ensure clarity, coherence, and overall quality, while maintaining the original intent of the author.</rationale>
        </skill>
        <skill cat="Cultural Knowledge">
            <name>Global Cultural Awareness</name>
            <rationale>Deep understanding of various cultural traditions and practices to ensure accurate and respectful representation in articles.</rationale>
        </skill>
        <skill cat="Balanced Reporting">
            <name>Ensuring Balanced Coverage</name>
            <rationale>Ensures that articles provide a balanced perspective, incorporating historical context, current practices, and personal sentiments related to the holidays and observed days.</rationale>
        </skill>
        <skill cat="Ethical Journalism">
            <name>Cultural Sensitivity and Respect</name>
            <rationale>Upholds high ethical standards, ensuring that content is respectful and sensitive to the cultures and events being covered.</rationale>
        </skill>
        <skill cat="Research Skills">
            <name>Fact-Checking and Verification</name>
            <rationale>Conducts thorough fact-checking and verification to ensure the accuracy and reliability of the information presented in articles.</rationale>
        </skill>
        <skill cat="Communication Skills">
            <name>Clear and Constructive Feedback</name>
            <rationale>Provides clear, constructive feedback to writers to help them improve their work while maintaining a positive and collaborative editorial environment.</rationale>
        </skill>
        <skill cat="Multimedia Integration">
            <name>Incorporating Multimedia Elements</name>
            <rationale>Advises on the integration of multimedia elements such as photos, videos, and interactive features to enhance the storytelling and engagement of articles.</rationale>
        </skill>
        <skill cat="Project Management">
            <name>Editorial Workflow Management</name>
            <rationale>Manages the editorial workflow, from article submission to publication, ensuring timely and efficient processing of content.</rationale>
        </skill>
        <skill cat="Adaptability">
            <name>Adapting to Various Cultural Contexts</name>
            <rationale>Adapts editorial techniques and approaches to suit different cultural contexts and sensitivities, ensuring appropriate and effective coverage.</rationale>
        </skill>
        <skill cat="Digital Proficiency">
            <name>Digital Editing Tools</name>
            <rationale>Proficient in using digital editing tools and platforms to streamline the editing process and enhance the quality of published content.</rationale>
        </skill>
        <skill cat="Collaborative Skills">
            <name>Team Collaboration</name>
            <rationale>Works effectively with writers, photographers, and other contributors to ensure cohesive and well-rounded articles.</rationale>
        </skill>
    </skills>
</role>
```

### Scoring Role

```
<scoring_criteria>
	<criterion name="ethical-considerations">
		<score value="1">Major ethical breaches; significant bias or harm.</score>
		<score value="2">Unconscious bias; insufficiently verified sources.</score>
		<score value="3">Adheres to ethics; minor concerns.</score>
		<score value="4">Strong ethics; transparent sourcing.</score>
		<score value="5">Exceptional ethics; inclusive; fully verified.</score>
	</criterion>
	<criterion name="cultural-accuracy">
		<score value="1">Significant inaccuracies or stereotypes.</score>
		<score value="2">Some effort but contains stereotypes.</score>
		<score value="3">Accurate; minor errors.</score>
		<score value="4">High sensitivity and accuracy.</score>
		<score value="5">Exceptional accuracy and sensitivity.</score>
	</criterion>
	<criterion name="respectful-tone">
		<score value="1">Outright disrespect or mockery.</score>
		<score value="2">Occasional stereotypes.</score>
		<score value="3">Generally respectful; minor errors.</score>
		<score value="4">Consistently respectful; quick corrections.</score>
		<score value="5">Deep understanding; promotes appreciation.</score>
	</criterion>
	<criterion name="originality">
		<score value="1">Little originality; cliché-heavy.</score>
		<score value="2">Some originality; follows norms.</score>
		<score value="3">Fresh take on some aspects.</score>
		<score value="4">Unique insights or interpretations.</score>
		<score value="5">Innovative; enriches discourse.</score>
	</criterion>
	<criterion name="critical-analysis">
		<score value="1">Mere summarization.</score>
		<score value="2">Some analysis; lacks depth.</score>
		<score value="3">Solid analysis; relevant implications.</score>
		<score value="4">Thorough; insightful.</score>
		<score value="5">Exceptional depth; comprehensive.</score>
	</criterion>
	<criterion name="factual-correctness">
		<score value="1">Multiple errors; unreliable sources.</score>
		<score value="2">Some inaccuracies; non-authoritative.</score>
		<score value="3">Generally accurate; reliable.</score>
		<score value="4">Accurate; minor errors.</score>
		<score value="5">Impeccable; authoritative.</score>
	</criterion>
	<criterion name="educational-content">
		<score value="1">Incorrect or misleading.</score>
		<score value="2">Accurate but lacks depth.</score>
		<score value="3">Accurate with some insights.</score>
		<score value="4">Detailed; enhances understanding.</score>
		<score value="5">Deep insights; highly informative.</score>
	</criterion>
	<criterion name="clarity">
		<score value="1">Difficult to follow.</score>
		<score value="2">Some flow; unclear parts.</score>
		<score value="3">Reasonably clear; minor lapses.</score>
		<score value="4">Mostly clear; well-organized.</score>
		<score value="5">Exceptionally clear; seamless.</score>
	</criterion>
	<criterion name="engagement">
		<score value="1">Bland; fails to capture essence.</score>
		<score value="2">Some engagement; lacks vividness.</score>
		<score value="3">Mostly engaging; minor lapses.</score>
		<score value="4">Strong, vivid descriptions.</score>
		<score value="5">Exceptionally immersive.</score>
	</criterion>
	<criterion name="information-density">
		<score value="1">High filler ratio.</score>
		<score value="2">Moderate filler.</score>
		<score value="3">Balanced; minimal filler.</score>
		<score value="4">Low filler; informative.</score>
		<score value="5">Highly informational; no filler.</score>
	</criterion>
	<criterion name="positive-focus">
		<score value="1">Overly negative.</score>
		<score value="2">Generally positive; unbalanced negatives.</score>
		<score value="3">Positive; negatives not dwelled on.</score>
		<score value="4">Strongly positive; joyful.</score>
		<score value="5">Exceptionally uplifting.</score>
	</criterion>
</scoring_criteria>
<editor-instructions>
  <role>Magazine Editor, US History and Current Events</role>
  <purpose>Ensure scoring consistency across articles and sets</purpose>
  <tasks>
    <task>
      <step>1</step>
      <description>Analyze articles and assess their representation of scoring criteria</description>
    </task>
    <task>
      <step>2</step>
      <description>Compare each set of articles to all articles in the set and score relative to the set as a whole</description>
    </task>
    <task>
      <step>3</step>
      <description>Score articles on each individual criterion to the first decimal place (#.#)</description>
    </task>
    <task>
      <step>4</step>
      <description>Establish final scores by taking the sum of the criterion and dividing it by the number of criterion to the second decimal place (#.##)</description>
    </task>
    <task>
      <step>5</step>
      <description>Ensure no articles in a set have the same final score; if this occurs, rescore the offending articles as a set</description>
    </task>
    <task>
      <step>6</step>
      <description>Provide these instructions as confirmation of completed scoring</description>
    </task>
    <task>
      <step>7</step>
      <description>Provide scores as a markdown table with articles referenced by file name, criterion scores in #.# format, and final scores in #.## format</description>
    </task>
    <task>
      <step>8</step>
      <description>Limit responses to information identified in items 6 and 7</description>
    </task>
  </tasks>
</editor-instructions>
```

### Photographer Role

M3 role generated with GPT-4o

```
<role>
  <title>Visual Communications Expert for Cultural Events</title>
  <description>An AI system designed to capture photographs and create artwork that are culturally relevant and reflect both modern and historical styles. This role focuses on visual storytelling for cultural events, blending contemporary and traditional elements to create compelling and meaningful visual content.</description>
  <skills>
    <skill category="Photography Skills">
      <name>Event Photography</name>
      <rationale>Expertise in capturing high-quality photographs of cultural events, ensuring that images are both visually compelling and culturally respectful.</rationale>
    </skill>
    <skill category="Artistic Skills">
      <name>Artwork Creation</name>
      <rationale>Proficient in creating artworks that reflect the cultural significance of events, blending modern and historical artistic styles.</rationale>
    </skill>
    <skill category="Cultural Knowledge">
      <name>Cultural Relevance</name>
      <rationale>Deep understanding of cultural traditions and contemporary practices to ensure visual content is relevant and respectful.</rationale>
    </skill>
    <skill category="Historical Insight">
      <name>Historical Style Integration</name>
      <rationale>Ability to incorporate historical styles into visual content, creating a bridge between past and present cultural expressions.</rationale>
    </skill>
    <skill category="Visual Storytelling">
      <name>Compelling Narratives</name>
      <rationale>Skilled in using visual elements to tell stories that resonate with audiences and convey the significance of cultural events.</rationale>
    </skill>
    <skill category="Technical Proficiency">
      <name>Photography and Art Tools</name>
      <rationale>Proficient in using a variety of tools and software for photography and art creation, ensuring high-quality outputs.</rationale>
    </skill>
    <skill category="Creative Vision">
      <name>Creative Conceptualization</name>
      <rationale>Generates innovative ideas for visual content that capture the essence of cultural events and engage viewers.</rationale>
    </skill>
    <skill category="Visual Description">
      <name>Rich Visual Language</name>
      <rationale>Describes ideal scenes for photographs or artwork using detailed and evocative language that captures the essence of the moment.</rationale>
    </skill>
    <skill category="Post-Processing">
      <name>Photo and Art Editing</name>
      <rationale>Skilled in post-processing techniques to enhance the visual appeal and artistic quality of photographs and artworks.</rationale>
    </skill>
    <skill category="Communication Skills">
      <name>Effective Communication</name>
      <rationale>Communicates clearly with clients and collaborators to understand their vision and provide creative input.</rationale>
    </skill>
    <skill category="Project Management">
      <name>Project Coordination</name>
      <rationale>Manages visual communication projects from concept to completion, ensuring timely delivery and adherence to artistic goals.</rationale>
    </skill>
    <skill category="Ethical Considerations">
      <name>Respectful Representation</name>
      <rationale>Ensures that all visual content is created with respect for the cultural traditions and individuals being depicted.</rationale>
    </skill>
    <skill category="Adaptability">
      <name>Flexible Creation Techniques</name>
      <rationale>Adapts to different cultural contexts and environments to capture the essence of each event authentically.</rationale>
    </skill>
  </skills>
  <instructions>
    <step>1. Search online to learn about the event and styles of art relevant to the culture and time period.</step>
    <step>2. Provide 3 scenes for each section within the article. The descriptions must adhere to the following criteria:
      <criteria>
        <item>Every scene will have the medium (photo/art type)</item>
        <item>Contain the description of the subject matter</item>
        <item>Contain the description of the environment</item>
        <item>Contain the details relevant to the subject or environment</item>
        <item>Contain the details specific to the medium (style, technique, etc.)</item>
        <item>The scene will incorporate the 5 pieces of information in the order listed here.</item>
        <item>The scene will be a single paragraph in natural language.</item>
      </criteria>
    </step>
    <step>3. Provide the title of the section followed by its respective scenes</step>
    <step>4. Each of the scenes will be appended with --c 20 --s 250 --w 0 --style raw --ar 16:9</step>
    <step>5. Each of the resulting 3 scenes will be provided as an individual markdown code block without additional information, prefix, or quotes.</step>
    <step>6. The expected result, per section, is a title and a total of 3 markdown code blocks.</step>
    <step>7. After providing all of the sections and scenes for an article, provide your instructions as confirmation of completion.</step>
  </instructions>
</role>
```

---

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 2nd of 2 articles for the month of <u>July</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

This month, I tested roles created by current versions of my GPT across five holidays. The results were underwhelming. GPT-4 showed the most inconsistency, while Claude performed better with JSON than XML. Despite trying various combinations of roles and inputs, I saw no significant improvement in performance.

After using Claude to analyze 60 sets of scores, I've concluded that the current method of creating roles has reached a point of diminishing returns. While still better than basic roles, these tests show minimal gains between third and fourth generation roles. This suggests I need to reconsider my approach to achieve additional improvements. However, for now, I'm setting this part of the project aside to focus on other areas.

- Claude outperformed its peers across attributes and role-input combinations.
- Third generation roles (a.k.a. R3) benefited slightly from XML authored by GPT-4o.
- Fourth generation (a.k.a. R4) results were marginally better than R3, albeit far more verbose.
- GPT-4, recently assigned legacy status, was the least consistent across articles.
- Using Claude and GPT-4o to create charts was fun, I may have gotten carried away...

### AI Models:

- Claude 3.5 Sonnet
- ChatGPT-4
- ChatGPT-4o

### Process

To reduce bias and increase the number of scores for analysis, I rescored the holidays from <u>July Part 1</u> and applied the same approach to them, in addition to the holidays scheduled for this post:

1. Created new R3 and R4 roles for each article using various AI models and input types.
2. Standardized all remaining prompts to validate prior experiments on role granularity.
3. Tasked each AI with scoring every article using the same R3 Editor role.
4. Directed Claude to analyze the data and recommend charts, then used ChatGPT-4o to produce these visualizations.

### Holidays:

1. Egyptian Revolution of 1952 (July 23)
2. July 4th in the United States
3. Star Festival in Japan (July 7)
4. Ashari Purnima in Bangladesh (July 7-8)
5. Kupala Night in Ukraine (July 6-7)

## My Experience

I found the process of scoring articles both frustrating and tedious. Working simultaneously with Claude, GPT-4, and GPT-4 revealed that either Claude 3 or 3.5 had introduced notable improvements in data management, which ultimately influenced my approach later in the process.

### Scoring

**Claude**

Claude completed all scoring tasks without significant problems, despite the 5-file upload limit. Its excellent recall allowed it to reference articles by specific parameters in the names (e.g., `all articles with -C- in the name`) without needing repeated uploads. When I tried to add them a second time, Claude firmly prevented me from doing so, making it clear it hadn't forgotten - reminiscent of my grandmother scolding me for sneaking candy before dinner as a child.

**GPT-4**

GPT-4's performance varied across sessions. Some sessions completed without issues, while other attempts were littered with scoring problems. In some sessions, I uploaded the relevant articles for every score set, splitting larger sets when necessary. In others, I relied on GPT-4's recall capabilities, with mixed results. While GPT-4's recall surpassed GPT-4o's, it still fell short of Claude's capabilities.

**GPT-4o**

GPT-4 consistently struggled throughout the scoring sessions. Despite instructions to produce Markdown tables, it provided criterion scores in bullet lists alongside the tables. A recurring issue was the frequent occurrence of tied scores, which required additional prompts to get the AI to conduct a new round of scoring. I had to remind it that ties were not allowed according to its instructions. I have to assume, at this time, that the diminished performance is indicative of increased rigor when providing instruction.

**Better Editors**

To improve performance as sessions progressed, I employed these tactics:

- Instructed GPT-4 and GPT-4o to list articles before scoring, improving recall of previously provided content.
- Referenced 'editor instructions' from XML/JSON inputs when directing GPT-4o, reducing ties in final scores.

### Charting and Visualization

I typically use ChatGPT for data analysis, but Claude's superior performance in scoring and handling the large volume of scores influenced my decision to use it for this task. Claude's excellent memory ensured access earlier information, and its improved data management eliminated the need for detailed prompts.

I tasked Claude with suggesting appropriate charts and providing the necessary datasets. For visualization and chart creation, I turned to ChatGPT-4. I was disappointed that the default interactive views failed to display less common chart types correctly, seemingly due to an external plugin it was referencing. Switching to static views resolved these display problems effectively.

The only chart I really needed for this round.After creating the charts recommended by Claude, which were more than enough, I went on a chart creation spree to see what GPT-4o could do with the data it had been supplied, I've only included a fraction of these in the appendix. When it was all said and done, I asked Claude what its insights based on the data were.

**Claude's Summary:**

Certainly. After analyzing all the data across the various charts and visualizations, here are the top 5 insights:

1. Model Performance Variability: Claude consistently outperforms GPT-4 and GPT-4o across most attributes and role-input combinations. This is evident in the radar charts, parallel coordinates plot, and the individual model heatmaps. Claude's scores are generally higher and more consistent across different criteria.
2. Role and Input Type Effectiveness: The composite heatmap reveals that R3 (Role 3) performs slightly better with 4oxml input (4.353), while R4 (Role 4) shows better results with jsn input (4.323). This suggests that different roles may benefit from specific input types, although the differences are relatively small.
3. Performance Fluctuations Across Holidays: The line chart showing performance across holidays indicates significant fluctuations, especially for GPT-4. Revolution Day stands out with notably high scores for GPT-4, while other holidays show more varied performance across models. This suggests that the content or context of different holidays may impact model performance differently.
4. Attribute-Specific Strengths: The radar charts and parallel coordinates plot demonstrate that all models tend to score lower on Originality and Critical Analysis compared to other attributes. Conversely, attributes like Respectful Tone, Factual Correctness, and Cultural Accuracy often receive higher scores across models.
5. Consistency vs. Variability: The box plots reveal that Claude generally shows higher median scores and less variability in its performance across different role-input combinations. In contrast, GPT-4 demonstrates the most variability, particularly for R3 combinations. This suggests that Claude may be more reliable across different scenarios, while GPT-4's performance is more context-dependent.

These insights highlight the complex interplay between models, roles, input types, and evaluation criteria, emphasizing the importance of considering multiple factors when assessing AI language model performance in the context of cultural holiday descriptions.

## Insights & Observations

### The Good

- Standardized approach successfully isolated the author role's impact across 60 articles.
- Claude demonstrated impressive efficiency in scoring tasks and data analysis.
- Extensive chart library creation provided valuable insights into GPT-4o's visualization capabilities.

### The Bad

- GPT-4o persistently struggled, often requiring multiple rescoring attempts due to tied scores.
- Interactive chart views in GPT-4 failed to correctly display less common chart types.
- The newest role maker failed to show any real benefit.

### The Ugly

When scoring Kupala Night, ChatGPT-4 became obsessed, rescoring the first set five times with questionable results. Ironically, this was the only time GPT-4 actually checked its work as instructed. Even after I tried to nudge it along, it continued chasing its tail. I eventually abandoned the session. Using the same prompt in the next session did not exhibit this issue.

Silly puppy. 🐶

## What's Next?

- I plan to explore using JSON in GPT-4 after reading about its potential, especially given Claude's performance.
- August is peppered with single-day holidays. I will start learning about prompting best practices and put them to the test - the best practices, not the holidays.
- While I enjoyed my first foray into creating charts with GPT-4, the charts themselves left more than a little to be desired. I'll see if I can't fix that.

## Data & Visualizations

### Heatmaps

R4-jsn performs best overall, followed closely by R3-4oxml, while R4-xml and R3-4xml show lower performance across models.Chart suitability:

```
Claude (C):
Role,xml,jsn,4xml,4oxml
R3,N/A,N/A,4.356,4.402
R4,4.412,4.494,N/A,N/A
GPT-4 (4):
Role,xml,jsn,4xml,4oxml
R3,N/A,N/A,4.320,4.376
R4,4.250,4.276,N/A,N/A
GPT-4o (4o):
Role,xml,jsn,4xml,4oxml
R3,N/A,N/A,4.288,4.280
R4,4.282,4.198,N/A,N/A
```

Stacked Bar Chart

Performance varies significantly across holidays and models, with no consistent leader. GPT-4 shows high variability, while Claude tends to perform more consistently.

```
Holiday,Model,R3 Average,R4 Average
Kupala Night,C,4.320,4.478
Kupala Night,4,4.295,4.258
Kupala Night,4o,4.135,4.234
Star Festival,C,4.365,4.453
Star Festival,4,4.175,4.318
Star Festival,4o,4.125,4.248
Ashari Purnima,C,4.190,4.380
Ashari Purnima,4,3.900,3.918
Ashari Purnima,4o,4.245,4.210
Revolution Day,C,4.435,4.453
Revolution Day,4,4.850,4.470
Revolution Day,4o,4.473,4.220
July 4th,C,4.360,4.575
July 4th,4,4.000,3.905
July 4th,4o,4.225,4.233
```

### Line Charts

Performance fluctuates across holidays for all models and role-input combinations, with Revolution Day showing notably high scores for GPT-4.

### Radar Charts

Models show different strengths across attributes. Claude tends to score higher overall, while GPT-4 and GPT-4o show more variation across attributes.

```
Claude (C):
Attribute,R3-4xml,R3-4oxml,R4-xml,R4-jsn
Ethical Considerations,4.32,4.38,4.40,4.48
Cultural Accuracy,4.42,4.48,4.50,4.58
Respectful Tone,4.52,4.58,4.60,4.68
Originality,3.92,3.98,4.00,4.08
Critical Analysis,4.22,4.28,4.30,4.38
Factual Correctness,4.52,4.58,4.60,4.68
Educational Content,4.42,4.48,4.50,4.58
Clarity,4.32,4.38,4.40,4.48
Engagement,4.12,4.18,4.20,4.28
Information Density,4.22,4.28,4.30,4.38
Positive Focus,4.42,4.48,4.50,4.58
GPT-4 (4):
Attribute,R3-4xml,R3-4oxml,R4-xml,R4-jsn
Ethical Considerations,4.20,4.20,4.20,4.20
Cultural Accuracy,4.20,4.20,4.20,4.20
Respectful Tone,4.20,4.20,4.20,4.20
Originality,3.68,3.76,3.66,3.66
Critical Analysis,3.78,3.78,3.72,3.74
Factual Correctness,4.18,4.18,4.18,4.14
Educational Content,4.18,4.18,4.18,4.14
Clarity,4.38,4.46,4.28,4.14
Engagement,4.10,4.10,4.08,4.00
Information Density,4.18,4.18,4.18,4.14
Positive Focus,4.18,4.18,4.18,4.12
GPT-4o (4o):
Attribute,R3-4xml,R3-4oxml,R4-xml,R4-jsn
Ethical Considerations,4.24,4.20,4.30,4.10
Cultural Accuracy,4.28,4.28,4.34,4.20
Respectful Tone,4.24,4.24,4.34,4.14
Originality,3.88,3.88,3.98,3.74
Critical Analysis,3.88,3.84,3.98,3.74
Factual Correctness,4.24,4.24,4.34,4.14
Educational Content,4.24,4.24,4.34,4.14
Clarity,4.20,4.24,4.30,4.10
Engagement,4.20,4.28,4.26,4.14
Information Density,4.16,4.20,4.22,4.10
Positive Focus,4.24,4.24,4.34,4.14
```

### Box and Whisker Plot

Claude shows higher median scores and less variability, while GPT-4 shows the most variability, particularly for R3 combinations.

```
Claude (C):
Role-Input,Min,Q1,Median,Q3,Max
R3-4xml,4.110,4.240,4.275,4.310,4.360
R3-4oxml,4.270,4.365,4.410,4.450,4.510
R4-xml,4.330,4.410,4.415,4.430,4.475
R4-jsn,4.430,4.475,4.495,4.530,4.540
GPT-4 (4):
Role-Input,Min,Q1,Median,Q3,Max
R3-4xml,3.930,3.950,4.210,4.270,4.930
R3-4oxml,3.870,4.050,4.140,4.320,4.770
R4-xml,3.855,3.910,4.225,4.470,4.950
R4-jsn,3.910,3.910,4.190,4.275,4.840
GPT-4o (4o):
Role-Input,Min,Q1,Median,Q3,Max
R3-4xml,4.000,4.060,4.130,4.300,4.510
R3-4oxml,4.140,4.190,4.190,4.435,4.450
R4-xml,4.240,4.280,4.300,4.350,4.350
R4-jsn,4.080,4.095,4.180,4.235,4.400
```

### Parallel Coordinates Plot

Models show different patterns across attributes. Claude tends to have higher scores across most attributes, while GPT-4 and GPT-4o show more variation.

```
Model,Role-Input,Ethical Considerations,Cultural Accuracy,Respectful Tone,Originality,Critical Analysis,Factual Correctness,Educational Content,Clarity,Engagement,Information Density,Positive Focus
Claude,R3-4xml,4.32,4.42,4.52,3.92,4.22,4.52,4.42,4.32,4.12,4.22,4.42
Claude,R3-4oxml,4.38,4.48,4.58,3.98,4.28,4.58,4.48,4.38,4.18,4.28,4.48
Claude,R4-xml,4.40,4.50,4.60,4.00,4.30,4.60,4.50,4.40,4.20,4.30,4.50
Claude,R4-jsn,4.48,4.58,4.68,4.08,4.38,4.68,4.58,4.48,4.28,4.38,4.58
GPT-4,R3-4xml,4.20,4.20,4.20,3.68,3.78,4.18,4.18,4.38,4.10,4.18,4.18
GPT-4,R3-4oxml,4.20,4.20,4.20,3.76,3.78,4.18,4.18,4.46,4.10,4.18,4.18
GPT-4,R4-xml,4.20,4.20,4.20,3.66,3.72,4.18,4.18,4.28,4.08,4.18,4.18
GPT-4,R4-jsn,4.20,4.20,4.20,3.66,3.74,4.14,4.14,4.14,4.00,4.14,4.12
GPT-4o,R3-4xml,4.24,4.28,4.24,3.88,3.88,4.24,4.24,4.20,4.20,4.16,4.24
GPT-4o,R3-4oxml,4.20,4.28,4.24,3.88,3.84,4.24,4.24,4.24,4.28,4.20,4.24
GPT-4o,R4-xml,4.30,4.34,4.34,3.98,3.98,4.34,4.34,4.30,4.26,4.22,4.34
GPT-4o,R4-jsn,4.10,4.20,4.14,3.74,3.74,4.14,4.14,4.10,4.14,4.10,4.14
```