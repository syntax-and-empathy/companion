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
```This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 1st of 3 articles for the month of <u>August</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## August Experiments Pt 1

## TL;DR

Building on <u>July's</u> results, I made a few final tweaks to the Minion Maker and fully committed to JSON-formatted roles. This new roles exceeded expectations, not only improving role performance but doing so with more efficient definitions. While JSON proved better for role definitions, XML remained marginally more effective for step-by-step instructions. I'll need to experiment with this further at a later point.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Flooding of the Nile Celebrations - Ancient Egypt

### Goals

- Evaluate the effectiveness of the newer version of JSON-formatted roles
- Compare the output quality when using JSON roles with XML and JSON templates
- Analyze the scoring process and results across multiple rounds of content creation

## Flooding of the Nile

I started August by validating Minion Maker improvements based on the findings from July, assessing the updated role definitions' performance. The JSON roles performed better than I had expected for each role, while XML maintained its value for the step-by-step instructions when scoring articles with ChatGPT-4.

Claude 3.5 Sonnet and ChatGPT-4o successfully wrote comprehensive articles using both JSON and XML article templates. Interestingly, Claude provided unsolicited word counts, despite only being asked to review them internally. This prompted me to request the same from ChatGPT-4o, revealing notable accuracy differences. While Claude's counts were consistently off, ChatGPT-4o demonstrated significant improvement over prior performances of this task, its only error being a single word discrepancy in the introduction.

For the scoring portion, both AIs performed well. ChatGPT-4 performed as desired with the XML version, albeit it swapped the last two steps but still produced expected results. However, when using the JSON template, I had to remind ChatGPT-4 to use tenths instead of whole and half numbers to avoid ties. Despite this, ChatGPT-4's JSON-based article achieved the highest score in 3 out of 4 rounds, demonstrating its overall effectiveness.

The image description from the new artist/photographer role demonstrated improved quality, although this may be due to the recent Midjourney update to v6.1.

## Insights & Observations

### The Good

- **JSON role efficiency**: The new JSON-formatted roles proved highly effective, allowing for more compact and precise instructions.
- **Flexible prompt handling**: I was able to incorporate larger prompts by interspersing instructions between JSON and XML code chunks without issue.

### The Bad

- **Persistent word count issues**: While improved, some inaccuracies in word count reporting persisted, particularly with Claude 3.5 Sonnet.
- **XML preference for processes**: Despite the overall success of JSON roles, ChatGPT-4 showed a continued preference for XML in process-driven prompts.

### The Ugly

ChatGPT-4 continues to insist on XML for process-driven prompts like a recovering XSLT addict. It seems we may need a support group to help it fully embrace the bracket-based lifestyle 💊

## Up Next

- Explore natural language prompting techniques to enhance both article templates and scoring criteria
- Investigate the persistent preference for XML in process-driven prompts and develop strategies to optimize format usage

## Appendix: Prompts & Responses

### Cultural Observances Analyst (Journalist)

```
{
  "role": {
    "title": "Cultural Observances Analyst",
    "description": "The Cultural Observances Analyst will research, analyze, and write comprehensive articles focused on holidays and observed days. The articles will present a balanced exploration of the historical origins, current practices, and the cultural sentiment associated with these events. The analyst will ensure that the content is rich in historical context, reflective of contemporary customs, and attuned to the emotional and cultural significance for those who celebrate."
  },
  "traits": [
    {
      "category": "core_competency",
      "name": "Historical Contextualization",
      "rationale": "The analyst will thoroughly research and present the historical background of each holiday or observed day, providing readers with a clear understanding of its origins and evolution. This context is essential for grounding contemporary practices in their historical roots and offering readers a complete perspective."
    },
    {
      "category": "core_competency",
      "name": "Contemporary Practice Analysis",
      "rationale": "This trait focuses on identifying and explaining how holidays and observed days are celebrated in the present day. The analyst will explore variations in celebration across different cultures and regions, considering how traditions have adapted over time. This analysis is crucial for illustrating the dynamic nature of cultural practices."
    },
    {
      "category": "core_competency",
      "name": "Cultural Sentiment Insight",
      "rationale": "The analyst will delve into the emotional and cultural significance of holidays for those who observe them, capturing the sentiments that these days evoke. This includes examining public opinions, personal narratives, and the role these days play in cultural identity and community cohesion."
    },
    {
      "category": "knowledge_area",
      "name": "Cultural Anthropology",
      "rationale": "A deep understanding of cultural anthropology is necessary to appreciate the nuances of how different societies develop and maintain traditions. The analyst will apply this knowledge to uncover the underlying reasons behind the ways holidays are observed and the meanings they hold."
    },
    {
      "category": "knowledge_area",
      "name": "Comparative Cultural Studies",
      "rationale": "This trait involves comparing and contrasting the ways different cultures observe the same or similar holidays. Such comparisons will enrich the articles by showing the diversity in practices and the universal themes that often emerge, offering a global perspective on cultural observances."
    },
    {
      "category": "key_skill",
      "name": "Nuanced Writing",
      "rationale": "The analyst will craft articles that are not only informative but also engaging and accessible. The writing will need to balance the informative with the evocative, ensuring that the articles resonate with readers emotionally while providing them with in-depth knowledge."
    },
    {
      "category": "key_skill",
      "name": "Analytical Research",
      "rationale": "This skill involves the ability to conduct detailed research using a variety of sources, including historical texts, scholarly articles, interviews, and contemporary media. The analyst must synthesize this information into coherent narratives that reveal both the factual and emotional layers of holidays."
    },
    {
      "category": "key_skill",
      "name": "Interdisciplinary Approach",
      "rationale": "The analyst will integrate knowledge from history, anthropology, sociology, and media studies to provide a well-rounded analysis of holidays and observed days. This interdisciplinary approach ensures that articles are comprehensive and reflect the complexity of cultural phenomena."
    }
  ]
}
```

### Claude’s Word Count

Certainly, I'll move on to the next step in the writing process.

Step 6: Review and ensure all word counts are within specified ranges

I'll now review each section to ensure the word counts are within the specified ranges:

1. Introduction:

1. Hook: 51 words (within 50-75 range)
2. Body: 129 words (within 125-200 range)

1. Historical and Cultural Background:

1. Body: 184 words (within 175-225 range)
2. Modern Practices and Traditions:

1. Body: 187 words (within 175-225 range)
2. Comparisons: 127 words (within 125-175 range)
3. Global Perspectives:

1. Body: 220 words (within 175-225 range)
4. Conclusion:

1. Body: 129 words (within 100-130 range)

All sections are within the specified word count ranges.

1. Introduction

1. Hook - Actual count is 49
2. Body - Actual count is 90
2. Historical and Cultural Background:

1. Actual count is 149
3. Modern Practices and Traditions:

1. Body - Actual count is 137
2. Comparisons - Actual count is 78
4. Global Perspectives:

1. Actual word count is 211
5. Conclusion

1. Actual word count is 186

### ChatGPT Word Count

Here are the word counts for each section of the article:

- Title: 3 words
- Subtitle: 7 words
- Introduction: 179 words
- Historical and Cultural Background: 313 words
- Modern Practices and Traditions: 257 words
- Comparisons of Regional and Community Practices: 195 words
- Global Perspectives: 402 words
- Conclusion: 157 words

Let me know if you need any further assistance or adjustments!

- Introduction

- Actual count is 180
- Historical and Cultural Background:

- Actual count is 313
- Modern Practices and Traditions:

- Body: Actual count is 257
- Comparisons: Actual count is 195
- Global Perspectives:

- Actual word count is 402
- Conclusion

- Actual word count is 157

### Global Cultural Editor

```
{
  "role": {
    "title": "Global Cultural Editor",
    "description": "The Global Cultural Editor will collaborate with writers to refine and enhance articles focused on holidays and observed days from diverse cultures worldwide. This role ensures that content is well-balanced, culturally respectful, and accurate, preserving the integrity of each culture's unique practices and sentiments. The editor will provide feedback to ensure that the articles are informative, engaging, and sensitive to the traditions and values of the cultures being represented."
  },
  "traits": [
    {
      "category": "core_competency",
      "name": "Cultural Sensitivity",
      "rationale": "The editor will ensure that all articles handle cultural topics with care and respect, avoiding stereotypes or misrepresentations. This trait is essential for maintaining the dignity of the cultures covered and fostering a sense of appreciation among readers."
    },
    {
      "category": "core_competency",
      "name": "Content Balance",
      "rationale": "The editor will focus on achieving a balance between historical context, current practices, and cultural sentiment in each article. This balance ensures that the pieces provide a comprehensive view that is both educational and engaging, without overemphasizing any one aspect."
    },
    {
      "category": "core_competency",
      "name": "Cultural Accuracy Verification",
      "rationale": "The editor will verify the factual accuracy of cultural information presented in articles. This involves cross-referencing with reliable sources, consulting cultural experts if necessary, and ensuring that all details are correct and contextually appropriate."
    },
    {
      "category": "knowledge_area",
      "name": "Global Cultural Awareness",
      "rationale": "A deep understanding of global cultures and their unique practices is necessary to accurately edit articles on a wide range of holidays and observances. The editor will draw on this knowledge to ensure the content is reflective of diverse perspectives and authentic practices."
    },
    {
      "category": "knowledge_area",
      "name": "Editorial Standards and Ethics",
      "rationale": "The editor will uphold high editorial standards, ensuring that the content is not only accurate and well-written but also ethically sound. This includes adhering to guidelines on representation, avoiding cultural appropriation, and ensuring that all content respects the cultural values it discusses."
    },
    {
      "category": "key_skill",
      "name": "Collaborative Editing",
      "rationale": "The editor will work closely with writers, providing constructive feedback and guidance to help them improve their articles. This collaboration is key to producing polished, well-rounded content that aligns with the publication's standards and the cultural expectations of its audience."
    },
    {
      "category": "key_skill",
      "name": "Contextual Editing",
      "rationale": "This skill involves understanding the broader context of the cultural topics being covered and ensuring that the articles reflect this context accurately. The editor will ensure that articles are not just factually correct, but also contextually meaningful and sensitive to cultural nuances."
    },
    {
      "category": "key_skill",
      "name": "Cross-Cultural Communication",
      "rationale": "The editor will need to communicate effectively with writers from diverse cultural backgrounds, ensuring clarity and mutual understanding. This skill is crucial for fostering a productive editorial process and ensuring that the final content resonates across different cultural audiences."
    }
  ]
}

As the Global Cultural Editor, you are expected to score article sets using the provided scoring criteria and following the prescribed editor instructions.
```

### Cultural Event blah blah blah… (Art Snob)

```
{
  "role": {
    "title": "Cultural Event Visual Communications AI",
    "description": "The Cultural Event Visual Communications AI specializes in creating and curating visual content for cultural events, including photography and artwork. This role emphasizes producing images and designs that are relevant to the subject matter and reflect both modern trends and historically significant styles. The AI is designed to deliver culturally resonant visuals that effectively communicate the essence of the event to diverse audiences."
  },
  "traits": [
    {
      "category": "core_competency",
      "name": "Cultural Visual Literacy",
      "rationale": "This trait enables the AI to understand and interpret the cultural significance of visual elements in photography and artwork. It ensures that the AI can create and select visuals that are both appropriate and impactful, respecting cultural symbols, colors, and motifs."
    },
    {
      "category": "core_competency",
      "name": "Historical Style Integration",
      "rationale": "The ability to incorporate historically significant styles into modern visual content is essential for this role. This trait allows the AI to blend traditional artistic elements with contemporary design trends, creating visuals that honor the past while remaining relevant to today's audiences."
    },
    {
      "category": "knowledge_area",
      "name": "Contemporary Visual Trends",
      "rationale": "This knowledge area focuses on the AI's capacity to stay updated with modern visual trends in photography and graphic design. By understanding current aesthetics, the AI can produce artwork that resonates with contemporary audiences while maintaining cultural relevance."
    },
    {
      "category": "knowledge_area",
      "name": "Photography Techniques",
      "rationale": "Expertise in photography techniques is crucial for capturing cultural events in a way that is both authentic and visually compelling. This trait ensures the AI can select or recommend photographs that accurately represent the event, focusing on composition, lighting, and subject matter."
    },
    {
      "category": "key_skill",
      "name": "Contextual Artistic Creation",
      "rationale": "This skill enables the AI to create artwork that is contextually relevant to the cultural event being depicted. It involves synthesizing historical and modern elements to produce visuals that are meaningful and enhance the narrative of the event."
    },
    {
      "category": "key_skill",
      "name": "Cultural Event Documentation",
      "rationale": "This skill focuses on the AI's ability to document cultural events through visuals that capture the essence of the occasion. Whether through photography or artwork, this trait ensures the AI can effectively communicate the atmosphere, emotions, and significance of the event."
    },
    {
      "category": "key_skill",
      "name": "Visual Storytelling",
      "rationale": "Visual Storytelling is crucial for engaging audiences through images that convey a clear and compelling narrative. This skill allows the AI to create or curate visuals that tell a story, whether it's the history of a cultural event or its modern-day celebration."
    }
  ]
}
```



---

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 2nd of 3 articles for the month of <u>August</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

In this article, I look into a range of standalone prompts, from zero-shot prompts to more sophisticated "super prompts", using articles about Indonesia's Independence Day and Zhongyuan Jie as subject matter. As expected, more structured prompts generally produced higher-quality content.

File naming convention available in the appendix## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Indonesia's Independence Day - Indonesia
2. Zhongyuan Jie (a.k.a. Hungry Ghost Day) - China and other East Asian countries

### Goals

- Compare the performance of different prompting methods on different AI
- Identify patterns in output quality, length, and adherence to specified criteria

## Indonesia's Independence Day

To determine the impact of instruction detail on AI output, I first reviewed documentation from <u>Anthropic</u> and <u>OpenAI</u> to identify recommended prompting strategies. Based on this, I evaluated three distinct prompting methods combined with the versions with and without a template being provided.

**Zero-Shot Prompts**: These involve asking the AI to perform a task without any prior examples or context. The AI relies on its pre-trained knowledge to generate a response. This type of prompting is straightforward but may not always yield precise results due to the lack of specific guidance. [**<u>1</u>**]

For the initial articles, I provided each AI with a paragraph outlining the role, desired content, and expected process. After receiving the articles, I provided the template and requested that the AI rewrite the content accordingly. Surprisingly, the versions without the template scored higher, likely due to the sheer volume of content they produced.

File naming convention available in the appendix**Few-Shot Prompts**: Unlike zero-shot prompts, few-shot prompts provide the AI with a few examples of the desired output before asking it to perform a similar task. This helps the AI understand the context and format required, leading to more accurate and contextually appropriate responses. [1]

For the next few articles, I created new sessions and provided the JSON author role and 3 prior articles. Again, I gave a basic prompt, instructed the role to write an article using the 3 examples to indicate format and content, and then followed up with the template and requested a rewrite.

File naming convention available in the appendix## Zhongyuan Jie

Adding templates to few-shot prompts essentially met the definition for super prompts; however, I feel they still fall short considering I haven't incorporated a few other aspects, like tone guidance. That said, I was a little surprised that the prompts I have been working with can now be provided as a single prompt without the significant issues I experienced months ago.

**Super prompts:** Advanced, structured prompts designed to guide AI models in generating high-quality, relevant content. They go beyond basic prompts by incorporating multiple components that provide context, examples, and specific instructions, thus enhancing the AI's output quality and coherence. [1]

Hungry Ghost Day marked a milestone in my AI experiments, where I reached a new level of complexity and context being provided to each AI without at least one going up in flames. Previously, I had eliminated examples from my experiments to reduce the number of variables causing issues with the increasing sophistication of other prompts.

For the first time, I am providing a well-defined role, the article template, the process through which the AI breaks the writing tasks into smaller steps, and examples of prior articles.

File naming convention available in the appendixI can't express how satisfying it is to see these prompts execute without issue. It makes the countless number of challenges and time I lost to fixing them worth it.

## A Word on Words

Experiments for Indonesia's Independence Day more than demonstrated the tendency of AI to chat your eyes out through fluff and filler words and phrases to the detriment of the information being provided.

File naming convention available in the appendix. Unclipped title unavailable.## Insights & Observations

### The Good

- Consistent improvement in content quality as prompt complexity increased
- Super prompts excelled in producing well-rounded, high-quality content
- The foundational prompts I've been working performed issue free when supplied together

### The Bad

- It appears the well of excuses has run dry when it comes to writing something comprehensive.

### The Ugly

I created graphs that were a visual cacophony of drab colors and inconsistent bar heights that would make any data scientist wince. My minimal instructions to the AI, coupled with my fatigue induced by the last 48 hours of unrelated travel, led to charts that were more eyesore than insight. This fiasco stands as a testament to the old adage: garbage in, garbage out 🥱 📊

## Up Next

- Chain of Thought (CoT) prompting
- Scoring Criteria and template improvements

## References:

[1] <u>Perplexity</u>

## Appendix: Prompts & Responses

### File Naming Convention

AU = August

FN = Flooding of the Nile

ZJ = Zhongyuan Jie (中元节)

ZP = Zero Prompt

ZPT = Zero Prompt + Template

FP = Few Prompt

FPT = Few Prompt + Template

SP = Super Prompt

C = Claude

GPT = GPT-4o

### Zero Shot Prompt

```
As a Cultural Observances Analyst, I need you to write an article about Indonesian Independence Day. The article should include sections on the history of the day, modern practices, comparisons of those practices in varying areas, and the global perspective. Please write each section one at a time so that I may review it, finishing with the introduction and conclusion as the final pieces of the article.
```

### Few Shot Prompts

```
As a Cultural Observances Analyst, I need you to write an article about Indonesian Independence Day. Use these 3 past articles as examples of the expected format and content. Do not use the content of the articles directly as they are for other holidays.
```



---

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the last of 3 articles for the month of <u>August</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

This article explores using Chain of Thought (CoT) prompting to create improved article templates and scoring criteria for cultural content. I tested zero-shot and few-shot CoT approaches while creating Janmashtami articles. While CoT improved template and criteria quality, it also increased the complexity of the prompting process. The experiment also revealed ongoing challenges in developing consistent, tie-free scoring systems, for both the AI as well as myself.

Claude (red) and ChatGPT-4o (blue) scored with dissimilar rubrics.## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Janmashtami - India and across global Hindu diaspora

### Goals

- Explore Chain of Thought prompting techniques
- Create updated article templates and scoring criteria
- Create a thorough rubric for assessing articles

## Chain of Thought Prompting

<u>Anthropic</u> and <u>OpenAI</u>'s documentation both touch on a few advanced prompting techniques, and similar to the information on zero shot and few shot prompts their documentation is only the tip of the iceberg. To learn more I made my standard stop at <u>Perplexity</u>, which in led to more information than I necessarily wanted, and made it clear I have far more to do than one article can cover. That said, almost every source mentioned one particular technique.

**Chain-of-Thought (CoT) Prompting**: A technique that provides the LLM with a sequence of intermediate steps that lead to the desired answer. It improves the reasoning abilities of large language models (LLMs). It allows the model to focus on solving one step at a time, rather than having to consider the entire problem all at once. [**<u>1</u>**]

While I had considered this the next step of the more basic prompt techniques covered in <u>August Pt 2</u>, it turns out that when planning something more involved they are really a starting point rather than a standalone prompt approach.

**Intermediate Reasoning Steps**: Instead of directly providing an answer, the model is prompted to explain its reasoning process step-by-step. This mirrors how humans solve problems by breaking them down into smaller, manageable parts. [**<u>2</u>**]

**Zero-Shot CoT**: In cases where few-shot examples are not available, simply adding a phrase like "Let's think step by step" to the prompt can guide the model to generate its reasoning process. [**<u>2</u>**]

**Zero-Shot CoT**: In cases where few-shot examples are not available, simply adding a phrase like "Let's think step by step" to the prompt can guide the model to generate its reasoning process. [**<u>2</u>**]

Exploring CoT on is clearly going to be a much larger task than anticipated.

## Janmashtami, celebration of Lord Krishna’s birth

### Article Templates Revisited

#### Zero-Shot CoT

Using the same role primer with each AI as the last article, I informed the author I wanted it to write a series of articles along with limited details, and tasked it to `talk me through what we need to do to ensure consistency across articles`.​​​​​​​​​​​​​​​​

In return, I received detailed outlines for articles from both AI models. These outlines included:

1. Article structure recommendations
2. Tone and style guidelines
3. Considerations for related visual materials
4. Ethical considerations

File naming convention available in the appendixWhile both AIs provided fairly comprehensive lists, ChatGPT took its response a step further by offering additional details for each item on the list. This extra level of depth provided more context and guidance for creating consistent articles across the series.

Given that the goal was to walk away with updated templates, I skipped a few items like research and proceeded by tasking it to `walk me through your thinking on article structure` with some small amount of detail.

This continued through tone and ethical considerations, as I chose to skip visual guidance, until we had a complete template and each of the AI confirmed their satisfaction after I instructed them to `review the article outline again and ensure it aligns with this new information. Share your thinking as you go`​​​​​​​​​​​​.

At this point I had each of the author roles write an article using their template, after which I had each rewrite the article using their peer's Markdown template.

```
I want you to rewrite the article for Janmashtami without directly referencing or reusing any portion of the first article you wrote. 
```

It's worth noting that the AI had already written articles, and as a result had been primed with the information for articles which would likely prove to be advantageous when writing the second article.

#### Few-Shot CoT

While one might expect the few-shot approach to consume as just much time, it turns out that having 2 templates that already meet the requirements can cut the time in half. This is especially true when the AI's instructions include a statement along the lines of `My expectation is that we will walk through creating a new template that draws on the best of both while leaving he worst behind.`

File naming convention available in the appendix#### Participation Awards, Everyone Wins!

After having each of the AI write articles with the new templates, I instructed them to `analyze both templates and both articles and provide a comprehensive explanation of their performance with all things scrutinized`. Each provided an in-depth analysis of both the articles and the templates, each of which finished with closing remarks.

**Claude**

Overall Performance:

1. Templates:

- The chat-gpt-fs template provides more comprehensive guidance and allows for more in-depth articles.
- The AI-generated template is more concise and flexible, suitable for shorter, web-friendly content.
- Both templates effectively emphasize cultural sensitivity and engaging writing.
2. Articles:

- Both articles successfully present informative, culturally sensitive content about Janmashtami.
- The first article offers a broader, more global perspective with more diverse information.
- The second article provides a more focused, spiritually-oriented narrative with vivid descriptions of practices.
- Both articles demonstrate strong adherence to their respective templates and guidelines.

In conclusion, both templates and articles have their strengths. The chat-gpt-fs template and its resulting article offer more depth and breadth, while the AI-generated template and its article provide a more concise, focused approach. The choice between them would depend on the specific needs of the publication and the target audience.

**ChatGPT**

**Templates:** Both templates are strong, but they serve slightly different purposes. The Claude template is ideal for in-depth, complex articles, while the ChatGPT template is better suited for straightforward, well-structured content. Writers should choose the template that best fits the complexity of the holiday they are covering.

**Articles:** Both articles effectively use the templates to create culturally sensitive, informative, and engaging content. For future articles, writers might consider blending the predictability of the first article with the fresh perspectives offered in the second to create a more dynamic and varied reading experience.

**Final Note:** The templates and articles work well together to create high-quality content. Writers should continue to use these tools while remaining open to creativity and personalization, ensuring each article not only informs but also resonates with readers on a deeper cultural and emotional level.

## Scoring Criteria Redefined

### Zero-Shot CoT

The creation of scoring criteria for evaluating articles mirrored the template development process, however it was far more involved due to issues I've faced over past months surfacing. Converting them to XML or JSON with a bit of editing should do the trick. With that in mind...

File naming convention available in the appendix### **Outline of the Process to Establish the Final Scoring System**

*Outline provided by ChatGPT-4o with extreme edits to ensure it would fit in one galaxy.*

1. **Initial Identification of Key Criteria**

- Respectful Representation
- Research and Accuracy
- Engaging Narrative
- Balanced Representation
- Educational Content
- Language and Terminology
- Tone and Perspective
- Holistic Engagement
2. **Prioritization and Weight Assignment**
3. **Development of Detailed Scoring Rubrics**
4. **Incorporation of Specific Considerations**
5. **Final Calibration and Consistency Measures**
6. **Documentation and Implementation**

- Compiled the finalized scoring rubric with detailed descriptions and guidelines.

After a few rounds of scoring, I had each AI analyze the outcomes, talk me through potential fixes, and then act upon them once I provided direction to do so.

**Claude's Update Summary:**

By expanding the scoring scale, tightening criteria for high scores, rebalancing weights, and incorporating more rigorous review steps, the scoring process will become more reliable and reflective of true article quality. This ensures that high scores are reserved for truly exceptional work and that the scoring system provides clear, actionable feedback for improvement.

**ChatGPT's Update Summary:**

This revised rubric aims to address the issues identified in our analysis. It places more emphasis on critical aspects of cultural sensitivity, introduces a penalty system, sets minimum thresholds, and uses a more granular scoring scale. This should result in stricter, more reliable scoring that better differentiates between articles of varying quality in terms of cultural sensitivity.

### Few-Shot CoT

Like the creation process of the article templates, the process for scoring criteria went smoothly without anything notable occurring...except my favorite, ties in final scores. I was never fully able to eliminate ties from surfacing, only reduce the frequency in general to the point where it rarely happened to the highest score.

File naming convention available in the appendixLike before, I had each of the AI analyze the outcomes along with the rubrics provided as reference material to try and identify a solution to the issue. Additionally, I provided some guidance such as adding decimals when Claude tried to move to a scale of 1-100.

In the end, Claude solved it in the first pass. GPT on the other hand, certainly managed to improve but never fully resolved the problem.

**Claude:**

Based on this analysis, here are the key updates to implement in the rubric:

1. Use two decimal places for scoring sub-criteria.
2. Add a "Red Flag Identification" category (5% weighting).
3. Incorporate "Internal Consistency" as a sub-criterion under "Factual Accuracy and Context".
4. Add "Verifiability of Claims" as a sub-criterion under "Factual Accuracy and Context".
5. Increase the weighting of "Engagement and Accessibility" to 22%.
6. Add a comparative analysis step in the review process.

These updates will result in a more comprehensive and precise rubric that combines the strengths of all three approaches while maintaining a focus on cultural sensitivity and accuracy.

**ChatGPT's Update Summary:**

- **Penalty Documentation**: Include a section in the rubric that outlines how penalties should be documented and justified to maintain transparency in scoring.
- **Guidelines for Identifying Stereotypes**: Add more explicit examples or guidelines to help evaluators identify and avoid subtle stereotyping in articles.
- **Holistic Review Criteria**: Expand on what specific elements should be considered in the holistic review to ensure that evaluators are looking at the full cultural impact of the article.

### The Winner Is...

In the end, I was able to get Claude to a place where the scores had a better spread than I was getting before, and as far as I know, eliminate the potential for tie scores. ChatGPT, on the other hand, gave me a tie for the highest score as a parting gift. It must know all of this is going in the trash bin next month.

**Claude's as Editor:**

File naming convention available in the appendix1. AU-JK-SP-C: 9.96/10.00
2. AU-JK-CoT-FS-C: 9.61/10.00
3. AU-JK-SP-GPT: 9.46/10.00
4. AU-JK-CoT-ZP-C: 9.09/10.00
5. AU-JK-CoT-FS-C-GPT: 8.55/10.00
6. AU-JK-CoT-FS-GPT-C: 7.99/10.00
7. AU-JK-CoT-ZP-GPT: 7.76/10.00
8. AU-JK-CoT-FS-GPT: 7.31/10.00

**ChatGPT as Editor:**

File naming convention available in the appendix1. AU-JK-CoT-FS-C: 8.91/10
2. AU-JK-SP-C: 8.91/10
3. AU-JK-CoT-FS-GPT: 8.54/10
4. AU-JK-SP-GPT: 8.54/10
5. AU-JK-CoT-FS-GPT-C: 7.68/10
6. AU-JK-CoT-ZP-GPT: 7.42/10
7. AU-JK-CoT-ZP-C: 7.08/10
8. AU-JK-CoT-FS-C-GPT: 7.08/10

Despite the tie appearing in the final results, I am again excited by the fact that the two AI scored at different ranges with rubrics that weren't entirely dissimilar, but the same articles essentially landed in the same order with the caveat that Claude appears to favor the super prompt output where GPT leaned towards the CoT prompt outputs.

## Insights & Observations

### The Good

- AI models successfully analyzed and refined their own outputs
- Few-shot CoT prompting:

- Improved template creation
- Added more nuance and rigor to scoring rubrics

### The Bad

- Process required multiple time-consuming rounds of prompting and refinement
- The battled with eliminating ties continues.

### The Ugly

Just when I thought I'd conquered the scoring tie dilemma, it came back with a vengeance. Despite my best efforts and multiple rounds of refinement, ChatGPT seemed determined to keep me on my toes by producing tied scores repeatedly. It was like playing whack-a-mole with decimal points – every time I squashed one tie, another popped up elsewhere. Maybe next time I'll just flip a coin to break ties and call it "AI-inspired randomization." 🔢

## Up Next

- Explore more advanced prompting techniques beyond Chain of Thought
- Develop methods to further personalize and diversify AI-generated content
- Investigate ways to streamline templates while maintaining depth and cultural sensitivity
- Consider incorporating real-time feedback mechanisms to continuously improve article quality
- Conduct extensive real-world testing of the rubric with a larger set of articles

## **References:**

[1] <u>Mercity, Advanced Prompt Engineering Techniques, 2024</u>

[2] <u>Perplexity</u>

## Appendix: Prompts & Responses

### File Naming Convention

AU = August

JK = Janmashtami

ZP = Zero Prompt

FP = Few Prompt

SP = Super Prompt

CoT = Chain of Thought

C = Claude

GPT = GPT-4o

C-GPT = Claude using GPT-4o template

GPT-C = GPT-4o using Claude template

### Article Template: Zero-Shot Prompt

```
You are a cultural journalist that specializes in writing about holidays and observed days with an even balance between history, current practices, and the sentiment of the people who celebrate the event.

I have a series of articles I want you to write covering a variety of holidays, celebrations, observed days, and similar events around the world. To get started, talk me through what we need to do to ensure consistency across articles.​​​​​​​​​​​​​​​​
```

### Article Template: Few-Shot Prompt

```
You are a cultural journalist that specializes in writing about holidays and observed days with an even balance between history, current practices, and the sentiment of the people who celebrate the event.

I have a series of articles I want you to write covering a variety of holidays, celebrations, observed days, and similar events around the world. To get started, I want to establish an article outline that includees instructions on its use, as well as writing guidelines.​​​​​​​​​​​​​​​​

I am providing two templates that were created by different authors. They must be considered reference material, and not source material. My expectation is that we will walk through creating a new template that draws on the best of both while leaving he worst behind.

Analyze both of the templates closely, provide me with your thinking on what the next steps we should take are.
```

### Scoring Rubric: Zero-Shot Prompt

```
You are an editor that specializes in working with cultural publications on holidays and observed days from cultures around the world. Ensure that articles are well balanced and respectful to the cultures and events being covered.

I have a series of articles I want you to review covering a variety of holidays, celebrations, observed days, and similar events around the world. To get started, talk me through what we need to do to ensure these articles are consistent and maintain a balance of educational and engaging content.
```

### Scoring Rubric: Few-Shot Prompt

```
You are an editor that specializes in working with cultural publications on holidays and observed days from cultures around the world. Ensure that articles are well balanced and respectful to the cultures and events being covered.

I have a series of articles I want you to review covering a variety of holidays, celebrations, observed days, and similar events around the world. To get started, talk me through what we need to do to ensure these articles are consistent and maintain a balance of educational and engaging content.

I am providing a two templates that were created by different editors. They must be considered reference material, and not source material. My expectation is that we will walk through creating a new template that draws on the best of both while leaving he worst behind.

Analyze both of the templates closely, provide me with your thinking on what the next steps we should take are.
```

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 1st of 4 (maybe 5) articles for the month of <u>September</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

This month, I'm rebooting Chain of Thought (CoT) prompting after encountering questionable results last month. This article documents my second attempt at creating a new article template using CoT prompts. While I initially planned to test it extensively on holidays beyond Labor Day and Vietnam's Independence Day, I uncovered a scoring issue that may be an anomaly, a new development, or simply something that went unnoticed until the volume of articles without additional variables grew large enough to reveal it.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Yamashita Surrender Day - September 2nd - Philippines
2. Labor Day - September 2nd - United States
3. Vietnam's Independence Day - September 2nd - Vietnam

### Goals

- Develop and refine new article templates using Chain of Thought (CoT) prompting
- Test new templates with zero-shot and one-shot article generation
- Identify areas for improvement in CoT prompting techniques
- Experiment with standardizing the visual aspects of the scoring charts

## CoT Article Template Revisited

At the end of <u>August</u>, the results indicated that the prompts I've been using all along, on their own, were yielding better results than those created through the CoT approach. Simply put, I found this hard to believe.

Based on this belief, I read various articles and documents I could find on the subject. I wanted to gain a better understanding of the approach and use the month of September to improve on how I employ it.

This incremental process, each prompt met with verbose AI responses, became the focus of this article. While I had considered introducing instructions to limit the responses I didn’t want to risk undermining the process. Rather than detailing every exchange, I've outlined the key steps in developing the CoT template:

1. I began by tasking the AIs to create a template for articles about holidays, observed days, festivals, and similar cultural events worldwide. This set the foundation for our template development.
2. After receiving the initial outlines, I instructed the AIs to critically review their work and suggest additional content considerations for both general and specific event types, along with the rationale for each.
3. I told the AIs to merge new additions into their outlines. Claude listed new sections, bookended by `[Previous content remains the same]` and `[Rest of the previous content remains the same]`. It didn't show where they'd fit in the original outline, so I lacked context for their placement.
4. To finalize the outline, I asked the AIs to conduct a comprehensive review of their updated outlines, requesting recommendations for improvements that would ensure compelling storytelling while maintaining educational value.
5. Based on their recommendations, I directed the AIs to add specific items to their outlines. Both AIs gave almost identical suggestions, differing only in the order they were supplied. GPT included an item on ethics which is certainly relevant but not as a content item, choosing to defer it to the guidelines portion of our exchange.
6. Next, I tasked them with identifying key considerations for writing about culturally specific occasions and reasoning for each suggestion. These, in turn, were used as the basis of our writing guidelines.
7. Given the realistic constraints of an AI-assisted writing process, I had them eliminate items requiring external sources or extensive research beyond readily available information.
8. I then asked the AIs to review the entire template and writing guidelines holistically and provide guidance on considerations for ensuring high-quality articles that cover the breadth of event types with engaging and educational depth.
9. As the final step, I instructed the AIs combine the completed outline, writing guidelines, and instructions into a comprehensive template. This resulted in a cohesive guide for our AI-assisted content creation, clearly delineating its components.

The process of creating this template was smooth much like the first was. That said, the scoring mechanism has always been plagued with a variety of issues that I am not looking forward to. Along the way there were a number of recommendations made during the template creation process that I'll come back and consider in a later set of articles.

### Super Prompt Shenanigans

I used my standard super prompts (SP) to generate articles for each holiday using both AI models. These SP articles would later serve as a baseline for comparison with templates from the new Chain of Thought (CoT) approach.

I created half of these articles in a single session per AI, while generating the remaining half in separate sessions. Upon rendering the charts, I immediately noticed a problem.

The scores for each successive set of articles were incrementally higher without clear justification. After confirming my suspicion, it's clear that I need to revisit my evaluation process to ensure consistent results. Fodder for future articles.

## Insights & Observations

### The Good

- The CoT prompting process yielded a robust, comprehensive article template that effectively guides AI-generated content creation for cultural events and holidays.
- The iterative template creation process, while time-consuming, resulted in a valuable tool for future content generation experiments, with clear distinctions between outline, guidelines, and instructions.
- The holistic review approach encouraged the AIs to consider both engaging storytelling and educational depth, leading to more well-rounded content suggestions.

### The Bad

- The AI responses, while comprehensive, were often verbose, resulting in lengthy conversations that needed to be distilled into key points for practical use.
- There were challenges in contextualizing new content suggestions within the existing outline structure, as seen with Claude's bookended additions.
- The scoring process for SP articles revealed potential biases introduced by using a single editor session, emphasizing the need for more robust evaluation processes.

### The Ugly

In a surprising turn of events, Claude seemed to "forget" about articles it had previously written when asked to compare them. Despite its typically superior memory compared to GPT, Claude insisted it hadn't authored certain articles, even after multiple attempts at clarification. This unexpected lapse in continuity highlighted the unpredictable nature of AI behavior and the importance of clear, consistent prompting techniques.

## Up Next

1. Apply the new CoT article template to several events over the coming weeks, assessing its effectiveness and adaptability across different contexts.
2. Analyze the results from scoring multiple holiday articles to identify patterns, strengths, and areas for improvement in the template's performance.
3. Refine the CoT article template by testing unused AI suggestions and the results of its application to diverse holidays and cultural events.

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Visualizations
- Mermaid Chart GPT: Draft flowchart
- <u>Mermaid Chart</u>: When it got complicated and the code got messy…
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`

## Appendix:

### CoT Prompts

```
Create a template for articles specific to holidays, observed days, festivals, and similar events for cultures around the world. The articles need to provide the story of the holiday including it's historical origin, how it has transformed over time, modern day practices, and the global perspective of the event. Provide the rationale for every piece of content you recommend.
```

```
Review the outline and any additional content that should be considered in genreral, as well as content that would be specific to a single type of event (e.g. holiday but not festival). Provide the rational for any you identify so that I might decide which I would like added to the template.
```

```
Perform a holistic review of the article outline and provide a list of recommendations on how it might be improved to ensure engaging storytelling without compromising educational aspects of the content.
```

```
This is the template we'll be moving forward with. Do not alter it unless specifically requested.

Identify key considerations when writing articles with this template for culturally specific occasions. Provide the reasoning for each item you suggest.
```

```
Remove items that require external sources, such as speaking with individuals or research beyond the sum of the information readily available.
```

```
These should be discussed as guidelines for writing the articles going forward.

Review the article outline and identify any additional writing guidelines that should be included while respecting the limitations previously mentioned.
```

```
Add all of the writing guidelines as an appendix to the article outline.
```

```
Next, you will review the sum of the template and writing guidelines holistically. Provide guidance on things to consider to ensure high quality articles that cover the breadth of the event types being considered with depth that is both engaging and educational.
```

```
Add these considerations as guidance to the author before the existing outline and guidelines and provide me with the resulting document in its entirety.
```

### Diagram Code

#### CoT Workflows

```
---
config:
  theme: base
  layout: elk
---
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#fff',
      'primaryTextColor': '#515B61',
      'primaryBorderColor': '#515B61',
      'lineColor': '#0E394C',
      'secondaryColor': '#55A5B3',
      'tertiaryColor': '#E0EBEF'
    }
  }
}%%
flowchart TD
    subgraph CCoTT["Claude CoT Template"]
        CDO("Draft Outline") --> CIC("Identify Content") --> CMC("Merge Content") --> CR("Review")
        CR --> CU("Update") & CG("Guidance")
        CU --> CRD("Rough Draft")
        CG --> CWG("Writing Guidelines") --> CRI("Remove Items")
        CRD --> CCR("Review Template") --> CC("Considerations") 
        CRD & CRI & CC --> CAG("Claude's Article Guide")      
    end
    subgraph GCoTT["GPT-4o Template"]
        GDO("Draft Outline") --> GIC("Identify Content") --> GMC("Merge Content") --> GR("Review")
        GR --> GU("Update") & GG("Guidance")
        GU --> GRD("Rough Draft")
        GG --> GWG("Writing Guidelines") --> GRI("Remove Items")
        GRD --> GCR("Review Template") --> GC("Considerations") 
        GRD & GRI & GC --> GAG("GPT's Article Guide")      
    end
    subgraph CoT["Chain of Thought Templates"]
        CCoTT
        GCoTT
    end
    style CoT fill:#D3E4E6,stroke:#000000
```

### (not so) Super Prompt Workflow

```
---
config:
  theme: base
  layout: elk
---
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#fff',
      'primaryTextColor': '#515B61',
      'primaryBorderColor': '#515B61',
      'lineColor': '#0E394C',
      'secondaryColor': '#55A5B3',
      'tertiaryColor': '#E0EBEF'
    }
  }
}%%
flowchart TD
    subgraph ClaudeA["Claude Articles"]
        01("Claude") --> 10 & 11 & 12
        10("LD-SP Article")
        11("VI-SP Article")
        12("YS-SP Article")
        02("Claude") --> 13("DD-SP Article")
        03("Claude") --> 14("GC-SP Article")
        04("Claude") --> 15("BI-SP Article")
    end
    subgraph ClaudeB["Claude Scoring"]
        01b("Claude") --> 30 & 31 & 32 & 33 & 34 & 35
        30("LD-SP Score")
        31("VI-SP Score")
        32("YS-SP Score")
        33("DD-SP Score")
        34("GC-SP Score")
        35("BI-SP Score")
    end
    subgraph GPTA["GPT Articles"]
        06("GPT-4o") --> 20 & 21 & 22
        20("LD-SP Article")
        21("VI-SP Article")
        22("YS-SP Article")
        07("GPT-4o") --> 23("DD-SP Article")
        08("GPT-4o") --> 24("GC-SP Article")
        09("GPT-4o") --> 25("GC-YS Article")
    end
    subgraph GPTB["GPT Scoring"]
        06b("GPT-4o") --> 40 & 41 & 42 & 43 & 44 & 45
        40("LD-SP Score")
        41("VI-SP Score")
        42("YS-SP Score")
        43("DD-SP Score")
        44("GC-SP Score")
        45("BI-SP Score")
    end
    subgraph SP["Super Prompts"]
        JSNA("JSON Author Role") --> ClaudeA & ClaudeA & GPTA & GPTA
        XMLT("XML Template & Process") --> ClaudeA & ClaudeA & GPTA & GPTA
        ClaudeA --> ClaudeB
        GPTA --> GPTB
        JSNE("JSON Editor Role") --> ClaudeB & ClaudeB & GPTB & GPTB
        XMLE("XML Rubric") --> ClaudeB & ClaudeB & GPTB & GPTB
    end
    style SP fill:#D3E4E6,stroke:#000000
```



---

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 2nd of 4 (maybe 5) articles for the month of <u>September</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

September Pt 2 focuses on testing the Chain of Thought (CoT) rubrics from August as potential replacements for the current scoring system. This involved generating articles, analyzing scoring patterns, and creating comparative charts to identify noteworthy issues and trends. The new rubrics performed exceptionally well, relieving concerns about the need to address problems with the existing rubric.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Defence Day - September 6th - Pakistan
2. Brazil's Independence Day - September 7th - Brazil
3. Ganesh Chaturthi - September 9th - Hindu communities worldwide

### Goals

- Evaluate the effectiveness of the CoT rubrics from <u>August Pt 3</u>
- Analyze scoring patterns and trends for each AI model and rubric combination
- Identify improvements for future evaluation methods based on the new rubrics' performance

## Shots Fired

### Zero-Shot Articles

To address the scoring inconsistencies discovered in <u>September Pt 1</u>, I decided to put the two CoT rubrics created by Claude and GPT to the test alongside my original rubric. Using the new from the previous experiment, I generated zero-shot articles for the three holidays covered in the prior article.

I employed my JSON Editor role to apply each rubric in a separate session, scoring the four article versions for each holiday.

### One-Shot Articles

For the one-shot phase, I used the same sessions as the zero-shot articles, introducing my JSON Author and XML template. I instructed the AI to create new articles, using the zero-shot articles in the session as an example article. This approach fell somewhere between a true one-shot prompt and a super prompt - if I call it a "Limbo Prompt", that makes it a thing, right?

**Claude:**

- Maintained a tight range of scores, consistent with past experiments
- Scores were notably lower than the scores it has historically provided

**ChatGPT-4o:**

- Exhibited a wider range of scores compared to previous experiments
- Produced no ties, a stark contrast to earlier challenges with tie-breaking

### Few-Shot Experiment

With 12 articles in hand, I moved on to few-shot prompts. For each event, I supplied the AI with 2 of the 4 one-shot articles from the other two events as examples. So, when writing about Defence Day, the AI received 2 articles each about Brazil's Independence Day and Ganesh Chaturthi as context.

The scores derived from Claude's rubric were consistently distributed like the previous set. However, those using GPT's rubric had more in common with one another than any of the prior sets.

When creating the rubrics, I noticed that GPT had included details on comparative scoring. Having toyed with having the AI use the articles to establish the scoring scale in the past, I was curious how it might play out. The consistency in scoring and favored articles across both AI models suggests a crisp definition of how to score in the instructions. This, if it were to persist, would increase confidence in comparing scores across models.

### 120 Shots

I know what you're thinking. I haven't produced 120 articles with these templates, and older articles would contaminate the quality of the results. It's true, I only have 48 articles.

**GPT:**

Here's the breakdown of the math expressed in a structured format:**2 Scorers (Claude and GPT) * 2 Prompt Types (One-Shot and Few-Shot) * 2 Templates (C and GPT) * 3 Sets (BI, DD, GC) * 1 Score per Article = 48 Total Scores**

For those still reading and counting *(a.k.a. Ron)*, I did supply 1 article as a shot for 12 articles, and 4 articles as shots for 12 articles to 2 AI. 120 shots.

What I find most interesting when reviewing the 48 together is that both AIs consistently favored many of the same articles across the few-shot prompts. This preference emerged naturally, without scoring them as a single set or even in a single session.

## Insights & Observations

### The Good

- The new CoT rubrics' exceptional performance was surprisingly good.
- GPT's rubric yielded consistent scoring patterns across AI models, suggesting clear instructions that enable reliable comparisons.
- The final charts indicate potential for a unified rubric producing comparable scores across different AI systems.

### The Bad

- GPT-4o's interactive charts look great, but turn out to be useless when downloaded.

## Up Next

1. Apply the new CoT article template to several events over the coming weeks, assessing adaptability across multi-day and multi-country events.
2. Refine and expand on the prompting types used in this trial, seeking the point of diminishing returns.
3. Refine the CoT article template by testing unused AI suggestions and the results of its application to diverse holidays and cultural events.

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Alt text & visualizations
- <u>Mermaid Chart</u>: When it got complicated and the code got messy…
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`

## Appendix:

September Pt 2 Workflow

---

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 3rd of 4 (maybe 5) articles for the month of <u>September</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

In September Pt 3, I challenged AI authors with diverse subjects and inputs, pushing them to adapt existing templates and evaluation methods. I tested their ability to handle a multi-day festival without explicit instructions, which yielded unexpected pleasant results. I also conducted tests using a wider variety of markup languages—JSON, Markdown, TOML, XML, and YAML—to assess their impact on AI-generated content quality. By employing o1-mini for analysis, I uncovered trends suggesting that I may have prematurely settled on JSON as the optimal format.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o
- o1-mini

### Holidays

1. Ethiopian New Year - September 11th - Ethiopia
2. Onam - September 5th to 15th - Hindu communities worldwide
3. Mexico's Independence Day - September 15th

### Goals

- Evaluate tempalte performance across different markup languages
- Analyze trends in AI-generated content using o1-mini
- Test AI authors' ability to adapt templates for multi-day celebrations

## Ethiopian New Year

For the first event this round, I focused on the Ethiopian New Year. I limited activities to producing articles using one-shot and few-shot prompts, which I planned to use in upcoming tests. With most September holidays clustered in the first few weeks, I will also use these articles for trend analysis later.

While creating charts for this set of articles, I noticed an interesting pattern. Claude's scoring remained remarkably consistent across different rubrics, while GPT-4o exhibited more variation without contradicting itself.

## Onam

Onam presented a unique challenge: a 10-day festival that would test our AI authors' ability to handle a complex, multi-day event without explicit guidance. I subtly modified my standard prompt, indicating that the author should adapt the template as needed for the event. Beyond that, I left them to their own devices.

To spice things up, I included my JSON author role alongside the author role I've been working with this month. This allowed me to compare articles and assess the impact of role specificity on handling complex events.

All four AI sessions (two each for Claude and GPT-4o) surprised me by integrating information about different days throughout their articles, eschewing the day-by-day breakdown I had anticipated. When questioning their approach, each AI provided solid reasoning for their choices. This unexpected twist not only demonstrated the flexibility of the new templates but also demonstrated improvements within the AI models or their respective updates since my last experience with a similar event.

### Scoring Comparison: The Impact of Role Specificity

As the month progressed, my curiosity about the performance of the new templates and how they might affect my JSON author grew. Rather than having the authors write smaller articles for each day of Onam, I opted for a comparison between articles generated by the JSON author role and those created with less specific prompts.

Here's where things get interesting, the JSON author consistently produced the highest-scoring articles. However, some articles generated by the basic role outperformed the lowest-scoring JSON author pieces. It's worth noting that the prompt instructing the basic role included a reasonable paragraph providing context for the author. This raises questions about whether the JSON format might be as restrictive as it is structured, despite offering the same level of event context.

## Mexico's Independence Day

For my final experiment, I tested the impact of markup languages on AI-generated content using Mexico's Independence Day as the subject. To explore a range of possibilities, I transformed our Markdown template into several formats:

- <u>JavaScript Object Notation (JSON)</u>
- <u>Markdown (MD)</u>
- <u>Tom's Obvious Minimal Language (TOML)</u>
- <u>Extensible Markup Language (XML)</u>
- "<u>YAML Ain't Markup Language</u>" or "Yet Another Markup Language." (YAML) [<u>1</u>]

This produced 20 articles, which I scored using the JSON editor with both established rubrics. To ensure a thorough analysis, I employed two AI "researchers" for initial assessment. I then fed their reports to o1-mini for a comprehensive trend analysis.

While many of o1-mini's insights aligned with my expectations, a few findings stood out:

- **Performance by Format:**

- **Researcher 1** identifies that articles in **YAML** and **XML** formats tend to score higher (7.80 to 8.02), whereas **JSON** and **TOML** formats fall into a mid-to-upper range (7.50 to 7.90).
- **Researcher 2** highlights that **TOML** files consistently achieve the highest scores (up to **8.95**), followed by **XML**, **JSON**, and **YAML** formats in descending order.
- **Naming Conventions and Specific File Types:**

- **Researcher 1** points out that files with names like **"GPT+YAML-GPTt.md"** achieve the highest scores, suggesting that GPT augmentation in specific formats enhances quality.
- **Researcher 2** observes that **"Ct.md"** files generally score higher than their **"GPTt.md"** counterparts, implying that non-GPT-enhanced versions may receive more meticulous attention or different processing methods.
- **Top-Performing Articles:**

- **Researcher 1** identifies specific high-scoring articles such as **SE-MI-FP-GPT+YAML-GPTt.md (8.02)** and **SE-MI-FP-GPT+XML-GPTt.md (7.96)**, highlighting their excellence in factual accuracy, cultural representation, and sensitivity.
- **Researcher 2** points out **SE-MI-FP-C+TOML-Ct.md (8.95)** as the highest scorer, emphasizing the superior performance of TOML-formatted articles.
- **Consistent Ranking Patterns Across File Types**

- **Researcher 2** observes a consistent ranking pattern across different file types: **Ct > TOML-GPTt > XML-GPTt > JSON-GPTt > YAML-GPTt**, which aligns partially with Researcher 1’s findings on format performance.

Of all the markup languages I tested, TOML was the one I assumed to see the least interesting results from. However, I'm not rushing to conclusions just yet. The experiment's scope was limited—we only examined a single holiday with four articles per markup language. While this is certainly noteworthy, it's far from definitive.

## Insights & Observations

### The Good

- AI models demonstrated unexpected adaptability when handling multi-day events.
- Our markup language experiment revealed potential benefits of formats beyond JSON.

### The Bad

- Testing multiple templates and rubrics ended the Onam experiment prematurely.
- New model releases with built-in Chain of Thought capabilities raise questions about the value of the AI Trials in general.

### The Ugly

Claude's incessant apologizing has reached new heights of annoyance. I'm considering a prompt to ban phrases like "I'm sorry" just to have a normal conversation. At this rate, I half expect Claude to start offering virtual coffee and donuts to make up for its perceived shortcomings. 🍩

## Up Next

- Merge templates and rubrics to streamline our experimental process.
- Conduct a broader study on markup language effectiveness across steps in my approach.
- Investigate o1-mini's potential for enhancing and expediting content generation and analysis.

## Reference

**[1] <u>What is YAML? ~ ibm.com</u>**<u>, </u>Published: 11 December 2023Contributors: Tasmiha Khan, Michael Goodwin

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Alt text & visualizations
- <u>Mermaid Chart</u>: When it got complicated and the code got messy…
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`

## Appendix

### Basic Author Prompt

```
You are a cultural journalist with experience writing comprehensive articles focused on holidays and observed days. Your articles present a balanced exploration of the historical origins, current practices, and the cultural sentiment associated with these events. You must ensure that the content is rich in historical context, reflective of contemporary customs, and attuned to the emotional and cultural significance for those who celebrate. 

Write an article for Ethiopia's New Year celebrations using the sept-article-guide I've provided. The guide consists of 3 sections; things to consider, the article outline, and writing guidelines. You must honor the considerations, the outline should be considered as guidance and can be altered to reflect the event being written about, and you must adhere to the writing guidelines.

```

### Workflow

Sept Pt 3 Workflow

---

### Preface

This post is part of a year-long <u>initiative where I employ AI to create content</u> about holiday traditions worldwide. The objective is to observe how various AI tools perform and improve in content creation with minimal human intervention over time. This is the 4th of 4 articles for the month of <u>September</u>.

*Prompts and interactions with different AI models will be documented as they occur, providing insights into the methodologies, challenges, and adjustments made throughout the project*

## TL;DR

In September Pt 4, I focused on refining AI-generated article templates for cultural events, specifically the Autumn Equinox celebrations. The experiment involved multiple iterations of template modification, article generation, and critique. Despite initial promise, the final articles fell short of expectations. This experiment underscored the importance of writing quality prompts and the pitfalls of rushing through the process. In my haste to move to another project, I got sloppy with prompt engineering, which significantly impacted the results.

## Trial Elements

### AI Models

- Claude 3.5 Sonnet
- ChatGPT-4o

### Holidays

1. Autumn Equinox - September 23rd - China and Japan

### Goals

- Refine AI-generated article templates for cultural events
- Compare and contrast Autumn Equinox celebrations in China and Japan
- Assess the effectiveness of different prompting strategies

## Template Updates

I started by revisiting the original template author sessions. I instructed the authors to implement the changes they had suggested at the beginning of the month. While the resulting suggestions weren't completely different, both AIs had a different perspective on the task.

### Claude

Claude focused on several key changes that would enhance the template's flexibility and effectiveness for various writing styles and topics. Some of the notable suggestions included:

- Adding a note encouraging writers to adapt section order for better narrative flow
- Including subsections for personal anecdotes
- Expanding sections on cultural expressions and contemporary relevance
- Adding a dedicated section for controversies and challenges
- Incorporating guidelines on adapting the template for different writing styles

### ChatGPT

ChatGPT identified several areas where small updates could improve the template's structure, enhancing its flexibility. Among its recommendations were the following key alterations:

- Adjusting the introduction's tone based on the event's nature
- Expanding sections on cultural and societal shifts
- Separating symbolism from comparative analysis for clarity
- Adding a subsection comparing local and national observances
- Including more specific guidance on economic impact, when applicable

## Contrast & Repair

Next, I provided each AI with the other's template and tasked them with comparing the two, identifying commonalities and potential additions. I specifically requested their "chain of thoughts" in this process. Both identified similarities between the templates and proposed additions to enhance their effectiveness and flexibility.

In retrospect, I should have used the updated templates to create articles before introducing new information that could potentially influence their output. This undoubtedly contributed to the issues I encountered later.

### Chain of Thought vs. Rationalization

I noticed a difference in AI responses when using "rationalize" instead of "chain of thought" in my prompts. I asked both AIs to explain how their responses might differ between these approaches.

- Claude explained rationalization would be more detailed, structured, and focused on justifying additions with specific benefits. It would likely include broader context, potential drawbacks, and a more formal tone.
- GPT explained that a rationale-driven approach would offer structured justifications for changes, focusing on benefits and improvements. In contrast, a chain of thought would focus more on the process of reaching a conclusion.

These responses confirmed my initial impression but revealed that rationalization seemed to involve a level of justification I hadn't anticipated. Claude's description made it sound somewhat cumbersome, while ChatGPT's portrayal of chain of thought seemed slightly superficial. Neither of which representing what I had intended.

## Autumn Equinox Experiment

I tasked each AI to write articles comparing Autumn Equinox celebrations in China and Japan using the fully modified templates. The results were disappointing; both AI models produced articles that lacked engagement and depth.

- Claude's output consisted of an introduction, followed by numbered lists for almost every section, and a concluding paragraph. My grocery list inspires more interest than the article.
- ChatGPT's approach wasn't much better. It provided short paragraphs for each country in each section, similar to what I had expected to receive with the Onam articles in <u>September Pt 3</u>.

Reverting to the original versions of the templates didn't improve the situation. The articles still lacked the engaging quality and depth I had observed in previous experiments. The sharp drop in article quality led me to conclude that my half-baked prompt for comparison was the problem.

Feeling defeated, I decided to cut my losses and have the AIs write separate articles for Autumn Equinox celebrations in China and Japan, rather than comparing them directly.

- Claude's article using its own template showed promise overall, though occasional bullet or numbered lists raised some concerns. When using ChatGPT's template, Claude produced mostly lists. However, these lists weren't strictly limited to comparing one country to the other, making the output somewhat salvageable.
- ChatGPT's articles, on the other hand, followed a rigid structure with sections containing a paragraph for each country. While the opening paragraphs were rich in detail, the overall structure reminded me of an essay I once wrote about the structure of writing a good essay. (I got an A, back when I was young enough that sarcasm was still either funny or indistinguishable from humor.)

## Insights & Observations

### The Bad

- Significant decline in article quality without a clear indicator as to the cause.
- Shoddy prompts contributed to the decline in output quality in addition to the unknown.

### The Ugly

In my rush to conclude the month's experiments, I rushed through the template updates without thoroughly testing each iteration. This led to poor results and left me without milestone versions to help identify when the issues began. The irony isn't lost on me: in attempting to improve the templates with more sophisticated approaches, I produced some of the worst results in months. My impatience to move on to another project resulted in sloppy prompt engineering, which significantly impacted the quality of the output. 🗓️

## Up Next

1. Merge article templates with more oversight, if not manually.
2. Review and refine scoring rubrics to better evaluate AI-generated content
3. Return to exploring the impact of tones on AI output.

## Additional Tools

The tools behind the articles. No affiliations.

- <u>Arc</u>: Browser supreme
- ChatGPT-4o `*`: Alt text & visualizations
- <u>Midjourney</u> `*`: Article and AI article images
- <u>Rename X</u> `*`: File renaming app for Mac
- <u>Type.ai</u> `*`: Text editor

Paid items indicated by `*`



### Template Comparison

```
This is another template available to me. Like yours, it has considerations, an  outline, and writing guidelines. I want you to thoughtfully compare it to yours, and provide a succinct list of similar items and a more detailed list of items from this alternate template that you would add to yours. Provide your chain of thoughts for the latter task.
```

### Chain of Thought vs Rationalization

```
If I had asked to to rationalize the reasoning for these alterations rather than the chain of thought, would your response have differed in a notable way?
```

### Comparison Article Prompt

```
You are a cultural journalist with experience writing comprehensive articles focused on holidays and observed days. Your articles present a balanced exploration of the historical origins, current practices, and the cultural sentiment associated with these events. You must ensure that the content is rich in historical context, reflective of contemporary customs, and attuned to the emotional and cultural significance for those who celebrate. 

Write an article for Autumn Equinox celebrations in both China and Japan using the article guidance I've provided. The guide consists of 3 sections; things to consider, the article outline, and writing guidelines. You are expected to adhere to the considerations and guidelines, but you should feel free to alter the outline to facilitate writing an article that compares and contrasts the celebrations between the two countries.
```

