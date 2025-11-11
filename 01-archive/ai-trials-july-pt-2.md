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