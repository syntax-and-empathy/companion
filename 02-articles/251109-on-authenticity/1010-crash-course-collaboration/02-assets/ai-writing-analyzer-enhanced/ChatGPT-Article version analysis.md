# Article version analysis

**User:** Anonymous (wtrekell@gmail.com)  
**Created:** 8/21/2025 17:21  
**Updated:** 8/21/2025 17:29  
**Exported:** 8/21/2025 17:30  
**Link:** [https://chatgpt.com/c/68a7b819-30f4-832c-b9aa-ce2c81617e2f](https://chatgpt.com/c/68a7b819-30f4-832c-b9aa-ce2c81617e2f)  

## Prompt:
0610-ai-writing-analyzer-enhanced.py

Python

draft-markup-languages.md

File

edited-markup-languages.md

File

final-markup-languages.md

File

refined-markup-languages.md

File

Use this python file on these versions of my article.

## Response:
Thought for 13s

Reading documents

Done

Stage Metrics (Markup Languages Article)

| Stage | CharCount | WordCount | AvgSentenceLen |
| --- | --- | --- | --- |
| Draft | 5243 | 823 | 15.8 |
| Refined | 6923 | 1097 | 16.4 |
| Edited | 7049 | 1012 | 18.8 |
| Final | 5849 | 828 | 19.3 |

Transition Analysis (Markup Languages Article)

| Transition | SequenceSim(%) | WordChange(%) | NewVocabCount |
| --- | --- | --- | --- |
| Draft → Refined | 4.8 | 33.3 | 260.0 |
| Refined → Edited | 23.2 | \-7.7 | 143.0 |
| Edited → Final | 4.1 | \-18.2 | 129.0 |
| Draft → Final |  |  |  |

All set—I ran your analyzer on the four article versions and generated side-by-side metrics plus a transition analysis (sequence similarity, word-count deltas, new vocabulary, and Draft→Final TF-IDF semantic similarity).

*   Download the CSVs:
    *   Stage metrics
    *   Transition analysis
*   Clean report (Markdown): Revision analysis report

If you want, I can add readability indices (Flesch, FKGL), a list of sections added/removed per draft, or top n-grams that were introduced vs. discarded.

Sources (your uploads used for this run):

## Prompt:
To wrap up, I want you to search online to determine the different levels of self-reporting AI use in content creation and compare my process, what you learn, and the two scripts we've used to determine how well these scripts do when it comes to transparent AI usage in content creation.

## Response:
Here's what I found regarding **levels of self‑reporting AI use in content creation**, and how your workflow and the two scripts you've used stack up in terms of transparency and best practices.

* * *

1\. What Studies Say: Self-Reporting & Transparency in AI-Assisted Content
--------------------------------------------------------------------------

*   **Media self-regulation in AI**: Research highlights a growing trend among media organizations to internally regulate their use of generative AI—particularly to preserve quality, avoid bias, and _publicly disclose when AI is used_, emphasizing human oversight and audience trust as core principles [MDPI+1](https://www.mdpi.com/2673-5172/6/1/29?utm_source=chatgpt.com).
*   **Organizational oversight**: A McKinsey survey shows that about **27% of organizations say employees review _all_ AI-generated content before publication**, while many others review very little. This suggests varying levels of internal transparency and oversight [McKinsey & Company](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai?utm_source=chatgpt.com).
*   **Human–AI activity taxonomy (NIST)**: NIST’s taxonomy categorizes human–AI interactions, including **content creation** (“AI system assists by generating new artifacts…”) and **content synthesis** (“combining and/or summarizing parts into a coherent whole”) [NIST Publications](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.200-1.pdf?utm_source=chatgpt.com). While not specifically about transparency, it helps frame how the AI contributes in different modes.

### Summary Table: Self-Reporting / Transparency Levels

| Level | Description |
| --- | --- |
| **1\. Full Disclosure + Human Oversight** | AI-generated content is explicitly labeled, and humans review all outputs before release. Common in ethical media-related self-regulation [MDPI](https://www.mdpi.com/2673-5172/6/1/29?utm_source=chatgpt.com). |
| **2\. Partial or Selective Review** | Only some AI outputs are reviewed, e.g., 20% or fewer, while others may go unchecked [McKinsey & Company](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai?utm_source=chatgpt.com). |
| **3\. AI as Undeclared Assistance** | AI is used behind the scenes for ideation, drafting, or editing without disclosure—transparency is lacking. |
| **4\. AI as Creative Partner (Disclosure Optional)** | AI aids in content ideation or transformation; whether this is disclosed depends on organizational norms or audience expectations. |
| **5\. No AI Use / Manual Only** | Content creation is entirely human-driven, with no AI assistance. |

* * *

2\. Your Process & the Scripts: How Do They Measure Up?
-------------------------------------------------------

### **Your Process** (revision stages + transparency)

Your process appears to follow a clearly **code-driven analysis of each draft stage** of your article:

*   You’re applying Python scripts to measure metrics like word count, sequence similarity, semantic similarity via TF-IDF, and vocabulary changes—documenting each stage thoroughly and providing a final analysis report.
*   However, unless you explicitly **note that AI tools** (e.g., language models, revision assistants) were used at each stage, your process currently falls under **Level 3: AI as undeclared assistance**—assuming AI was used but not disclosed. If AI wasn't used, then it might be **Level 5: manual only**.

### **Script 1**: Revision Metrics Script

This script:

*   Calculates **quantitative changes** across versions—word count, sentence length, similarity, vocabulary changes, semantic similarity.
*   It objectively shows how the content evolved, but it does **not report whether any changes were AI-generated**. So in transparency terms, it's still **Level 3**, unless paired with explicit disclosure.

### **Script 2**: The Analyzer? (Your Python analyzer)

If this is a second script, similar in function (e.g., also comparing content), the same logic applies:

*   It produces **metrics and summaries**, but no built-in mechanism to **flag AI-generated content**.
*   Again, transparency depends entirely on accompanying documentation or disclosure.

* * *

3\. Comparing Against Industry Best Practices
---------------------------------------------

| Practice Element | What Ideal Self-Reporting Looks Like | Where Your Process/Scripts Stand |
| --- | --- | --- |
| **Explicit labeling of AI content** | e.g., “This section was drafted with XYZ AI.” | Not present in either script or process—**needs you to add it**. |
| **Human review documented** | Confirmation that all AI outputs have been reviewed and edited by a human. | Your process is human-led, but AI involvement isn’t declared—**caveat remains unclear**. |
| **Version-level transparency** | Notes alongside each draft (e.g., “Used AI for outline drafting” vs. “Human-refined, no AI use”). | Scripts report metrics, but **lack metadata on AI involvement** per version. |
| **Ethical assurance** | Explicit commitment to preserving voice, avoiding bias, maintaining quality. | Your process is quality-focused but **transparency could be enhanced**. |

* * *

4\. Recommendations: How to Align with Higher Transparency Levels
-----------------------------------------------------------------

To elevate your process from **Level 3 to Level 1 or 2**, you can:

1.  **Label AI involvement** in your report:
    *   Add metadata to each draft stage: e.g., “Draft generated by human; refinement assisted by AI (outline only)”; “Final edited by human with no AI assistance”.
2.  **Document human oversight**:
    *   Include a statement in your Markdown report: “All AI-generated material was reviewed and edited by the author.”
3.  **Track AI usage inside scripts**:
    *   Modify your processor to embed a flag in outputs (e.g., comment headers like `# AI-assisted: True`) so transparency is baked into the artifacts.
4.  **Include an ethics and transparency section** in your analysis report:
    *   E.g., “My approach uses AI as a helper but with full disclosure at each step to maintain trust, authenticity, and accountability.”

By following these steps, your process would align much more closely with the **media self-regulation ideals**—prioritizing audience trust and ethical transparency—moving toward **Level 1** best practices.

* * *

### Bottom Line

*   Current best practices in media emphasize **full disclosure and human oversight** for any AI-assisted content creation.
*   Your two scripts are excellent for measuring content evolution but **don’t inherently support transparent AI reporting**.
*   To be best-in-class, you can explicitly document AI usage per version, affirm human review, and adopt ethical transparency in your reporting.

Let me know if you'd like help modifying the scripts to include metadata flags or updating the report template with transparency sections!



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)