# Markup Languages Article – Revision Analysis

## Stage Metrics

| Stage   |   CharCount |   WordCount |   AvgSentenceLen |
|:--------|------------:|------------:|-----------------:|
| Draft   |        5243 |         823 |             15.8 |
| Refined |        6923 |        1097 |             16.4 |
| Edited  |        7049 |        1012 |             18.8 |
| Final   |        5849 |         828 |             19.3 |


## Transition Analysis

| Transition       |   SequenceSim(%) |   WordChange(%) |   NewVocabCount |   SemanticSim(%) |
|:-----------------|-----------------:|----------------:|----------------:|-----------------:|
| Draft → Refined  |              4.8 |            33.3 |             260 |            nan   |
| Refined → Edited |             23.2 |            -7.7 |             143 |            nan   |
| Edited → Final   |              4.1 |           -18.2 |             129 |            nan   |
| Draft → Final    |            nan   |           nan   |             nan |             38.4 |


**Largest word-count change:** Draft → Refined (33.3%).