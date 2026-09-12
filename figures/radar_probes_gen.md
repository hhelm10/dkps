# Linear-probe results

Held-out balanced accuracy (%); brackets are 95% group-bootstrap intervals.
Higher means the attribute is more linearly recoverable, not necessarily better.

| Representation | Target | N | Classes | Balanced accuracy [95% CI] | Chance |
|---|---|---:|---:|---:|---:|
| openai_generic | system | 2129 | 107 | 60.9 [58.6, 63.4] | 0.9 |
| openai_generic | model | 1191 | 36 | 63.0 [60.2, 65.8] | 2.8 |
| openai_generic | vendor | 1591 | 8 | 64.2 [60.5, 67.8] | 12.5 |
| openai_generic | harness | 2069 | 61 | 76.5 [74.7, 78.6] | 1.6 |
| openai_generic | task | 2129 | 20 | 98.9 [97.7, 99.9] | 5.0 |
| openai_generic | outcome | 2129 | 2 | 83.9 [81.2, 86.5] | 50.0 |
| nomic_generic | system | 2129 | 107 | 55.5 [54.0, 57.2] | 0.9 |
| nomic_generic | model | 1191 | 36 | 59.2 [57.1, 61.4] | 2.8 |
| nomic_generic | vendor | 1591 | 8 | 62.7 [58.8, 66.1] | 12.5 |
| nomic_generic | harness | 2069 | 61 | 69.6 [67.4, 71.8] | 1.6 |
| nomic_generic | task | 2129 | 20 | 98.9 [97.6, 99.8] | 5.0 |
| nomic_generic | outcome | 2129 | 2 | 85.6 [83.8, 87.5] | 50.0 |
| bge_generic | system | 2129 | 107 | 58.1 [56.3, 60.0] | 0.9 |
| bge_generic | model | 1191 | 36 | 59.6 [57.0, 62.4] | 2.8 |
| bge_generic | vendor | 1591 | 8 | 64.2 [61.0, 68.1] | 12.5 |
| bge_generic | harness | 2069 | 61 | 72.7 [70.8, 74.6] | 1.6 |
| bge_generic | task | 2129 | 20 | 98.9 [97.6, 99.9] | 5.0 |
| bge_generic | outcome | 2129 | 2 | 83.6 [81.2, 86.1] | 50.0 |
| gte_generic | system | 2129 | 107 | 55.9 [54.0, 57.8] | 0.9 |
| gte_generic | model | 1191 | 36 | 59.4 [56.6, 62.6] | 2.8 |
| gte_generic | vendor | 1591 | 8 | 59.2 [55.6, 63.3] | 12.5 |
| gte_generic | harness | 2069 | 61 | 70.1 [67.8, 72.7] | 1.6 |
| gte_generic | task | 2129 | 20 | 98.9 [97.6, 99.9] | 5.0 |
| gte_generic | outcome | 2129 | 2 | 84.7 [82.1, 87.1] | 50.0 |
| e5_generic | system | 2129 | 107 | 62.4 [60.6, 64.5] | 0.9 |
| e5_generic | model | 1191 | 36 | 64.2 [61.4, 67.0] | 2.8 |
| e5_generic | vendor | 1591 | 8 | 65.8 [62.2, 69.2] | 12.5 |
| e5_generic | harness | 2069 | 61 | 76.7 [74.6, 79.1] | 1.6 |
| e5_generic | task | 2129 | 20 | 99.1 [97.9, 99.9] | 5.0 |
| e5_generic | outcome | 2129 | 2 | 84.6 [82.2, 86.9] | 50.0 |
| mpnet_generic | system | 2129 | 107 | 42.6 [40.6, 45.0] | 0.9 |
| mpnet_generic | model | 1191 | 36 | 49.4 [46.6, 52.4] | 2.8 |
| mpnet_generic | vendor | 1591 | 8 | 51.8 [48.0, 56.0] | 12.5 |
| mpnet_generic | harness | 2069 | 61 | 56.8 [54.1, 59.8] | 1.6 |
| mpnet_generic | task | 2129 | 20 | 99.0 [97.8, 99.9] | 5.0 |
| mpnet_generic | outcome | 2129 | 2 | 84.3 [81.5, 86.9] | 50.0 |
| minilm_generic | system | 2129 | 107 | 45.1 [42.5, 47.9] | 0.9 |
| minilm_generic | model | 1191 | 36 | 52.6 [50.3, 54.5] | 2.8 |
| minilm_generic | vendor | 1591 | 8 | 53.0 [49.3, 56.9] | 12.5 |
| minilm_generic | harness | 2069 | 61 | 59.2 [56.6, 61.7] | 1.6 |
| minilm_generic | task | 2129 | 20 | 98.8 [97.5, 99.9] | 5.0 |
| minilm_generic | outcome | 2129 | 2 | 84.1 [81.8, 86.5] | 50.0 |

See README.md for the classifier, splits, preprocessing, and interpretation limits.
