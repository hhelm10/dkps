# Linear-probe results

Held-out balanced accuracy (%); brackets are 95% group-bootstrap intervals.
Higher means the attribute is more linearly recoverable, not necessarily better.

| Representation | Target | N | Classes | Balanced accuracy [95% CI] | Chance |
|---|---|---:|---:|---:|---:|
| openai_qubric | system | 2129 | 107 | 20.3 [18.1, 22.7] | 0.9 |
| openai_qubric | model | 1191 | 36 | 28.8 [26.9, 30.9] | 2.8 |
| openai_qubric | vendor | 1591 | 8 | 36.2 [33.2, 38.8] | 12.5 |
| openai_qubric | harness | 2069 | 61 | 28.9 [26.2, 31.4] | 1.6 |
| openai_qubric | task | 2129 | 20 | 98.8 [98.2, 99.4] | 5.0 |
| openai_qubric | outcome | 2129 | 2 | 84.0 [81.8, 86.1] | 50.0 |
| openai_raw | system | 2129 | 107 | 89.2 [88.1, 90.5] | 0.9 |
| openai_raw | model | 1191 | 36 | 87.7 [86.3, 89.2] | 2.8 |
| openai_raw | vendor | 1591 | 8 | 97.2 [96.3, 97.9] | 12.5 |
| openai_raw | harness | 2069 | 61 | 96.0 [95.2, 96.8] | 1.6 |
| openai_raw | task | 2129 | 20 | 100.0 [100.0, 100.0] | 5.0 |
| openai_raw | outcome | 2129 | 2 | 81.8 [79.7, 83.9] | 50.0 |
| nomic_qubric | system | 2129 | 107 | 15.7 [13.9, 17.7] | 0.9 |
| nomic_qubric | model | 1191 | 36 | 23.1 [21.1, 25.3] | 2.8 |
| nomic_qubric | vendor | 1591 | 8 | 30.4 [26.5, 34.1] | 12.5 |
| nomic_qubric | harness | 2069 | 61 | 22.5 [19.9, 25.3] | 1.6 |
| nomic_qubric | task | 2129 | 20 | 98.6 [98.0, 99.2] | 5.0 |
| nomic_qubric | outcome | 2129 | 2 | 83.6 [81.6, 85.5] | 50.0 |
| nomic_raw | system | 2129 | 107 | 84.1 [82.4, 85.7] | 0.9 |
| nomic_raw | model | 1191 | 36 | 81.9 [80.0, 84.2] | 2.8 |
| nomic_raw | vendor | 1591 | 8 | 89.7 [87.5, 91.9] | 12.5 |
| nomic_raw | harness | 2069 | 61 | 91.5 [90.1, 92.8] | 1.6 |
| nomic_raw | task | 2129 | 20 | 98.3 [97.4, 99.0] | 5.0 |
| nomic_raw | outcome | 2129 | 2 | 78.4 [76.3, 80.7] | 50.0 |
| bge_qubric | system | 2129 | 107 | 15.7 [13.9, 17.5] | 0.9 |
| bge_qubric | model | 1191 | 36 | 23.6 [20.7, 26.9] | 2.8 |
| bge_qubric | vendor | 1591 | 8 | 33.5 [29.5, 36.7] | 12.5 |
| bge_qubric | harness | 2069 | 61 | 22.8 [20.6, 24.9] | 1.6 |
| bge_qubric | task | 2129 | 20 | 98.9 [98.4, 99.3] | 5.0 |
| bge_qubric | outcome | 2129 | 2 | 84.7 [82.8, 86.7] | 50.0 |
| bge_raw | system | 2129 | 107 | 73.5 [71.7, 75.3] | 0.9 |
| bge_raw | model | 1191 | 36 | 79.3 [76.9, 81.6] | 2.8 |
| bge_raw | vendor | 1591 | 8 | 79.5 [76.2, 82.8] | 12.5 |
| bge_raw | harness | 2069 | 61 | 85.7 [84.3, 87.3] | 1.6 |
| bge_raw | task | 2129 | 20 | 95.2 [93.0, 97.1] | 5.0 |
| bge_raw | outcome | 2129 | 2 | 78.1 [76.1, 80.2] | 50.0 |
| gte_qubric | system | 2129 | 107 | 17.0 [15.2, 18.9] | 0.9 |
| gte_qubric | model | 1191 | 36 | 23.4 [20.6, 26.0] | 2.8 |
| gte_qubric | vendor | 1591 | 8 | 32.7 [28.6, 36.7] | 12.5 |
| gte_qubric | harness | 2069 | 61 | 22.6 [20.2, 25.0] | 1.6 |
| gte_qubric | task | 2129 | 20 | 98.9 [98.4, 99.4] | 5.0 |
| gte_qubric | outcome | 2129 | 2 | 83.6 [81.5, 85.8] | 50.0 |
| gte_raw | system | 2129 | 107 | 77.4 [75.5, 79.2] | 0.9 |
| gte_raw | model | 1191 | 36 | 80.4 [78.7, 82.1] | 2.8 |
| gte_raw | vendor | 1591 | 8 | 83.5 [80.7, 85.8] | 12.5 |
| gte_raw | harness | 2069 | 61 | 89.9 [88.4, 91.2] | 1.6 |
| gte_raw | task | 2129 | 20 | 95.0 [92.7, 96.8] | 5.0 |
| gte_raw | outcome | 2129 | 2 | 77.5 [75.2, 79.8] | 50.0 |
| e5_qubric | system | 2129 | 107 | 20.8 [18.6, 23.1] | 0.9 |
| e5_qubric | model | 1191 | 36 | 28.7 [26.0, 31.3] | 2.8 |
| e5_qubric | vendor | 1591 | 8 | 37.4 [34.0, 40.7] | 12.5 |
| e5_qubric | harness | 2069 | 61 | 28.7 [25.6, 31.9] | 1.6 |
| e5_qubric | task | 2129 | 20 | 98.7 [98.2, 99.3] | 5.0 |
| e5_qubric | outcome | 2129 | 2 | 84.9 [82.8, 86.9] | 50.0 |
| e5_raw | system | 2129 | 107 | 78.6 [76.6, 80.5] | 0.9 |
| e5_raw | model | 1191 | 36 | 81.7 [79.2, 84.0] | 2.8 |
| e5_raw | vendor | 1591 | 8 | 86.0 [82.5, 89.1] | 12.5 |
| e5_raw | harness | 2069 | 61 | 90.4 [88.5, 92.3] | 1.6 |
| e5_raw | task | 2129 | 20 | 95.5 [93.4, 97.3] | 5.0 |
| e5_raw | outcome | 2129 | 2 | 77.0 [74.6, 79.1] | 50.0 |
| mpnet_qubric | system | 2129 | 107 | 11.1 [9.8, 12.5] | 0.9 |
| mpnet_qubric | model | 1191 | 36 | 17.9 [15.5, 20.2] | 2.8 |
| mpnet_qubric | vendor | 1591 | 8 | 28.8 [25.4, 32.3] | 12.5 |
| mpnet_qubric | harness | 2069 | 61 | 16.0 [13.4, 18.9] | 1.6 |
| mpnet_qubric | task | 2129 | 20 | 98.7 [98.2, 99.2] | 5.0 |
| mpnet_qubric | outcome | 2129 | 2 | 83.4 [81.5, 85.3] | 50.0 |
| mpnet_raw | system | 2129 | 107 | 68.3 [66.3, 70.3] | 0.9 |
| mpnet_raw | model | 1191 | 36 | 73.6 [71.1, 76.0] | 2.8 |
| mpnet_raw | vendor | 1591 | 8 | 75.7 [72.7, 78.4] | 12.5 |
| mpnet_raw | harness | 2069 | 61 | 81.5 [80.0, 83.0] | 1.6 |
| mpnet_raw | task | 2129 | 20 | 92.9 [90.2, 95.4] | 5.0 |
| mpnet_raw | outcome | 2129 | 2 | 77.2 [75.1, 79.1] | 50.0 |
| minilm_qubric | system | 2129 | 107 | 11.0 [9.2, 12.9] | 0.9 |
| minilm_qubric | model | 1191 | 36 | 18.3 [16.3, 20.1] | 2.8 |
| minilm_qubric | vendor | 1591 | 8 | 30.7 [27.8, 33.9] | 12.5 |
| minilm_qubric | harness | 2069 | 61 | 16.4 [13.6, 19.2] | 1.6 |
| minilm_qubric | task | 2129 | 20 | 98.6 [98.1, 99.1] | 5.0 |
| minilm_qubric | outcome | 2129 | 2 | 82.4 [80.1, 84.7] | 50.0 |
| minilm_raw | system | 2129 | 107 | 70.5 [68.6, 72.6] | 0.9 |
| minilm_raw | model | 1191 | 36 | 76.1 [73.3, 78.8] | 2.8 |
| minilm_raw | vendor | 1591 | 8 | 75.4 [71.6, 78.9] | 12.5 |
| minilm_raw | harness | 2069 | 61 | 85.7 [84.0, 87.3] | 1.6 |
| minilm_raw | task | 2129 | 20 | 91.3 [88.5, 93.9] | 5.0 |
| minilm_raw | outcome | 2129 | 2 | 74.6 [72.5, 76.8] | 50.0 |

See README.md for the classifier, splits, preprocessing, and interpretation limits.
