| | High-fidelity stage | Stage surrogate | Tube surrogate chain |
|---|---|---|---|
| Test MAPE, all cations (%) | - | 0.17 | 0.52 |
| Test MAPE, Na / Ca / La (%) | - | 0.04 / 0.03 / 0.43 | 0.43 / 0.24 / 0.89 |
| Test maximum error (%) | - | 0.6 | 1.4 |
| Variables | 10,763 | 4 | 41 |
| Constraints | 9,985 | 1 | 33 |
| Solve time (s) | ~2.7 (continuation move) | 0.038 | 0.102 |
| Training HF runs | - | 48 stage solves | 186 tube solves |
| Training cost (tube-solve equiv.) | - | 240 | 186 |
| Kriging training time (s) | - | 5.8 | 147.0 |
| Reusable for other stage designs | - | No | Yes |

Test set: 18 held-out high-fidelity stage conditions (6 per cation). Solve time of the HF stage is a continuation move (several solves). HF size counts unit models only.
