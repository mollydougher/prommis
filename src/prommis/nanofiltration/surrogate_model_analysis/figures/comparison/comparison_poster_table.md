| | High-fidelity stage | Stage surrogate | Tube surrogate chain |
|---|---|---|---|
| Test MAPE, all cations (%) | - | 0.60 | 0.52 |
| Test MAPE, Na / Ca / La (%) | - | 0.22 / 0.37 / 1.22 | 0.43 / 0.24 / 0.89 |
| Test maximum error (%) | - | 1.8 | 1.4 |
| Variables | 10,763 | 4 | 41 |
| Constraints | 9,985 | 1 | 33 |
| Solve time (s) | ~2.7 (continuation move) | 0.027 | 0.059 |
| Training HF runs | - | 27 stage solves | 186 tube solves |
| Training cost (tube-solve equiv.) | - | 135 | 186 |
| Kriging training time (s) | - | 3.6 | 82.3 |
| Reusable for other stage designs | - | No | Yes |

Test set: 18 held-out high-fidelity stage conditions (6 per cation). Solve time of the HF stage is a continuation move (several solves). HF size counts unit models only.
