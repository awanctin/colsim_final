# evaluation

Encounter banks and rule scoring.

| File | Role |
|---|---|
| `bank.py` | Draws, labels and writes encounters to CSV, balanced per fleet size and situation, resumes a partial file |
| `compliance_check.py` | Per-pair rule verdicts (give way, stand on) and Woerner score of an episode |

`bank.py` runs from the command line through `main()`.
