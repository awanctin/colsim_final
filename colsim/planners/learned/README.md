# learned

Learned controllers from the literature.

| File | Role |
|---|---|
| `algo_kras.py` | KRAS: provably safe RL under a COLREGs statechart, the state-of-the-art hybrid arm |

The `Shield` masks unsafe actions from the most urgent target pair.
`Emergency` takes over when a target can reach the own path.
A trained policy is needed (`COLSIM_KRAS_POLICY`), none is shipped.
