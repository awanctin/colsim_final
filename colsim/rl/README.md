# rl

Hybrid controller, learned part.
RRT* sails compliant, the certificate switch hands control to this policy when a breach is predicted, the policy hands it back once clear.

| File | Role |
|---|---|
| `cert_env.py` | `CertifiedEscapeEnv`: Gymnasium env over a bank of encounters, reward, release rule, action shield |
| `cert_obs.py` | Policy observation from the certificate quantities, action decoding |
| `train_cert.py` | Trains the policy (PPO or SAC, stable-baselines3), held out evaluation, training figures |
| `evaluate_cert.py` | Runs arms on bank rows, rule verdicts, paired McNemar test, JSON output |
| `policy_io.py` | Loads a saved policy whatever algorithm wrote it |
| `fleet_bank.py` | Mines encounters of 1-4 targets that trap a rule-based planner |
| `reactive_bank.py` | Mines two-target encounters against reactive targets, used by `fleet_bank` |
| `episode_figs.py` | Per episode figures: actions, reward split, positions |
| `figure_scenario.py` | To-scale picture of one scenario and its escape fan |
| `night_figs.py` | Shared figure style and the overnight run figures |
| `kras_env.py` | Training env of the KRAS baseline (`planners/learned/algo_kras.py`) |
| `train_kras.py` | Trains KRAS with masked PPO, needs `sb3-contrib` |

Kept: every module reachable from the training and evaluation entry points, plus the bank miners and the KRAS trainer.
Left in `aaaa_colsim`: reports, diagnostics, sweeps and viewers (`compare_arms`, `diagnose_policy`, `trace_row`, `watch_policy`, ...).
