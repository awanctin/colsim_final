# certificate

COLREGs duty model and safety certificate.
Answers two questions each step: which headings are lawful, and does any safe heading remain.

| File | Role |
|---|---|
| `geometry.py` | `Vessel` state and angle wrap |
| `duty.py` | Situation, role toward each target, lawful heading intervals, overtaking memory |
| `escape_angle.py` | CPA, COLREGs class, physical and compliant escape width, encounter placement |
| `alpha.py` | Exact width of headings that avoid every target (alpha), with or without tubes |
| `tube.py` | `Tube`: reachable set of a target per behaviour class |
| `behaviour.py` | Target class: compliant, broken down or adversarial, Rule 17 relaxation |
| `certificate.py` | Deontic certificate (no lawful heading left) and geometric certificate (every heading meets contact) |
| `monitor.py` | Predicts when a certificate will fire, over sampled futures |
| `_monitor_nb.py` | Numba version of `duty`, `alpha` and `monitor`, plus the policy observation |
| `cert_switch.py` | `CertificateSwitch`: compliant mode, emergency policy when a breach is predicted |

`tube.py` and `_monitor_nb.py` hold a `_selfcheck()`.
