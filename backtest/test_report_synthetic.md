# Backtest report — synthetic panel (pipeline test)

Items 12, groups ['battery_materials', 'refractory_minor_metals', 'semiconductor_chemicals'], months 84, origins 36..72 (4), item-years at decision months 48.0, fit target N1.

Events in the evaluation window: trade-based N1 = 549, restriction N2 = 8, union N3 = 557; restriction events with alarm history = 8.

q-channel calibration: threshold {'M2': 0.95, 'M0': 0.95}, persistence {'M2': 4, 'M0': 4}, implied in-control run length {'M2': 9.7, 'M0': 18.0} months.

## Target N1: trade-based interruption events

| Method | Level | Detected ≤1 m | ≤3 m | False onsets / item-year | Alarm months / item-year | Lead (median) | AUC-PR | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| M2 (proposed) | L1 | 0.99 | 1.00 | 0.17 | 11.8 | 5.0 |  |  |  |
| M2 (proposed) | L2 | 0.99 | 1.00 | 0.17 | 11.5 | 4.0 | 0.883 | 0.177 | 0.077 |
| M2 (proposed) | L3 | 0.87 | 0.91 | 0.02 | 9.4 | 2.0 |  |  |  |
| Poisson M0 | L1 | 0.99 | 1.00 | 0.17 | 11.8 | 5.0 |  |  |  |
| Poisson M0 | L2 | 0.99 | 1.00 | 0.17 | 11.5 | 4.5 | 0.873 | 0.193 | 0.132 |
| Poisson M0 | L3 | 0.84 | 0.88 | 0.02 | 9.1 | 1.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L1 | 0.57 | 0.63 | 0.17 | 6.7 | 2.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L2 | 0.38 | 0.51 | 0.15 | 4.6 | 2.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L3 | 0.24 | 0.30 | 0.19 | 2.5 | 2.0 |  |  |  |
| EWMA |  | 0.64 | 0.73 | 0.08 | 6.5 | 1.5 |  |  |  |
| CUSUM |  | 0.58 | 0.59 | 0.06 | 5.6 | 1.0 |  |  |  |
| Farrington-type |  | 0.08 | 0.18 | 0.12 | 0.7 | 1.0 |  |  |  |
| XGBoost (pooled) | L1 | 0.99 | 1.00 | 0.17 | 11.8 | 5.0 |  |  |  |
| XGBoost (pooled) | L2 | 0.93 | 0.99 | 0.23 | 10.8 | 2.5 | 0.811 | 0.231 | 0.190 |
| XGBoost (pooled) | L3 | 0.71 | 0.79 | 0.10 | 7.5 | 2.0 |  |  |  |
| XGBoost, recalibrated | L1 | 0.99 | 1.00 | 0.17 | 11.6 | 4.5 |  |  |  |
| XGBoost, recalibrated | L2 | 0.94 | 1.00 | 0.23 | 11.0 | 3.0 | 0.772 | 0.219 | 0.146 |
| XGBoost, recalibrated | L3 | 0.83 | 0.89 | 0.08 | 9.1 | 2.0 |  |  |  |

M2 change channel (q): detected ≤1 m 0.81, share of months in force 0.75, onsets 12; headline level (max of triage level 2 and q): detected ≤1 m 0.99, false onsets 0.17, alarm months 11.5.

Cost (Model A, E = 5.1042 events per item-year): band of r in which each policy beats both never and always acting, and its largest saving against the better trivial policy

| Method | band of r | max saving | at r |
|---|---|---|---|
| M2 (proposed) | 2.12–8.08 | 8.7% | 2.35 |
| Poisson M0 | 2.14–11.03 | 7.5% | 2.35 |
| Sigma-rule (plug-in SRI) | 2.07–2.43 | 3.0% | 2.35 |
| EWMA | 1.98–3.03 | 10.5% | 2.35 |
| CUSUM | 1.91–2.99 | 11.4% | 2.35 |
| Farrington-type | 1.68–2.4 | 2.4% | 2.35 |
| XGBoost (pooled) | 2.07–5.98 | 8.6% | 2.35 |
| XGBoost, recalibrated | 2.17–5.71 | 6.7% | 2.35 |

## Target N2: export-restriction implementations (independent)

| Method | Level | Detected ≤1 m | ≤3 m | False onsets / item-year | Alarm months / item-year | Lead (median) | AUC-PR | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| M2 (proposed) | L1 | 1.00 | 1.00 | 0.25 | 11.8 | 6.0 |  |  |  |
| M2 (proposed) | L2 | 1.00 | 1.00 | 0.31 | 11.5 | 6.0 | 0.028 | 0.567 | 0.684 |
| M2 (proposed) | L3 | 0.50 | 0.50 | 0.44 | 9.4 | 5.5 |  |  |  |
| Poisson M0 | L1 | 1.00 | 1.00 | 0.25 | 11.8 | 6.0 |  |  |  |
| Poisson M0 | L2 | 1.00 | 1.00 | 0.29 | 11.5 | 6.0 | 0.026 | 0.609 | 0.705 |
| Poisson M0 | L3 | 0.62 | 0.62 | 0.38 | 9.1 | 3.5 |  |  |  |
| Sigma-rule (plug-in SRI) | L1 | 0.88 | 0.88 | 0.54 | 6.7 | 7.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L2 | 0.75 | 0.88 | 0.69 | 4.6 | 5.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L3 | 0.38 | 0.50 | 0.58 | 2.5 | 5.0 |  |  |  |
| EWMA |  | 0.38 | 0.38 | 0.71 | 6.5 | 1.0 |  |  |  |
| CUSUM |  | 0.00 | 0.00 | 0.29 | 5.6 |  |  |  |  |
| Farrington-type |  | 0.00 | 0.25 | 0.54 | 0.7 | 3.0 |  |  |  |
| XGBoost (pooled) | L1 | 1.00 | 1.00 | 0.25 | 11.8 | 6.0 |  |  |  |
| XGBoost (pooled) | L2 | 1.00 | 1.00 | 0.60 | 10.8 | 6.0 | 0.033 | 0.476 | 0.607 |
| XGBoost (pooled) | L3 | 0.75 | 0.88 | 0.62 | 7.5 | 3.0 |  |  |  |
| XGBoost, recalibrated | L1 | 1.00 | 1.00 | 0.29 | 11.6 | 6.0 |  |  |  |
| XGBoost, recalibrated | L2 | 1.00 | 1.00 | 0.56 | 11.0 | 6.0 | 0.030 | 0.489 | 0.627 |
| XGBoost, recalibrated | L3 | 0.75 | 0.88 | 0.40 | 9.1 | 3.0 |  |  |  |

M2 change channel (q): detected ≤1 m 0.50, share of months in force 0.75, onsets 12; headline level (max of triage level 2 and q): detected ≤1 m 1.00, false onsets 0.29, alarm months 11.5.

Cost (Model A, E = 0.1667 events per item-year): band of r in which each policy beats both never and always acting, and its largest saving against the better trivial policy

| Method | band of r | max saving | at r |
|---|---|---|---|
| M2 (proposed) | 69.12–100.0 | 4.2% | 72.39 |
| Poisson M0 | 69.92–100.0 | 3.8% | 72.39 |
| Sigma-rule (plug-in SRI) | 37.06–100.0 | 36.3% | 72.39 |
| EWMA | none | -16.7% | 72.39 |
| CUSUM | none | -47.2% | 71.55 |
| Farrington-type | none | -5.8% | 71.55 |
| XGBoost (pooled) | 60.18–100.0 | 12.7% | 72.39 |
| XGBoost, recalibrated | 66.0–100.0 | 8.5% | 72.39 |

## Target N3: union

| Method | Level | Detected ≤1 m | ≤3 m | False onsets / item-year | Alarm months / item-year | Lead (median) | AUC-PR | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| M2 (proposed) | L1 | 0.99 | 1.00 | 0.17 | 11.8 | 5.0 |  |  |  |
| M2 (proposed) | L2 | 0.99 | 1.00 | 0.17 | 11.5 | 4.0 | 0.883 | 0.178 | 0.081 |
| M2 (proposed) | L3 | 0.86 | 0.90 | 0.02 | 9.4 | 2.0 |  |  |  |
| Poisson M0 | L1 | 0.99 | 1.00 | 0.17 | 11.8 | 5.0 |  |  |  |
| Poisson M0 | L2 | 0.99 | 1.00 | 0.17 | 11.5 | 4.5 | 0.873 | 0.193 | 0.135 |
| Poisson M0 | L3 | 0.83 | 0.87 | 0.02 | 9.1 | 1.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L1 | 0.57 | 0.63 | 0.17 | 6.7 | 2.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L2 | 0.39 | 0.52 | 0.12 | 4.6 | 2.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L3 | 0.24 | 0.30 | 0.19 | 2.5 | 2.0 |  |  |  |
| EWMA |  | 0.64 | 0.72 | 0.08 | 6.5 | 1.5 |  |  |  |
| CUSUM |  | 0.58 | 0.59 | 0.06 | 5.6 | 1.0 |  |  |  |
| Farrington-type |  | 0.08 | 0.18 | 0.10 | 0.7 | 1.0 |  |  |  |
| XGBoost (pooled) | L1 | 0.99 | 1.00 | 0.17 | 11.8 | 5.0 |  |  |  |
| XGBoost (pooled) | L2 | 0.93 | 0.99 | 0.23 | 10.8 | 2.5 | 0.813 | 0.230 | 0.190 |
| XGBoost (pooled) | L3 | 0.71 | 0.80 | 0.10 | 7.5 | 2.0 |  |  |  |
| XGBoost, recalibrated | L1 | 0.99 | 1.00 | 0.17 | 11.6 | 4.5 |  |  |  |
| XGBoost, recalibrated | L2 | 0.94 | 1.00 | 0.23 | 11.0 | 3.0 | 0.774 | 0.218 | 0.146 |
| XGBoost, recalibrated | L3 | 0.83 | 0.89 | 0.08 | 9.1 | 2.0 |  |  |  |

M2 change channel (q): detected ≤1 m 0.81, share of months in force 0.75, onsets 12; headline level (max of triage level 2 and q): detected ≤1 m 0.99, false onsets 0.17, alarm months 11.5.

Cost (Model A, E = 5.1667 events per item-year): band of r in which each policy beats both never and always acting, and its largest saving against the better trivial policy

| Method | band of r | max saving | at r |
|---|---|---|---|
| M2 (proposed) | 2.12–7.98 | 8.1% | 2.32 |
| Poisson M0 | 2.14–11.03 | 7.0% | 2.32 |
| Sigma-rule (plug-in SRI) | 2.07–2.4 | 2.8% | 2.32 |
| EWMA | 1.98–2.93 | 9.8% | 2.32 |
| CUSUM | 1.91–2.89 | 10.7% | 2.32 |
| Farrington-type | 1.68–2.38 | 2.3% | 2.32 |
| XGBoost (pooled) | 2.05–5.98 | 8.6% | 2.32 |
| XGBoost, recalibrated | 2.14–5.65 | 6.6% | 2.32 |

## Export-restriction implementations: what was in force when each measure took effect

The posture dated t is set on the data through t - 1 (the convention of the one-month detection rate); the table also gives the level dated t - 1.

8 restriction events. M2 triage level ≥ 2 in force at the implementation month: 100%; level ≥ 1: 100%; q-alarm in force: 50%; median months since the item first reached the alert level: 8.0 (over 8 events).

Poisson M0 level ≥ 2 at the implementation month: 100%.
XGBoost level ≥ 2 at the implementation month: 100%.
XGBoost (recalibrated) level ≥ 2 at the implementation month: 100%.
sigma-rule level ≥ 2 at the implementation month: 75%.
EWMA alarm at the implementation month: 38%.
CUSUM alarm at the implementation month: 0%.
Farrington-type alarm at the implementation month: 0%.

| Item | Group | Month | Measure | M2 p(3) | M2 level | M2 level the month before | months since first alert | q-alarm |
|---|---|---|---|---|---|---|---|---|
| 900000 | battery_materials | 201807 | 156:synthetic restriction | 0.467 | 2 | 2 | 5 | False |
| 900000 | battery_materials | 201901 | 156:synthetic restriction | 0.765 | 2 | 2 | 11 | True |
| 900003 | battery_materials | 201807 | 156:synthetic restriction | 0.639 | 2 | 2 | 5 | False |
| 900003 | battery_materials | 201901 | 156:synthetic restriction | 0.833 | 3 | 2 | 11 | True |
| 900006 | battery_materials | 201807 | 156:synthetic restriction | 0.754 | 3 | 3 | 5 | False |
| 900006 | battery_materials | 201901 | 156:synthetic restriction | 0.802 | 3 | 3 | 11 | True |
| 900009 | battery_materials | 201807 | 156:synthetic restriction | 0.488 | 2 | 2 | 5 | False |
| 900009 | battery_materials | 201901 | 156:synthetic restriction | 0.965 | 3 | 3 | 11 | True |
