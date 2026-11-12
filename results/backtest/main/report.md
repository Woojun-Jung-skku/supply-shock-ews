# Backtest report — main

Items 64, groups ['battery_materials', 'ferroalloys', 'rare_earths_magnets', 'refractory_minor_metals', 'semiconductor_chemicals'], months 143, origins 36..138 (18), item-years at decision months 570.67, fit target N1.

Events in the evaluation window: trade-based N1 = 1295, restriction N2 = 65, union N3 = 1360; restriction events with alarm history = 64.

q-channel calibration: threshold {'M2': 0.95, 'M0': 0.95}, persistence {'M2': 4, 'M0': 4}, implied in-control run length {'M2': 2.2, 'M0': 1.4} months.

## Target N1: trade-based interruption events

| Method | Level | Detected ≤1 m | ≤3 m | False onsets / item-year | Alarm months / item-year | Lead (median) | AUC-PR | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| M2 (proposed) | L1 | 0.98 | 0.99 | 0.18 | 11.3 | 12.0 |  |  |  |
| M2 (proposed) | L2 | 0.91 | 0.93 | 0.20 | 8.2 | 8.0 | 0.600 | 0.184 | 0.029 |
| M2 (proposed) | L3 | 0.54 | 0.60 | 0.13 | 3.4 | 3.0 |  |  |  |
| Poisson M0 | L1 | 0.99 | 1.00 | 0.12 | 11.7 | 12.0 |  |  |  |
| Poisson M0 | L2 | 0.93 | 0.94 | 0.12 | 8.6 | 6.0 | 0.603 | 0.187 | 0.064 |
| Poisson M0 | L3 | 0.64 | 0.68 | 0.10 | 4.3 | 3.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L1 | 0.42 | 0.47 | 0.14 | 3.3 | 2.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L2 | 0.30 | 0.35 | 0.14 | 2.0 | 2.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L3 | 0.19 | 0.24 | 0.13 | 1.2 | 2.0 |  |  |  |
| EWMA |  | 0.24 | 0.29 | 0.11 | 1.4 | 2.0 |  |  |  |
| CUSUM |  | 0.18 | 0.19 | 0.02 | 1.4 | 1.0 |  |  |  |
| Farrington-type |  | 0.03 | 0.06 | 0.07 | 0.2 | 2.0 |  |  |  |
| XGBoost (pooled) | L1 | 0.98 | 1.00 | 0.22 | 10.5 | 11.0 |  |  |  |
| XGBoost (pooled) | L2 | 0.88 | 0.91 | 0.25 | 6.8 | 6.0 | 0.661 | 0.167 | 0.022 |
| XGBoost (pooled) | L3 | 0.57 | 0.66 | 0.15 | 3.1 | 2.0 |  |  |  |
| XGBoost, recalibrated | L1 | 0.98 | 0.99 | 0.25 | 10.3 | 11.0 |  |  |  |
| XGBoost, recalibrated | L2 | 0.87 | 0.91 | 0.26 | 7.0 | 6.0 | 0.641 | 0.172 | 0.027 |
| XGBoost, recalibrated | L3 | 0.55 | 0.65 | 0.16 | 3.1 | 2.0 |  |  |  |

M2 change channel (q): detected ≤1 m 0.00, share of months in force 0.00, onsets 0; headline level (max of triage level 2 and q): detected ≤1 m 0.91, false onsets 0.20, alarm months 8.2.

Cost (Model A, E = 2.1063 events per item-year): band of r in which each policy beats both never and always acting, and its largest saving against the better trivial policy

| Method | band of r | max saving | at r |
|---|---|---|---|
| M2 (proposed) | 3.03–21.05 | 25.4% | 5.71 |
| Poisson M0 | 3.21–22.82 | 28.0% | 5.71 |
| Sigma-rule (plug-in SRI) | 2.96–7.11 | 14.7% | 5.71 |
| EWMA | 2.83–6.56 | 12.1% | 5.65 |
| CUSUM | 3.6–6.12 | 6.6% | 5.65 |
| Farrington-type | 2.73–5.78 | 1.5% | 5.65 |
| XGBoost (pooled) | 2.64–44.58 | 31.2% | 5.71 |
| XGBoost, recalibrated | 2.67–35.39 | 29.4% | 5.71 |

## Target N2: export-restriction implementations (independent)

| Method | Level | Detected ≤1 m | ≤3 m | False onsets / item-year | Alarm months / item-year | Lead (median) | AUC-PR | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| M2 (proposed) | L1 | 0.91 | 0.91 | 0.22 | 11.3 | 58.0 |  |  |  |
| M2 (proposed) | L2 | 0.61 | 0.66 | 0.26 | 8.2 | 52.0 | 0.025 | 0.205 | 0.341 |
| M2 (proposed) | L3 | 0.22 | 0.30 | 0.32 | 3.4 | 31.0 |  |  |  |
| Poisson M0 | L1 | 0.94 | 0.94 | 0.15 | 11.7 | 61.0 |  |  |  |
| Poisson M0 | L2 | 0.69 | 0.69 | 0.17 | 8.6 | 58.0 | 0.029 | 0.235 | 0.378 |
| Poisson M0 | L3 | 0.33 | 0.36 | 0.19 | 4.3 | 53.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L1 | 0.20 | 0.30 | 0.34 | 3.3 | 34.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L2 | 0.16 | 0.22 | 0.33 | 2.0 | 35.5 |  |  |  |
| Sigma-rule (plug-in SRI) | L3 | 0.09 | 0.12 | 0.29 | 1.2 | 27.0 |  |  |  |
| EWMA |  | 0.11 | 0.14 | 0.29 | 1.4 | 31.5 |  |  |  |
| CUSUM |  | 0.12 | 0.12 | 0.07 | 1.4 | 29.0 |  |  |  |
| Farrington-type |  | 0.00 | 0.03 | 0.15 | 0.2 | 33.0 |  |  |  |
| XGBoost (pooled) | L1 | 0.92 | 0.95 | 0.26 | 10.5 | 43.0 |  |  |  |
| XGBoost (pooled) | L2 | 0.48 | 0.61 | 0.36 | 6.8 | 40.5 | 0.028 | 0.196 | 0.314 |
| XGBoost (pooled) | L3 | 0.23 | 0.30 | 0.41 | 3.1 | 33.5 |  |  |  |
| XGBoost, recalibrated | L1 | 0.92 | 0.95 | 0.31 | 10.3 | 59.0 |  |  |  |
| XGBoost, recalibrated | L2 | 0.48 | 0.62 | 0.39 | 7.0 | 36.0 | 0.026 | 0.196 | 0.319 |
| XGBoost, recalibrated | L3 | 0.25 | 0.31 | 0.46 | 3.1 | 34.0 |  |  |  |

M2 change channel (q): detected ≤1 m 0.00, share of months in force 0.00, onsets 0; headline level (max of triage level 2 and q): detected ≤1 m 0.61, false onsets 0.26, alarm months 8.2.

Cost (Model A, E = 0.1121 events per item-year): band of r in which each policy beats both never and always acting, and its largest saving against the better trivial policy

| Method | band of r | max saving | at r |
|---|---|---|---|
| M2 (proposed) | none | -8.4% | 100.0 |
| Poisson M0 | none | -5.7% | 100.0 |
| Sigma-rule (plug-in SRI) | none | -1.4% | 100.0 |
| EWMA | none | -1.8% | 100.0 |
| CUSUM | 98.85–100.0 | 0.2% | 100.0 |
| Farrington-type | none | -1.5% | 100.0 |
| XGBoost (pooled) | none | -1.6% | 100.0 |
| XGBoost, recalibrated | 100.0–100.0 | 0.1% | 100.0 |

## Target N3: union

| Method | Level | Detected ≤1 m | ≤3 m | False onsets / item-year | Alarm months / item-year | Lead (median) | AUC-PR | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| M2 (proposed) | L1 | 0.98 | 0.99 | 0.18 | 11.3 | 12.0 |  |  |  |
| M2 (proposed) | L2 | 0.90 | 0.92 | 0.19 | 8.2 | 7.0 | 0.602 | 0.192 | 0.036 |
| M2 (proposed) | L3 | 0.52 | 0.59 | 0.13 | 3.4 | 3.0 |  |  |  |
| Poisson M0 | L1 | 0.98 | 0.99 | 0.12 | 11.7 | 12.0 |  |  |  |
| Poisson M0 | L2 | 0.92 | 0.93 | 0.12 | 8.6 | 6.0 | 0.607 | 0.193 | 0.058 |
| Poisson M0 | L3 | 0.63 | 0.66 | 0.09 | 4.3 | 3.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L1 | 0.41 | 0.47 | 0.14 | 3.3 | 2.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L2 | 0.29 | 0.34 | 0.14 | 2.0 | 2.0 |  |  |  |
| Sigma-rule (plug-in SRI) | L3 | 0.19 | 0.23 | 0.13 | 1.2 | 2.0 |  |  |  |
| EWMA |  | 0.23 | 0.28 | 0.11 | 1.4 | 2.0 |  |  |  |
| CUSUM |  | 0.18 | 0.19 | 0.02 | 1.4 | 1.0 |  |  |  |
| Farrington-type |  | 0.03 | 0.06 | 0.07 | 0.2 | 3.0 |  |  |  |
| XGBoost (pooled) | L1 | 0.98 | 0.99 | 0.21 | 10.5 | 11.0 |  |  |  |
| XGBoost (pooled) | L2 | 0.86 | 0.89 | 0.25 | 6.8 | 6.0 | 0.662 | 0.176 | 0.035 |
| XGBoost (pooled) | L3 | 0.55 | 0.64 | 0.15 | 3.1 | 2.0 |  |  |  |
| XGBoost, recalibrated | L1 | 0.97 | 0.99 | 0.25 | 10.3 | 11.0 |  |  |  |
| XGBoost, recalibrated | L2 | 0.85 | 0.89 | 0.26 | 7.0 | 6.0 | 0.643 | 0.181 | 0.034 |
| XGBoost, recalibrated | L3 | 0.54 | 0.64 | 0.16 | 3.1 | 2.0 |  |  |  |

M2 change channel (q): detected ≤1 m 0.00, share of months in force 0.00, onsets 0; headline level (max of triage level 2 and q): detected ≤1 m 0.90, false onsets 0.19, alarm months 8.2.

Cost (Model A, E = 2.1939 events per item-year): band of r in which each policy beats both never and always acting, and its largest saving against the better trivial policy

| Method | band of r | max saving | at r |
|---|---|---|---|
| M2 (proposed) | 2.96–17.3 | 24.1% | 5.46 |
| Poisson M0 | 3.17–18.76 | 26.6% | 5.46 |
| Sigma-rule (plug-in SRI) | 2.96–6.72 | 13.6% | 5.46 |
| EWMA | 2.79–6.27 | 11.4% | 5.46 |
| CUSUM | 3.48–5.85 | 6.5% | 5.46 |
| Farrington-type | 2.73–5.52 | 1.4% | 5.46 |
| XGBoost (pooled) | 2.61–35.39 | 29.3% | 5.52 |
| XGBoost, recalibrated | 2.64–29.76 | 27.8% | 5.46 |

## Export-restriction implementations: what was in force when each measure took effect

The posture dated t is set on the data through t - 1 (the convention of the one-month detection rate); the table also gives the level dated t - 1.

64 restriction events. M2 triage level ≥ 2 in force at the implementation month: 61%; level ≥ 1: 91%; q-alarm in force: 0%; median months since the item first reached the alert level: 77.0 (over 53 events).

Poisson M0 level ≥ 2 at the implementation month: 69%.
XGBoost level ≥ 2 at the implementation month: 48%.
XGBoost (recalibrated) level ≥ 2 at the implementation month: 48%.
sigma-rule level ≥ 2 at the implementation month: 16%.
EWMA alarm at the implementation month: 11%.
CUSUM alarm at the implementation month: 12%.
Farrington-type alarm at the implementation month: 0%.

| Item | Group | Month | Measure | M2 p(3) | M2 level | M2 level the month before | months since first alert | q-alarm |
|---|---|---|---|---|---|---|---|---|
| 250410 | battery_materials | 202312 | 156:Export controls on graphite items | 0.036 | 0 | 0 |  | False |
| 261100 | refractory_minor_metals | 201706 | 643:Export tax (Resolution of the Government | 0.42 | 2 | 2 | 3 | False |
| 261590 | refractory_minor_metals | 201801 | 108:Fiscal tax on exports (Loi No.1/28 du 31 | 0.094 | 1 | 1 | 10 | False |
| 280450 | refractory_minor_metals | 202501 | 156:China: 2025 dual-use catalogue for expor | 0.823 | 3 | 3 | 94 | False |
| 280450 | refractory_minor_metals | 202502 | 156:Export controls on tungsten, tellurium,  | 0.855 | 3 | 3 | 95 | False |
| 280461 | semiconductor_chemicals | 202505 | 392:Japan: Government tightens the restricti | 0.499 | 2 | 3 | 98 | False |
| 280530 | rare_earths_magnets | 202201 | 156:China: 2022 export licensing goods catal | 0.236 | 2 | 2 | 58 | False |
| 280530 | rare_earths_magnets | 202504 | 156:Export controls on seven rare-earth elem | 0.82 | 3 | 3 | 97 | False |
| 280530 | rare_earths_magnets | 202510 | 156:China: Government announces export contr | 0.59 | 3 | 3 | 103 | False |
| 280530 | rare_earths_magnets | 202511 | 156:Expanded rare-earth export controls (Oct | 0.492 | 2 | 3 | 104 | False |
| 281111 | semiconductor_chemicals | 201901 | 156:China: 2019 catalogue of 'dual-use' good | 0.071 | 1 | 1 |  | False |
| 281111 | semiconductor_chemicals | 201907 | 392:Export licensing of HF / photoresists /  | 0.042 | 0 | 0 |  | False |
| 281290 | semiconductor_chemicals | 202505 | 392:Japan: Government tightens the restricti | 0.286 | 2 | 2 | 98 | False |
| 282200 | battery_materials | 201803 | 180:Other export measures (Loi n°18/001 du 0 | 0.072 | 1 | 1 | 12 | False |
| 282520 | battery_materials | 202507 | 156:China: Government revises catalogue cont | 0.285 | 2 | 2 | 74 | False |
| 282560 | refractory_minor_metals | 202308 | 156:Export controls on gallium and germanium | 0.252 | 2 | 2 | 77 | False |
| 282580 | refractory_minor_metals | 202409 | 156:Export controls on antimony and superhar | 0.157 | 1 | 1 | 90 | False |
| 282580 | refractory_minor_metals | 202501 | 156:China: 2025 export licensing goods catal | 0.205 | 2 | 1 | 94 | False |
| 282619 | semiconductor_chemicals | 202501 | 156:China: 2025 dual-use catalogue for expor | 0.124 | 1 | 1 | 16 | False |
| 282619 | semiconductor_chemicals | 202505 | 392:Japan: Government tightens the restricti | 0.078 | 1 | 1 | 20 | False |
| 283691 | battery_materials | 201809 | 32:Export tax (Decreto 793/2018) | 0.05 | 0 | 0 |  | False |
| 284180 | refractory_minor_metals | 202502 | 156:Export controls on tungsten, tellurium,  | 0.779 | 3 | 3 | 95 | False |
| 284690 | rare_earths_magnets | 202201 | 156:China: 2022 export licensing goods catal | 0.138 | 1 | 2 | 58 | False |
| 284690 | rare_earths_magnets | 202504 | 156:Export controls on seven rare-earth elem | 0.627 | 3 | 3 | 97 | False |
| 284690 | rare_earths_magnets | 202510 | 156:China: Government announces export contr | 0.37 | 2 | 2 | 103 | False |
| 284690 | rare_earths_magnets | 202511 | 156:Expanded rare-earth export controls (Oct | 0.355 | 2 | 2 | 104 | False |
| 284700 | semiconductor_chemicals | 202501 | 156:China: 2025 dual-use catalogue for expor | 0.778 | 3 | 3 | 94 | False |
| 285000 | semiconductor_chemicals | 202308 | 156:China: Government announces export contr | 0.495 | 2 | 2 | 77 | False |
| 285000 | semiconductor_chemicals | 202409 | 156:China: Government announces export contr;842:U | 0.213 | 2 | 1 | 90 | False |
| 370790 | semiconductor_chemicals | 201907 | 392:Export licensing of HF / photoresists /  | 0.346 | 2 | 2 | 28 | False |
| 380110 | battery_materials | 202312 | 156:Export controls on graphite items | 0.068 | 1 | 1 |  | False |
| 380110 | battery_materials | 202501 | 156:China: 2025 dual-use catalogue for expor | 0.083 | 1 | 1 |  | False |
| 381800 | semiconductor_chemicals | 202308 | 156:China: Government announces export contr | 0.071 | 1 | 1 |  | False |
| 381800 | semiconductor_chemicals | 202501 | 156:China: 2025 dual-use catalogue for expor | 0.056 | 0 | 0 |  | False |
| 381800 | semiconductor_chemicals | 202502 | 156:China: Government announces export contr | 0.039 | 0 | 0 |  | False |
| 381800 | semiconductor_chemicals | 202510 | 156:China: Ministry of Commerce issues expor | 0.087 | 1 | 0 |  | False |
| 720221 | ferroalloys | 201801 | 156:China: 2018 Tariff Implementation Plan r | 0.198 | 2 | 2 | 10 | False |
| 720221 | ferroalloys | 201901 | 156:China: 2019 list of licensed ferroalloy  | 0.271 | 2 | 2 | 22 | False |
| 720221 | ferroalloys | 202105 | 156:Export tax | 0.158 | 1 | 1 | 50 | False |
| 720221 | ferroalloys | 202201 | 156:China: 2022 Tariff Adjustment Plan relea | 0.115 | 1 | 1 | 58 | False |
| 720221 | ferroalloys | 202501 | 156:China: 2025 export licensing goods catal | 0.39 | 2 | 2 | 94 | False |
| 720249 | ferroalloys | 201801 | 156:China: 2018 list of licensed ferroalloy  | 0.563 | 3 | 3 | 10 | False |
| 720249 | ferroalloys | 201901 | 156:China: 2019 list of licensed ferroalloy  | 0.295 | 2 | 2 | 22 | False |
| 720249 | ferroalloys | 202105 | 156:China: Import/export tariffs on some ste | 0.215 | 1 | 1 | 50 | False |
| 720249 | ferroalloys | 202108 | 156:China: Ministry announces export tariffs | 0.248 | 2 | 2 | 53 | False |
| 720249 | ferroalloys | 202201 | 156:China: 2022 Tariff Adjustment Plan relea | 0.139 | 1 | 1 | 58 | False |
| 720260 | ferroalloys | 202111 | 360:Indonesia: Introduced export and import  | 0.604 | 3 | 3 | 56 | False |
| 720270 | ferroalloys | 201801 | 156:China: 2018 list of licensed ferroalloy  | 0.331 | 2 | 3 | 10 | False |
| 720270 | ferroalloys | 201901 | 156:China: 2019 list of licensed ferroalloy  | 0.923 | 3 | 2 | 22 | False |
| 720280 | ferroalloys | 201801 | 156:China: 2018 list of licensed ferroalloy  | 0.662 | 3 | 3 | 10 | False |
| 720280 | ferroalloys | 201901 | 156:China: 2019 list of licensed ferroalloy  | 0.533 | 2 | 2 | 22 | False |
| 720292 | ferroalloys | 201801 | 156:China: 2018 list of licensed ferroalloy  | 0.467 | 2 | 3 | 10 | False |
| 720292 | ferroalloys | 201901 | 156:China: 2019 list of licensed ferroalloy  | 0.422 | 2 | 2 | 22 | False |
| 810194 | refractory_minor_metals | 202502 | 156:Export controls on tungsten, tellurium,  | 0.268 | 2 | 2 | 92 | False |
| 810411 | refractory_minor_metals | 202501 | 156:China: 2025 dual-use catalogue for expor | 0.064 | 1 | 1 | 78 | False |
| 810520 | battery_materials | 202502 | 180:Cobalt export suspension | 0.583 | 3 | 3 | 95 | False |
| 811010 | refractory_minor_metals | 202409 | 156:Export controls on antimony and superhar | 0.805 | 3 | 3 | 90 | False |
| 811100 | battery_materials | 202306 | 360:Indonesia: Introduced export and import  | 0.039 | 0 | 0 |  | False |
| 811292 | refractory_minor_metals | 202201 | 156:China: 2022 export licensing goods catal | 0.634 | 3 | 3 | 58 | False |
| 811292 | refractory_minor_metals | 202308 | 156:Export controls on gallium and germanium | 0.494 | 2 | 3 | 77 | False |
| 811292 | refractory_minor_metals | 202502 | 156:Export controls on tungsten, tellurium,  | 0.504 | 2 | 2 | 95 | False |
| 850511 | rare_earths_magnets | 202312 | 156:China: Government releases revised catal | 0.081 | 1 | 1 | 81 | False |
| 850511 | rare_earths_magnets | 202504 | 156:Export controls on seven rare-earth elem | 0.124 | 1 | 1 | 97 | False |
| 850511 | rare_earths_magnets | 202511 | 156:Expanded rare-earth export controls (Oct | 0.072 | 1 | 1 | 104 | False |
