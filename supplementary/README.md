# supplementary/ — the supplementary material of the manuscript

`supplementary.tex` is the supplementary material of the manuscript: Supplementary Notes 1–7 and Tables S1–S16. `make_supp_short.py` writes it. The script takes the table bodies from the extended results in `../extended_results/`, whose scripts compute them from the archived result files. The notes are written in the script.

```bash
cd supplementary
python make_supp_short.py .                       # rewrites supplementary.tex; rerunning it changes nothing
xelatex supplementary.tex && xelatex supplementary.tex
```

The two lines under the heading of page 1 are set with the environment variables `SUPP_TITLE` and `SUPP_SUBTITLE`. The environment variable `SUPP_LONG` sets another folder for the extended results.

## Where each note and table comes from

The extended results keep their own numbers. For each note and table of the supplementary material, the table below gives the item of the extended results from which it is taken, the CSV form of the table in `../extended_results/tables/` and the script that writes it.

| Supplementary material | Extended results | CSV file | Written by |
|---|---|---|---|
| Note 1: sampler settings and convergence criteria | Note 1 | | |
| Note 2: baseline specifications | Note 2 | | |
| Note 3: severity submodel | Note 8 | | |
| Note 4: further results of the simulation study | Notes 4, 5 and 12; Table S16 | | |
| Note 5: expected cost of the alarm policies | Note 10 | | |
| Note 6: the backtest on public trade data | Note 14 | | |
| Note 7: the rare arm | Note 17 | | |
| Table S1: triage at level 1 | Table S1.1 | `S1_triage_level1.csv` | `make_supp.py` |
| Table S2: triage at level 3 | Table S1.3 | `S1_triage_level3.csv` | `make_supp.py` |
| Table S3: Study B by regime | Tables S2.1–S2.3, selected columns | `S2_table6_stable.csv`, `S2_table6_gradual.csv`, `S2_table6_abrupt.csv` | `make_supp.py` |
| Table S4: triage at level 2 by cell | Table S3.2 | `S3b_triage_by_cell.csv` | `make_supp.py` |
| Table S5: coverage in Study A by factor level | Table S4 | `S4_studyA_by_factor.csv` | `make_supp.py` |
| Table S6: alarm metrics in the robustness arms | Table S24, without the column on prediction sets | `S24_robustness_alarm_metrics.csv` | `gen_supp_part2.py` |
| Table S7: saving of M2 relative to each policy | Table S20 | `S20_savings_vs_M2.csv` | `gen_supp_part2.py` |
| Table S8: level switches, value of monitoring and break-even ratios | Table S21.1 | `S21_1_levels_breakeven.csv` | `gen_supp_part2.py` |
| Table S9: value of monitoring by method | Table S21.2 | `S21_2_value_of_monitoring_by_method.csv` | `gen_supp_part2.py` |
| Table S10: alarm budget and detection | Table S21.3 | `S21_3_alarm_budget_detection.csv` | `gen_supp_part2.py` |
| Table S11: the 65 items | Table S25.2 | `S25_2_items.csv` | `gen_supp_backtest.py` |
| Table S12: backtest, trade interruptions (E1) | Table S26.1 | `S26_1_metrics_N1.csv` | `gen_supp_backtest.py` |
| Table S13: backtest, export restrictions (E2) | Table S26.2 | `S26_2_metrics_N2.csv` | `gen_supp_backtest.py` |
| Table S14: sensitivity to the event definitions | Table S28 | `S28_sensitivity.csv` | `gen_supp_backtest.py` |
| Table S15: alarm performance in the rare arm | Table S29.1, without the columns on prediction sets | (text form: `../results_rare/tables_6_style.txt`) | `../rare_arm/gen_note17.py` |
| Table S16: expected cost in the rare arm | Table S29.2 | (text form: `../results_rare/cost_summary.txt`) | `../rare_arm/gen_note17.py` |

The scripts in the last column are in `../extended_results/` unless a path is given. The main README lists the commands that run them.
