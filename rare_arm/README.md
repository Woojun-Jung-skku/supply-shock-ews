# Rare arm (Sections 5.1, 5.3 and 5.6; Supplementary Note 17)

Three cells at a base rate of 0.02 shocks per item-month (0.24 per item-year), I = 100 items, T = 120 months,
alpha' = 2, one per regime (stable, gradual, abrupt), five replications each. The cells run through the Study B
pipeline (`code/study_b.py`) with the `t4_min` design constants; only the cell definitions are new.

| File | Contents |
|---|---|
| `../run_rare_full.py`, `../run_rare_full.sh` | Driver and launcher (repository root, next to `code/`). Resumable; `--no-nets` skips LSTM and TFT. |
| `../results_rare/B/B-a2-mu0.02-<regime>-I100__all.jsonl` | One record per replication in the Study B record format. The `lstm` and `tft` keys hold constant-prediction placeholders: the recurrent networks were not run in this arm. |
| `rare_cost.py` | Expected cost with the definitions of Section 5.6 (Table 8, Tables S21.1–S21.2): for each policy the lower envelope of its level-wise mean-cost lines, the band of cost ratios in which it beats the better of never and always acting at the subset mean E, its largest saving, the range with a saving above 10%, the level switches of M2 and the paired comparisons. Output: `../results_rare/cost_summary.txt` (Tables S29.2 and S29.4). |
| `check_cost_definition.py` | Applies the functions of `rare_cost.py` to the archived Study B records and checks that they reproduce Table 8 (eight policies) and the 'All cells' row of Table S21.1; exit code 1 on a mismatch. |
| `tables_full.py` | The Table 6 metrics per cell and pooled. Output: `../results_rare/tables_6_style.txt` (Table S29.1). |
| `analyze_full.py` | Replication-level values and the paired comparisons of AUC-PR, ECE and Brier score. Output: `../results_rare/per_rep_detail.txt` (Table S29.3). |
| `gen_note17.py` | Writes `../supplementary/supplementary_note17_rare_arm.tex` (the body of Supplementary Note 17 with Tables S29.1–S29.4, which `supplementary_part2.tex` inputs after Note 16) from the result files; every number in the note is computed, none is typed. |

## Reproducing

From the repository root, in the environment of `code/requirements.txt`:

```bash
python run_rare_full.py --out results_rare_new --reps 5 --no-nets          # about 6 CPU-hours per regime on one core
python rare_arm/rare_cost.py    . results_rare > cost_summary.txt          # archived as results_rare/cost_summary.txt
python rare_arm/check_cost_definition.py . results_rare             # same code on Study B: must print "reproduced"
python rare_arm/tables_full.py  . results_rare > tables_6_style.txt        # archived as results_rare/tables_6_style.txt
python rare_arm/analyze_full.py . results_rare > per_rep_detail.txt        # archived as results_rare/per_rep_detail.txt
python rare_arm/gen_note17.py   . results_rare supplementary/supplementary_note17_rare_arm.tex
cd supplementary && xelatex supplementary.tex && xelatex supplementary.tex   # the note is part of the supplementary material
```

## Two conventions that matter

* **E** is the number of shock item-months per item-year over the decision months (an item-month with several
  shocks counts once): 0.34, 0.53 and 0.74 in the three cells. The corresponding numbers of shocks are 0.36, 0.57
  and 0.82 per item-year.
* **Cost of a policy at ratio r** is the mean over replications at the level that is cheapest on average
  (the lower envelope of the level-wise mean-cost lines), as in Table 8; the level is not chosen replication by
  replication. The trivial policies are priced at the subset mean E, min(rE, 12).

Seeds are `dgp.seed_of("B", cell, rep)`, so the records do not depend on the machine: the archived run
(Python 3.13, JAX 0.12.0.1, NumPyro 0.22.0, XGBoost 3.4.1, two CPU cores, resumed in two-hour chunks) reproduced
two replications per cell of an earlier run on another machine to every recorded digit.
