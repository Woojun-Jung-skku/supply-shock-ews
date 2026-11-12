# backtest/ — the backtest on public trade data (Section 5.5; Supplementary Note 14)

Rolling-origin backtest of the alarm and its baselines on the monthly imports of the Republic of Korea (UN Comtrade reporter 410) for 64 critical-material items in five groups, 2014–2025, against two event series: trade interruptions (E1, `N1` in the files) and export-restriction implementations (E2, `N2`); `N3` is their union.

| File | Role |
|---|---|
| `items_hs6.csv` | The 65 items (HS six-digit code, name, group, note). One item is dropped by the data screen (`panel_410_dropped.csv`), so the panel has 64. |
| `pull_comtrade_multi.py` | Extraction of monthly imports by partner from the UN Comtrade API (resumable; stops by itself at the daily quota). Writes `data_bt/raw_<reporter>.csv`, which is not redistributed. |
| `restrictions_seed.csv` | Hand-collected export-restriction measures (announcement and implementation dates, HS scope, sources). |
| `events_restrictions.py`, `gta_map.json`, `gta_map_strict.json`, `oecd_map_qdd.json` | Merge of the seed list, the Global Trade Alert extract and the OECD inventory extract (`../data/`) into one table of measures; the JSON files hold the column mappings and filters (`_strict`: new controls only, without annual renewals of catalogues and quotas). |
| `restrictions.csv`, `restrictions_strict.csv` | The merged tables (307 and 119 measures). |
| `build_panel_multi.py` | The monthly panel: events E1 (`--e1 share --share-min 0.25` in the main run; `top` and `total` are the sensitivity definitions), E2 and its announcement months, and the four indicators. |
| `panel_410.csv`, `panel_410_strict.csv`, `panel_410_top.csv`, `panel_410_total.csv` | The panel of the main run, the panel with the strict restriction list and the two panels with alternative interruption definitions. |
| `gp_ews.py` | Self-contained implementation of the model (fit, particle propagation, two-channel alarm) used by the backtest. |
| `backtest_multi.py`, `mcmc_diag.py` | The backtest: M2 and the Poisson model refitted at 18 origins and propagated month by month, the sigma-rule, EWMA, CUSUM, the Farrington-type threshold and XGBoost (plain and recalibrated) at the same decision months; metrics, costs and the restriction detail for each event series; convergence check of every fit. |
| `report_backtest.py` | A markdown summary of an output folder (`../results/backtest/main/report.md`). |
| `fig_backtest.py` | Fig. 6. |
| `make_synthetic_panel.py`, `test_report_synthetic.md` | A synthetic panel for testing the pipeline without trade data, and the report of such a run (the numbers have no meaning). |

Supplementary Note 14 and Tables S25–S28 are written by `../supplementary/gen_supp_backtest.py`, which also writes the main-text Table 7 (`../results/backtest/main/table7.md`) and the numbers quoted in the text (`key_numbers.json`).

## Reproducing

From the repository root, in the environment of `code/requirements.txt`. The commands write to `out/`, so the archived files are not overwritten.

**Quick test of the pipeline** (synthetic panel, short MCMC; about two minutes):

```bash
mkdir -p out
python backtest/make_synthetic_panel.py out/synth.csv
python backtest/backtest_multi.py out/synth.csv out/synth --fast --boot 30 --refit 12
python backtest/report_backtest.py out/synth
```

**Restriction tables** (from the archived extracts; identical to the archived tables):

```bash
python backtest/events_restrictions.py out/restrictions.csv --seed backtest/restrictions_seed.csv --gta data/gta_critical_minerals.csv --gta-map backtest/gta_map.json \
       --oecd data/oecd_export.csv --oecd-map backtest/oecd_map_qdd.json --items backtest/items_hs6.csv
python backtest/events_restrictions.py out/restrictions_strict.csv --seed backtest/restrictions_seed.csv --gta data/gta_critical_minerals.csv --gta-map backtest/gta_map_strict.json \
       --items backtest/items_hs6.csv
```

**Trade records and panels.** A free UN Comtrade subscription key is needed (https://comtradeplus.un.org, My Comtrade, API Portal; 500 calls a day). The extraction takes about 80 calls; rerun the same command if it stops. From the records extracted on 26 October 2026 the four `build_panel_multi.py` commands rebuild the archived panels exactly; later revisions of the Comtrade data can change individual months.

```bash
export COMTRADE_KEY=...        # or paste the key when the script asks for it
python backtest/pull_comtrade_multi.py backtest/items_hs6.csv data_bt --reporters 410 --years 2010-2025 --budget 7200
python backtest/build_panel_multi.py data_bt/raw_410.csv backtest/items_hs6.csv backtest/restrictions.csv out/panel_410.csv --start 2014-01 --end 2025-12 --e1 share --share-min 0.25
python backtest/build_panel_multi.py data_bt/raw_410.csv backtest/items_hs6.csv backtest/restrictions_strict.csv out/panel_410_strict.csv --start 2014-01 --end 2025-12 --e1 share --share-min 0.25
python backtest/build_panel_multi.py data_bt/raw_410.csv backtest/items_hs6.csv backtest/restrictions.csv out/panel_410_top.csv --start 2014-01 --end 2025-12 --e1 top
python backtest/build_panel_multi.py data_bt/raw_410.csv backtest/items_hs6.csv backtest/restrictions.csv out/panel_410_total.csv --start 2014-01 --end 2025-12 --e1 total
```

**The runs** (from the archived panels). The main run has 18 origins and 200 bootstrap paths and takes about two hours on two cores and about 3 GB of memory; run the four one after another. The evaluation on the strict list reuses the fits of the main run; the two sensitivity runs are full runs.

```bash
python backtest/backtest_multi.py backtest/panel_410.csv out/backtest/main --boot 200 --refit 6
mkdir -p out/backtest/strict && cp -r out/backtest/main/fits out/backtest/strict/
python backtest/backtest_multi.py backtest/panel_410_strict.csv out/backtest/strict --boot 200 --refit 6
python backtest/backtest_multi.py backtest/panel_410_top.csv out/backtest/top --boot 200 --refit 6
python backtest/backtest_multi.py backtest/panel_410_total.csv out/backtest/total --boot 200 --refit 6
```

The model is refitted by MCMC with fixed seeds, and the draws depend on the JAX and NumPyro versions, so a rerun reproduces the archived results up to Monte Carlo error. `gp_ews.fit` starts its chains near the prior medians: NumPyro's default uniform initialisation can start a chain where the rate is clipped, where the gradient vanishes and the chain stays. `backtest_multi.py` computes the rank-normalised split-R̂ and the bulk ESS of every fit (`fits/diagnostics.csv`) and repeats once, with 1,000 warm-up iterations and 1,000 draws, any fit with R̂ ≥ 1.05 or ESS < 100 on a stored quantity (marked by a `fit_<origin>_<variant>.npz.refit` file); the repeated fit is used whether or not it passes.

**Report, figure and supplementary tables** (from the archived result files):

```bash
mkdir -p out figures/out
python backtest/report_backtest.py results/backtest/main out/report.md
python backtest/fig_backtest.py results/backtest/main backtest/panel_410.csv figures/out/Fig6_backtest
python supplementary/gen_supp_backtest.py results/backtest/main backtest/panel_410.csv backtest/restrictions.csv backtest/items_hs6.csv supplementary \
       --strict-out results/backtest/strict --top-out results/backtest/top --total-out results/backtest/total --dropped backtest/panel_410_dropped.csv
```

## Conventions

- A posture or alarm dated month t is set on the data through month t − 1. The event label scored against the three-month probability dated t is "at least one event in months t to t + 2".
- Rates per item-year are taken over the evaluation grid of 64 items × 107 decision months (February 2017 to December 2025), 570.7 item-years. One item, bismuth (HS 810600), has no Comtrade record under this code after December 2021 and no panel row after December 2022, so the panel file has 6,812 item-months in the decision window against 6,848 on the grid.
- `events_N2_detail.csv` gives, for each export-restriction implementation in the decision window, the posture in force in the implementation month (`*_at`) and the month before (`*_tm1`), and the months since the item first reached the alert level.

## Sources

- UN Comtrade (https://comtradeplus.un.org): monthly imports by partner; the records are not redistributed (UN Comtrade terms of use).
- OECD Inventory of Export Restrictions on Industrial Raw Materials and Global Trade Alert: see `../data/README.md` for the extracts and their licences.
