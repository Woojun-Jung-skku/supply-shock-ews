# Early warning of supply shocks with a hierarchical gamma–Poisson model: code, results and supplementary material

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24151137.svg)](https://doi.org/10.5281/zenodo.24151137)

This repository holds the code, the results and the supplementary material of a manuscript on a hierarchical gamma–Poisson early-warning model of supply shocks with cost-based alarms (Sungjin Jeon and Woojun Jung, equal contribution; under review, 2026).

Everything the manuscript reports in Section 5, its figures and the supplementary material can be regenerated from this repository. The archived result files are the ones the manuscript uses.

## Where each part of the manuscript is

| Manuscript | Repository |
|---|---|
| Sections 5.1–5.4, Tables 5 and 6, Figs. 4 and 5: the simulation studies (parameter recovery, alarm performance, robustness) | `code/`, `results/A`, `results/B`, `results/R`, `results/followup/`, `results/analysis/` |
| Sections 5.1, 5.3 and 5.6, Supplementary Note 17: the rare arm of the simulation | `run_rare_full.py`, `rare_arm/`, `results_rare/` |
| Section 5.5, Table 7, Fig. 6, Supplementary Note 14: the backtest on public trade data | `backtest/`, `data/`, `results/backtest/` |
| Section 5.6, Table 8, Fig. 7, Supplementary Note 10: the expected cost of the alarm policies | `code/tools/` (`extract_alarm_share.py`, `validate_regen.py`, `cost_curves.py`, `cost_tail.py`), `results/analysis/cost/` |
| Section 4.4, Fig. 2: run length of a sigma-rule under over-dispersion | `manuscript/arl0_sigma_limits.py` |
| Figs. 1–7 and S1–S4 | `figures/` |
| Supplementary material (Supplementary Notes 1–17, Tables S1–S29, Figs. S1–S4) | `supplementary/` |

## Layout

| Folder | Contents |
|---|---|
| `code/` | Simulator (`dgp.py`), hierarchical negative-binomial model (`models.py`, NumPyro/JAX NUTS in `nuts_gpu.py`), particle propagation (`pfilter.py`), alarm layer (`alarm.py`), baselines (Poisson-calibrated sigma-rule, EWMA and CUSUM charts in `charts.py`; XGBoost, LSTM and TFT in `ml.py`; the negative-binomial-designed charts and the Farrington-type threshold in `followup/b1_nbcharts.py` and `followup/a3_farrington.py`), evaluation (`evaluate.py`, `metrics.py`), runners for Study A, Study B and the robustness study (`run_mac.py`, `run.py`), the follow-up arms (`followup/`), the analysis and cost tools (`tools/`) and tests (`tests/`). Seeds and all design constants are in `config.py`. |
| `results/A`, `results/B`, `results/R` | One JSON-lines file per cell and model variant, one record per replication (Study A: 738 fits; Study B: 65 replications, each refitted at 14 origins; robustness study: 75 replications, the Hawkes arms included). |
| `results/followup/` | Follow-up runs: prior sensitivity (A1), Farrington-type threshold (A3), item-effect scale (A7), negative-binomial-designed control charts (B1), dispersion that differs across groups (B2), heterogeneity that is not gamma (B3) and the severity submodel (B5) in `out/`; indicators without information, β = 0 (A2, `R/`); 24-month refit schedule (A5, `B24/` and `A5_M2_24m_replications.csv`); reparameterised horseshoe (A6, `A6/`); `followup_summary.txt` summarises A2, A5 and A6. |
| `results/analysis/` | Replication-level CSVs and the tables of Sections 5.2–5.4 as produced by `code/tools/analysis_all.sh`. |
| `results/analysis/cost/` | Expected-cost evaluation: cost tables, savings and sign tests, level switches, break-even ratios, curves on the grid of cost ratios, the sensitivity runs and the tail-risk reading for cost models A and B, with the replication-level inputs (`cost_inputs_by_replication.csv`) and two digests. |
| `run_rare_full.py`, `run_rare_full.sh`, `rare_arm/`, `results_rare/` | The rare arm: driver, analysis scripts and the replication-level results of three cells at 0.02 shocks per item-month (`rare_arm/README.md`). |
| `backtest/` | The backtest: extraction of UN Comtrade records, the restriction table, the panel, the rolling-origin backtest, its figure and report (`backtest/README.md`). |
| `data/` | Extracts of the OECD Inventory of Export Restrictions on Industrial Raw Materials and of the Global Trade Alert database for the items of the panel (sources and licences in `data/README.md`). UN Comtrade records are not redistributed. |
| `results/backtest/` | Result files of the main run (`main/`: metrics by event definition, cost bands, the export-restriction detail, `paths.npz`, `fits/diagnostics.csv`, `table7.md`, `key_numbers.json`, `report.md`), of the evaluation on the strict restriction list (`strict/`) and of the two runs with alternative interruption definitions (`top/`, `total/`). The MCMC fit files are not archived; the script regenerates them. |
| `figures/` | Figure scripts, the shared style module `figstyle.py`, the figures of the manuscript in `out_final/` (PDF, and PNG at 600 dpi) and `out_final_log.txt`, the numbers printed while drawing Figs. 4 and 5. |
| `supplementary/` | LaTeX sources of the supplementary material, the scripts that write them and the CSV form of every table in `tables/`. |
| `manuscript/` | `arl0_sigma_limits.py`: the run-length numbers of Section 4.4 and Fig. 2. |

## Figures

| Manuscript | File in `figures/out_final/` | Drawn by |
|---|---|---|
| Fig. 1 | `Fig1_sri_curves` | `figures/make_figs_color.py` (`fig1`) |
| Fig. 2 | `Fig2_sigma_rule` | `figures/make_figs_color.py` (`fig2`) |
| Fig. 3 | `Fig3_inference_stack` | `figures/schem19.py` (`fig3`) |
| Fig. 4 | `Fig4_parameter_recovery` | `figures/make_figs_color.py` (`fig4`) |
| Fig. 5 | `Fig5_alarm_performance` | `figures/make_figs_color.py` (`fig5`) |
| Fig. 6 | `Fig6_backtest` | `backtest/fig_backtest.py` |
| Fig. 7 | `Fig7_cost_curves` | `code/tools/cost_curves.py` |
| Fig. S1 | `FigS1_lineages` | `figures/schem19.py` (`figS1`) |
| Fig. S2 | `FigS2_industry_map` | `figures/schem19.py` (`figS2`) |
| Fig. S3 | `FigS3_hawkes_dispersion` | `figures/make_figs_color.py` (`figS3`) |
| Fig. S4 | `FigS4_three_layer_correspondence` | `figures/schem19.py` (`figS4`) |

## Reproducing the results

Python 3.12 or 3.13; the figures use the Liberation Sans font (Arial metrics) and the supplementary material is typeset with XeLaTeX (TeX Gyre Pagella).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt          # numpy, scipy, pandas, jax, numpyro, arviz, scikit-learn, xgboost, matplotlib
```

The commands below write to new folders (`results_new/`, `results_rare_new/`, `figures/out/`, `out/`) unless the text says that they rewrite an archived file.

**Simulation studies** (Sections 5.1–5.4). The archived results were run with the `t4_min` profile (`design` field `v1.4-t4_min` in every record). The runner appends to existing result files, so the same command resumes an interrupted run.

```bash
cd code
python run_mac.py --out ../results_new --profile t4_min --workers 8        # Studies B, A and robustness
python followup/a1_priors.py ../results_new/followup ../results/A/A-a2-mu0.2-gradual-I30-T120__hier.jsonl   # A1 (Table S6)
python followup/a7_tauC.py ../results_new/followup                          # A7 (Table S7)
python followup/a3_farrington.py ../results/analysis ../results_new/followup            # A3 (Table S9)
python followup/mac_arms.py a2 --out ../results_new/followup_mac           # also a5, a6; then: mac_arms.py summary --out ...
python followup/b1_nbcharts.py ../results/analysis ../results_new/followup             # B1 (Table S14; POOL=panel for the panel-pooled rows)
python followup/b1_diag.py                                                 # Table S14b; its printed output is results/followup/out/B1_nbcharts_diag.csv
python followup/b2_alpha_groups.py ../results_new/followup                  # B2 (Table S13)
python followup/b3_lognormal.py ../results_new/followup/B3                  # B3 (Table S16)
python followup/b5_severity.py ../results_new/followup/B5                   # B5 (Table S15)
```

**Tables of Sections 5.2–5.4.** From the archived results, this command reproduces every file in `results/analysis/` byte for byte, except the folder `cost/`:

```bash
cd code
bash tools/analysis_all.sh ../results_new/analysis ../results               # ANALYSIS_DIR first, then the result folder(s)
```

**Expected cost** (Section 5.6; Table 8, Fig. 7, Supplementary Note 10). The evaluation uses the archived Study B results, so no model is refitted. The four commands reproduce the files of `results/analysis/cost/`. The CSV files of the first three commands are reproduced byte for byte, and those of `cost_tail.py` up to the last digits of the floating-point output.

```bash
cd code && mkdir -p ../results_new/cost
python tools/extract_alarm_share.py ../results/B ../results/analysis ../results_new/cost/B_jsonl_extract.csv   # alarm shares, onsets and false onsets per method and level
python tools/validate_regen.py . ../results/analysis                                                   # the panels regenerated from the seeds reproduce the archived metrics (65/65)
python tools/cost_curves.py . ../results/analysis ../results_new/cost/B_jsonl_extract.csv ../results_new/cost ../figures   # cost tables, curves, savings, break-even ratios, sensitivity; Fig. 7 to figures/out/
python tools/cost_tail.py ../results_new/cost/cost_inputs_by_replication.csv ../results_new/cost        # 90th percentile and CVaR of cost (Table S21.4)
```

**Rare arm** (Sections 5.1, 5.3 and 5.6; Supplementary Note 17). The first command reruns the arm, which takes about 6 CPU-hours per regime. The other commands recompute the archived summaries from the archived records. See `rare_arm/README.md` for details.

```bash
python run_rare_full.py --out results_rare_new --reps 5 --no-nets
python rare_arm/check_cost_definition.py . results_rare      # the cost code of the rare arm reproduces Table 8 on Study B: prints "reproduced"
python rare_arm/rare_cost.py . results_rare                  # prints Tables S29.2 and S29.4 (archived as results_rare/cost_summary.txt)
python rare_arm/tables_full.py . results_rare                # prints Table S29.1 (archived as results_rare/tables_6_style.txt)
python rare_arm/analyze_full.py . results_rare               # prints Table S29.3 (archived as results_rare/per_rep_detail.txt)
```

**Backtest on public trade data** (Section 5.5). `backtest/README.md` gives the commands. The restriction table and the panels can be rebuilt exactly from the archived extracts and the UN Comtrade records, for which a free API key is needed. The main run of the backtest takes about two hours on two cores.

**Figures.** Figs. 1–4 and S1–S4 were rendered with Matplotlib 3.10.9, and Fig. 5 with Matplotlib 3.11.2. With these versions, the first two scripts below reproduce the archived PNG files pixel for pixel. With other versions, the scripts draw the same figures but render the text slightly differently. The scripts of Figs. 6 and 7 also redraw the archived figures with small differences in layout.

```bash
cd figures
python schem19.py out                 # Fig. 3 and Figs. S1, S2, S4
python make_figs_color.py out         # Figs. 1, 2, 4, 5 and S3 (Figs. 4 and 5 read results/analysis and results/followup/out)
cd ..
python backtest/fig_backtest.py results/backtest/main backtest/panel_410.csv figures/out/Fig6_backtest      # Fig. 6
```

**Supplementary material.** The five scripts below rewrite the LaTeX sources and the CSV tables in `supplementary/` from the archived results. The sources in the repository are the output of these scripts, so rerunning them changes nothing.

```bash
cd supplementary
python make_supp.py .                                                       # Notes 1–6 are supplementary_notes.tex; Tables S1–S12 and supplementary.tex
python make_supp_extra.py ../results/followup/out ../results/analysis       # Tables S13–S16 (supplementary_extra.tex)
python gen_supp_part2.py ../results/analysis/cost . ../figures/out_final ../results/analysis   # Notes 7–13, 15, 16; Tables S17–S24; Figs. S1–S4 (supplementary_part2.tex)
cd ..
python supplementary/gen_supp_backtest.py results/backtest/main backtest/panel_410.csv backtest/restrictions.csv backtest/items_hs6.csv supplementary \
       --strict-out results/backtest/strict --top-out results/backtest/top --total-out results/backtest/total --dropped backtest/panel_410_dropped.csv   # Note 14, Tables S25–S28
python rare_arm/gen_note17.py . results_rare supplementary/supplementary_note17_rare_arm.tex                # Note 17, Tables S29.1–S29.4
cd supplementary && xelatex supplementary.tex && xelatex supplementary.tex
```

`make_supp.py` reads the archived results by default (the environment variables REPO, ANALYSIS, RESULTS, FOLLOWUP, FOLLOWUP_MAC and CODE change the paths). The two lines under the heading of page 1 are set with the environment variables `SUPP_TITLE` and `SUPP_SUBTITLE`.

**Section 4.4.** From the repository root: `python manuscript/arl0_sigma_limits.py`

## Names and numbers inside `code/`

The simulation code keeps its working names, so some names and numbers differ from the manuscript:

- `tools/table5b.py` writes Table 6 and Tables S1–S2 (`results/analysis/table5b_*`).
- `tools/fig6_final.py` and `tools/fig7_final.py` draw monochrome drafts of Figs. 4 and 5. `analysis_all.sh` runs them because they print the numbers archived in `results/analysis/fig6_numbers.txt` and `fig7_numbers.txt`.
- Labels such as `v1.4` are versions of the simulation design. They appear in comments and in the `design` field of the records.
- Comments in the simulation code cite sections, equations, tables and figures by the numbers of an earlier draft. Where these numbers differ from the manuscript, use the tables above and the headers of the scripts outside `code/`, which follow the manuscript.

## Citation

See `CITATION.cff`. Please cite the paper once it is published. Until then, cite this repository by its Zenodo DOI, [10.5281/zenodo.24151137](https://doi.org/10.5281/zenodo.24151137). This DOI covers every version of the repository and resolves to the latest one.

## Licence

Code: MIT (see `LICENSE`). Simulation results, tables and figures: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

Data from third parties keep their own terms:

- The extract of the Global Trade Alert database in `data/` is licensed CC BY-NC 4.0 (Global Trade Alert).
- The extract of the OECD Inventory of Export Restrictions on Industrial Raw Materials is subject to the OECD terms and conditions.
- UN Comtrade records are not redistributed. `backtest/pull_comtrade_multi.py` retrieves them with the user's own API key.
