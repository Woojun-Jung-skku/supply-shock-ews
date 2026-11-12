"""v1.4: fill the ⟨key⟩ placeholders of the §5.3 template in docs/results_text_draft.md from an analysis
folder and write docs/results_text_final.md (the template is kept). Every number is computed here.
usage: python tools/fill_text.py ANALYSIS_DIR [TEMPLATE] [OUT]"""
import os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arl as ARL

an = sys.argv[1]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tpl = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "docs", "results_text_draft.md")
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(root, "docs", "results_text_final.md")
d = pd.read_csv(os.path.join(an, "B_replications.csv"))
cell = ["alpha", "mu", "regime"]
v = {}
f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{100 * x:.0f}%"
sgn = lambda x: f"{x:+.3f}".replace("-", "−")
rng = lambda a: f"{min(a):.2f}–{max(a):.2f}" if f"{min(a):.2f}" != f"{max(a):.2f}" else f"{min(a):.2f}"

m = lambda c: d[c].mean()
v["aucpr_M2"], v["brier_M2"], v["ece_M2"] = f2(m("M2_aucpr")), f3(m("M2_brier")), f3(m("M2_ece"))
dd = lambda k, s="aucpr": (d[f"{k}_{s}"] - d[f"M2_{s}"]).mean()
v["d_aucpr_NBc"] = f3(abs(dd("NBcomplete")))
v["d_aucpr_M0"], v["d_aucpr_NBn"] = sgn(dd("M0")), sgn(dd("NBnone"))
v["ece_M0"], v["ece_NBn"] = f3(m("M0_ece")), f3(m("NBnone_ece"))
v["aucpr_xgb"], v["aucpr_lstm"], v["aucpr_tft"] = f2(m("xgb_aucpr")), f2(m("lstm_aucpr")), f2(m("tft_aucpr"))
v["ece_ml"] = f"{min(m(k + '_ece') for k in ('xgb', 'lstm', 'tft')):.3f}–{max(m(k + '_ece') for k in ('xgb', 'lstm', 'tft')):.3f}"
gain = max((d[f"{k}_aucpr"] - d["M2_aucpr"]).groupby([d[c] for c in cell]).mean().max() for k in ("M0", "NBnone", "NBcomplete"))
v["max_model_gain"] = f3(max(gain, 0))

v["dr1_M2"], v["dr3_M2"] = pct(m("M2_tri2_dr1")), pct(m("M2_tri2_dr3"))
v["lead_M2"] = f"{d['M2_tri2_lead'].median():.0f}"
v["fa_M2"] = f2(m("M2_tri2_fa_stable"))
for k, c in (("sig", "sigma_2"), ("ewma", "ewma"), ("cusum", "cusum")):
    v[f"dr1_{k}"], v[f"fa_{k}"] = pct(m(f"{c}_dr1")), f2(m(f"{c}_fa_stable"))
# cells where an ML baseline reaches M2's level-2 detection with fewer false alarms
t = d.groupby(cell).mean(numeric_only=True)
win = {}
for k, name in (("lstm", "the LSTM"), ("xgb", "XGBoost")):
    w = t[(t[f"{k}_tri2_dr1"] >= t["M2_tri2_dr1"] - 0.005) & (t[f"{k}_tri2_fa_stable"] < t["M2_tri2_fa_stable"] - 0.01)]
    n02 = int((w.index.get_level_values("mu") == 0.2).sum())
    win[k] = (name, len(w), n02)
parts = [f"{name} in {n} of the 12 cells ({'all six' if n02 == 6 else f'{n02} of the six'} with μ̄ = 0.2)"
         for name, n, n02 in win.values()]
ex = t.loc[(2.0, 0.2, "stable")]
v["ml_win_cells"] = (" and ".join(parts) + f"; at α′ = 2, μ̄ = 0.2, stable, for instance, the LSTM flags "
                     f"{pct(ex['lstm_tri2_dr1'])} at {f2(ex['lstm_tri2_fa_stable'])} false alarms against "
                     f"{pct(ex['M2_tri2_dr1'])} at {f2(ex['M2_tri2_fa_stable'])} for M2")
rare = t[(t.index.get_level_values("mu") == 0.05) & (t.index.get_level_values("regime") != "stable")]
prng = lambda a: f"{100 * min(a):.0f}–{100 * max(a):.0f}%"
v["dr1_M2_rare"], v["dr1_xgb_rare"] = prng(rare["M2_tri2_dr1"]), prng(rare["xgb_tri2_dr1"])

chan = dict(M2=("M2_q_arl0", "M2_q_stable_alarm_units", "q"), M0=("M0_q_arl0", "M0_q_stable_alarm_units", "q"),
            sig=("sigma_arl0", "sigma_stable_alarm_units", "item"), ewma=("ewma_arl0", "ewma_stable_alarm_units", "item"),
            cusum=("cusum_arl0", "cusum_stable_alarm_units", "item"))
for k, (a0, sh, u) in chan.items():
    pr = ARL.pooled(d, a0, sh, ARL.UNITS[u])
    v[f"arl0_{k}"] = f"{pr['est']:.0f}" if pr["est"] >= 20 else f"{pr['est']:.1f}"
    if k == "M2":
        v["arl0_M2_ci"] = f"{pr['lo']:.0f}–{pr['hi']:.0f}"
        v["noalarm_M2"] = pct(pr["no_alarm_share"])
a1 = dict(M2="M2_q_arl1", M0="M0_q_arl1", sig="sigma_arl1", cusum="cusum_arl1")
for k, c in a1.items():
    v[f"arl1_abr_{k}"] = f"{d.loc[d.regime == 'abrupt', c].mean():.1f}"
    v[f"arl1_gr_{k}"] = f"{d.loc[d.regime == 'gradual', c].mean():.1f}"

v["conf_M2"], v["conf_M2_pre"], v["conf_M2_post"] = f3(m("M2_conf_cover")), f3(m("M2_conf_cover_pre")), f3(m("M2_conf_cover_post"))
v["size_M2"], v["size_M0"], v["size_NBn"] = f2(m("M2_conf_size")), f2(m("M0_conf_size")), f2(m("NBnone_conf_size"))
v["conf_M0"], v["conf_NBn"] = f3(m("M0_conf_cover")), f3(m("NBnone_conf_cover"))

nf = d["M2_fit_failed_origins"].fillna(0)
v["n_M2_fits"] = f"{14 * len(d):,}"
v["fail_M2"] = f"{100 * nf.sum() / (14 * len(d)):.1f}%"
fl = nf > 0
v["n_fail_reps"] = str(int(fl.sum()))
ch = {s: max(abs((d[f"{k}_{s}"] - d[f"M2_{s}"]).mean() - (d.loc[~fl, f"{k}_{s}"] - d.loc[~fl, f"M2_{s}"]).mean())
             for k in ("M0", "NBnone", "NBcomplete", "xgb", "lstm", "tft")) for s in ("aucpr", "ece")}
v["d_excl_aucpr"], v["d_excl_ece"] = f3(ch["aucpr"]), f3(ch["ece"])

txt = open(tpl).read()
keys = set(re.findall(r"⟨([^⟩]+)⟩", txt))
missing = keys - set(v)
for k, val in v.items():
    txt = txt.replace(f"⟨{k}⟩", val)
open(out, "w").write(txt.replace("Numbers marked [S] are final", "Numbers marked [S] are final (filled by tools/fill_text.py)"))
print(f"replications {len(d)}; filled {len(keys & set(v))} keys; missing {sorted(missing)}")
for k in sorted(v):
    print(f"  {k}: {v[k]}")
