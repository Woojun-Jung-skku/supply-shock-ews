"""make_synthetic_panel.py — a synthetic panel in the build_panel_multi.py format, for testing the backtest pipeline
end to end without trade data. Counts come from gp_ews.simulate (the Table 4 data-generating process, abrupt regime
at T/2); the four trade indicators are AR(1) series built so that hhi12 and top_share12 carry the simulated indicator
signal; "restriction events" (N2) are placed in the regime-change month and six months later for the items of one
group, so that events_N2_detail.csv has something to show.

usage: python make_synthetic_panel.py OUT_CSV [--I 12] [--T 84] [--G 3] [--mu 0.2] [--seed 1] [--gp-ews DIR]
"""
import argparse, csv, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--I", type=int, default=12); ap.add_argument("--T", type=int, default=84); ap.add_argument("--G", type=int, default=3)
ap.add_argument("--mu", type=float, default=0.2); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--gp-ews", default=os.environ.get("GP_EWS", HERE))
a = ap.parse_args()
sys.path.insert(0, a.gp_ews)
import gp_ews as G

s = G.simulate(I=a.I, T=a.T, G=a.G, mu_bar=a.mu, alpha=2.0, regime="abrupt", K=3, seed=a.seed, miss=0.0)
rng = np.random.default_rng(a.seed + 100)
N, X, Z, group, t0 = s["N"], s["X_true"], s["Z_true"], s["group"], s["t0"]
I, T = N.shape
names = {0: "battery_materials", 1: "semiconductor_chemicals", 2: "refractory_minor_metals", 3: "rare_earths_magnets", 4: "ferroalloys"}
periods = []
y, m = 2015, 1
for _ in range(T):
    periods.append(f"{y}{m:02d}"); m += 1
    if m == 13:
        y += 1; m = 1
hhi = 0.3 + 0.1 * X[:, :, 0]; top = np.clip(0.5 + 0.1 * X[:, :, 1], 0.05, 0.95)      # indicators carrying the signal
qty = np.exp(10 + 0.3 * Z[:, 0][:, None] + 0.2 * rng.standard_normal((I, T)))
uv = np.exp(2 + 0.2 * Z[:, 1][:, None] + 0.05 * np.cumsum(rng.standard_normal((I, T)), 1))
N2 = np.zeros((I, T), int); N2a = np.zeros((I, T), int)
for i in np.flatnonzero(group == 0):
    N2[i, t0] = 1; N2a[i, t0 - 2] = 1
    if t0 + 6 < T:
        N2[i, t0 + 6] = 1
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cmd", "group", "period", "total_qty_kg", "total_value_usd", "N1", "events1", "N2", "events2", "N2a", "events2a", "N3", "hhi12", "top_partner", "top_share12", "n_partners12", "unit_value"])
    for i in range(I):
        for t in range(T):
            w.writerow([f"{900000 + i}", names[group[i] % 5], periods[t], round(qty[i, t], 1), round(qty[i, t] * uv[i, t]), int(N[i, t]),
                        ";".join(["156"] * int(N[i, t])), int(N2[i, t]), "156:synthetic restriction" if N2[i, t] else "", int(N2a[i, t]),
                        "156:synthetic announcement" if N2a[i, t] else "", int(N[i, t]) + int(N2[i, t]), round(hhi[i, t], 4), 156, round(top[i, t], 4), 5, round(uv[i, t], 4)])
print(f"synthetic panel written: {I} items x {T} months, {int(N.sum())} trade events, {int(N2.sum())} restriction events, change at month {t0}")

