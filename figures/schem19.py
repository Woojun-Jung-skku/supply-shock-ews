"""Schematic figures: Fig. 1 of the manuscript (the inference stack) and Figs. S1, S2 and S4 of the extended
results. Drawn, not boxed.

Design system
  * Structure comes from alignment, whitespace and hairlines, not from boxes or tinted lanes.
  * Each idea is shown by a small exact glyph drawn from the model itself (a wafer map, an item-period raster,
    the zero-probability curve, a control chart, a probability against cost thresholds, a graphical model, ...);
    text is limited to a bold title and one grey line.
  * Colour carries meaning only: source side (semiconductor) blue, target side (supply chain) red, everything
    else neutral grey; one dark tone for the single shared structure.
  * One type scale (Liberation Sans, as in the result figures): headers 8 pt bold, titles 7.4 pt bold, grey lines
    6.8 pt, glyph labels 6.5 pt; hairlines 0.5 pt, glyph axes 0.5 pt, data lines 1.0 pt.
usage: python schem19.py OUT_DIR [inference_stack lineages industry_map three_layer_correspondence]"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as rf

rf.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

OUT = sys.argv[1] if len(sys.argv) > 1 else "out19"
os.makedirs(OUT, exist_ok=True)
AP = "\\alpha\\prime"
SRC, TGT = rf.BLUE, rf.RED
T1, T2, T3 = "#1A1A1A", "#5C636C", "#8B9199"
HAIR = "#C9CED4"          # hairlines, cell grids
AXIS = "#8B9199"          # glyph axes, connectors
DARK = "#262B31"
FS_H, FS_T, FS_S, FS_G = 8.0, 7.4, 6.8, 6.5
PITCH = 1.6                                     # one cell size for every die / item-period grid
REF = dict(color=AXIS, lw=0.7, ls=(0, (3.0, 1.6)))   # dashed reference lines (Poisson curve, limits)
MARGIN = 1.0
PT = 25.4 / 72            # mm per point


def tint(c, f):
    return rf.tint(c, f)


def shade(c, f):
    return rf.shade(c, f)


# ------------------------------------------------------------------------------------------------ basic marks
def txt(ax, x, y, s, fs=FS_S, c=T2, w="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va, **kw)


def hline(ax, x1, x2, y, c=HAIR, lw=0.5, **kw):
    ax.plot([x1, x2], [y, y], color=c, lw=lw, solid_capstyle="butt", **kw)


def vline(ax, x, y1, y2, c=HAIR, lw=0.5, **kw):
    ax.plot([x, x], [y1, y2], color=c, lw=lw, solid_capstyle="butt", **kw)


def arrow(ax, x1, y1, x2, y2, c=AXIS, lw=0.6, both=False, ms=5.5, z=1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>" if both else "-|>", mutation_scale=ms,
                                 lw=lw, color=c, shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter"))


def dot(ax, x, y, r, c, z=4, ec="none", lw=0):
    ax.add_patch(Circle((x, y), r, fc=c, ec=ec, lw=lw, zorder=z))


class Box:
    """maps data coordinates of a glyph onto a millimetre rectangle (x0, y0, w, h); log axes optional."""

    def __init__(self, x0, y0, w, h, xl=(0, 1), yl=(0, 1), logx=False, logy=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xl, self.yl, self.logx, self.logy = xl, yl, logx, logy

    def _f(self, v, lo, hi, log):
        v = np.asarray(v, float)
        if log:
            v, lo, hi = np.log10(v), np.log10(lo), np.log10(hi)
        return (v - lo) / (hi - lo)

    def X(self, v):
        return self.x0 + self._f(v, *self.xl, self.logx) * self.w

    def Y(self, v):
        return self.y0 + self._f(v, *self.yl, self.logy) * self.h

    def axes(self, ax, xlabel=None, ylabel=None, c=AXIS, lw=0.5, fs=FS_G, xlab_dy=2.4, ylab_dx=1.6, yaxis=True):
        """L-shaped axes with end arrows; labels at the ends (x below the right end, y left of the top end)."""
        ax.add_patch(FancyArrowPatch((self.x0, self.y0), (self.x0 + self.w + 1.2, self.y0), arrowstyle="-|>",
                                     mutation_scale=3.6, lw=lw, color=c, shrinkA=0, shrinkB=0, zorder=2))
        if yaxis:
            ax.add_patch(FancyArrowPatch((self.x0, self.y0), (self.x0, self.y0 + self.h + 1.2), arrowstyle="-|>",
                                         mutation_scale=3.6, lw=lw, color=c, shrinkA=0, shrinkB=0, zorder=2))
        if xlabel:
            txt(ax, self.x0 + self.w + 1.2, self.y0 - xlab_dy, xlabel, fs=fs, c=T2, ha="right", va="center")
        if ylabel:
            txt(ax, self.x0 - ylab_dx, self.y0 + self.h + 0.6, ylabel, fs=fs, c=T2, ha="right", va="center")

    def line(self, ax, x, y, c, lw=1.0, ls="-", z=3, clip=True, **kw):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if clip:
            ok = (y >= min(self.yl)) & (y <= max(self.yl))
            x, y = x[ok], y[ok]
        ax.plot(self.X(x), self.Y(y), color=c, lw=lw, ls=ls, solid_capstyle="round", dash_capstyle="butt",
                zorder=z, **kw)


# ------------------------------------------------------------------------------------------------ glyphs
def wafer(ax, cx, cy, r, c, pitch=1.6, seed=3):
    """wafer with a flat, a die grid and spatially clustered defects; dies with a defect are tinted."""
    rng = np.random.default_rng(seed)
    flat = -0.86 * r                          # the flat at the bottom
    th = np.linspace(0, 2 * np.pi, 400)
    pts = [(cx + r * np.cos(t), cy + max(r * np.sin(t), flat)) for t in th]
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=AXIS, lw=0.6, zorder=1, joinstyle="round"))
    # die grid, offset so that it is centred
    n = int(np.ceil(2 * r / pitch)) + 2
    off = (n * pitch) / 2 + 0.5 * pitch * (n % 2 == 0) * 0
    dies = []
    for i in range(n):
        for j in range(n):
            x, y = cx - off + i * pitch, cy - off + j * pitch
            corners = [(x, y), (x + pitch, y), (x, y + pitch), (x + pitch, y + pitch)]
            if all((px - cx) ** 2 + (py - cy) ** 2 <= (r - 0.3) ** 2 and py >= cy + flat + 0.3 for px, py in corners):
                dies.append((x, y))
    # defects: two clusters and a few isolated points
    pts = []
    for (ux, uy, k, s) in [(0.38, 0.36, 10, 0.10), (-0.42, -0.28, 6, 0.08)]:
        pts += [(cx + r * (ux + s * rng.standard_normal()), cy + r * (uy + s * rng.standard_normal())) for _ in range(k)]
    pts += [(cx + r * u, cy + r * v) for u, v in [(-0.15, 0.62), (0.05, -0.55), (-0.62, 0.22), (0.25, -0.05)]]
    hit = set()
    keep = []
    for (px, py) in pts:
        for k, (x, y) in enumerate(dies):
            if x + 0.15 < px < x + pitch - 0.15 and y + 0.15 < py < y + pitch - 0.15:
                hit.add(k)
                keep.append((px, py))
                break
    for k, (x, y) in enumerate(dies):
        ax.add_patch(Rectangle((x, y), pitch, pitch, fc=tint(c, 0.72) if k in hit else "white", ec=HAIR, lw=0.3,
                               zorder=2))
    for (px, py) in keep:
        dot(ax, px, py, 0.26, c, z=3)
    return len(dies), len(hit)


def raster(ax, x0, y0, ncol, nrow, pitch, events, c, r=None):
    """item (rows) x period (columns) grid; events = {(row, col): count}; cells with a shock are tinted."""
    for i in range(nrow):
        for j in range(ncol):
            n = events.get((i, j), 0)
            x, y = x0 + j * pitch, y0 + (nrow - 1 - i) * pitch
            ax.add_patch(Rectangle((x, y), pitch, pitch, fc=tint(c, 0.72) if n else "white", ec=HAIR, lw=0.3,
                                   zorder=2))
            q = pitch / 1.6
            offs = {1: [(0, 0)], 2: [(-0.38 * q, 0.0), (0.38 * q, 0.0)],
                    3: [(-0.4 * q, -0.3 * q), (0.4 * q, -0.3 * q), (0, 0.38 * q)]}[n] if n else []
            for dx, dy in offs:
                dot(ax, x + pitch / 2 + dx, y + pitch / 2 + dy, r or min(0.26, 0.17 * pitch), c, z=3)
    ax.add_patch(Rectangle((x0, y0), ncol * pitch, nrow * pitch, fc="none", ec=AXIS, lw=0.5, zorder=3))


def sri(mu, a):
    return np.exp(-mu) if np.isinf(a) else (1 + mu / a) ** (-a)


# ------------------------------------------------------------------------------------------------ output
def save(fig, name):
    rf.save(fig, os.path.join(OUT, name))


def width_mm(fig, ax, s, fs, w="normal"):
    t = ax.text(0, 0, s, fontsize=fs, weight=w)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 25.4


# ============================================================================ Fig. S4 of the extended results: three-layer correspondence
def model_glyph(ax, b, c, xl, yl):
    """P(N = 0) against the mean (log scale): the mixed (negative-binomial) curve in colour above the Poisson
    curve (dashed)."""
    mu = np.logspace(-2, 1, 300)
    b.axes(ax, xlabel=xl, ylabel=yl)
    ax.plot(b.X(mu), b.Y(sri(mu, np.inf)), zorder=2, **REF)
    b.line(ax, mu, sri(mu, 0.5), c, lw=1.1)
    m = 1.55
    txt(ax, b.X(m) - 1.7, b.Y(sri(m, np.inf)), "Poisson", fs=FS_G - 0.5, c=T2, ha="right")


def yticklab(ax, b, y, s, c=T2):
    """a labelled tick on the left of a glyph axis (thresholds, limits)."""
    ax.plot([b.x0 - 0.7, b.x0], [b.Y(y)] * 2, color=AXIS, lw=0.5, solid_capstyle="butt", zorder=2)
    txt(ax, b.x0 - 1.0, b.Y(y), s, fs=FS_G - 0.5, c=c, ha="right")


def three_layer_correspondence():
    """source and target side by side, three layers as rows; each cell = a glyph of what the layer handles and a
    title with one grey line; the glyphs face each other across a centre channel that names the layer; in the
    model row the channel lists the variable mapping."""
    W = 180.0
    gw, gh = 30.0, 18.0                     # glyph size
    gut0, gut1 = 74.0, 106.0                # centre channel
    gxs, gxt = gut0 - 3.0 - gw, gut1 + 3.0  # glyph x of source / target column
    xs_txt, xt_txt = gxs - 3.5, gxt + gw + 3.5
    mid = (gut0 + gut1) / 2
    row_h = [22.0, 22.0, 22.0]
    row_gap = 3.5
    head_h, foot = 12.5, 0.6
    H = head_h + sum(row_h) + 2 * row_gap + foot
    fig, ax = rf.canvas(W, H)
    hy = H - 3.0
    for x0, x1, c, tag, name in ((MARGIN, gut0, SRC, "Source", "Semiconductor yield engineering"),
                                 (gut1, W - MARGIN, TGT, "Target", "Supply-chain early warning")):
        xc = (x0 + x1) / 2
        txt(ax, xc, hy, tag, fs=FS_S, c=T2, ha="center")
        txt(ax, xc, hy - 4.4, name, fs=FS_H, c=T1, w="bold", ha="center")
        hline(ax, x0, x1, hy - 8.0, c=c, lw=0.9)
    texts = [(("Inline inspection", "defects per die on wafer maps"),
              ("Event screening", "shock events per item-period")),
             (("Negative-binomial yield", "$Y=(1+AD_0/\\alpha)^{-\\alpha}$"),
              ("Mixed-Poisson reliability", f"$\\mathrm{{SRI}}=(1+RC\\Delta/{AP})^{{-{AP}}}$")),
             (("Disposition limits", "$\\sigma$-limits on yield and scrap"),
              ("Cost-based alarm", "four levels at thresholds $\\pi_k$"))]
    layers = ["Detection", "Model", "Alarm"]
    y1 = H - head_h
    for k, ((st, tt), layer) in enumerate(zip(texts, layers)):
        y0 = y1 - row_h[k]
        yc = y1 - 11.0                      # glyph centre: same offset from the row top in every row
        txt(ax, xs_txt, yc + 2.3, st[0], fs=FS_T, c=T1, w="bold", ha="right")
        txt(ax, xs_txt, yc - 2.4, st[1], fs=FS_S, c=T2, ha="right")
        txt(ax, xt_txt, yc + 2.3, tt[0], fs=FS_T, c=T1, w="bold", ha="left")
        txt(ax, xt_txt, yc - 2.4, tt[1], fs=FS_S, c=T2, ha="left")
        txt(ax, mid, yc + 2.6, layer, fs=FS_T, c=T1, w="bold", ha="center")
        arrow(ax, gut0 + 0.8, yc - 0.6, gut1 - 0.8, yc - 0.6, both=True, c=AXIS, lw=0.6)
        gy = yc - gh / 2
        if k == 0:
            wafer(ax, gxs + gw / 2, yc, 9.0, SRC, pitch=PITCH)
            ncol, nrow = 15, 8
            ev = {(1, 2): 1, (5, 3): 1, (0, 7): 1, (1, 7): 2, (2, 7): 1, (3, 7): 1, (1, 8): 1, (2, 8): 1, (6, 10): 1,
                  (4, 13): 1}
            raster(ax, gxt + (gw - ncol * PITCH) / 2, yc - nrow * PITCH / 2, ncol, nrow, PITCH, ev, TGT)
        elif k == 1:
            for x0, c, xl, yl in ((gxs, SRC, "$AD_0$", "$Y$"), (gxt, TGT, "$RC\\Delta$", "SRI")):
                model_glyph(ax, Box(x0 + 5.0, gy + 2.0, gw - 7.0, gh - 3.5, xl=(0.01, 10), yl=(0, 1), logx=True),
                            c, xl, yl)
            # variable mapping as a small table: source | arrow | target | role
            pairs = [("$D_0$", "$R$", "intensity"), ("$A$", "$C$", "exposure"), ("$\\alpha$", f"${AP}$", "dispersion"),
                     ("$Y$", "SRI", "reliability")]
            wl = max(width_mm(fig, ax, a, FS_G) for a, _, _ in pairs)
            wr = max(width_mm(fig, ax, b_, FS_G) for _, b_, _ in pairs)
            wrole = max(width_mm(fig, ax, r, FS_G - 0.3) for _, _, r in pairs)
            wa, g1, g2 = 2.6, 0.9, 2.2
            tot = wl + g1 + wa + g1 + wr + g2 + wrole
            xl0 = mid - tot / 2
            for j, (a, b_, role) in enumerate(pairs):
                yy = yc - 4.8 - j * 2.95
                txt(ax, xl0 + wl, yy, a, fs=FS_G, c=T1, ha="right")
                txt(ax, xl0 + wl + g1 + wa / 2, yy, "$\\leftrightarrow$", fs=FS_G, c=T2, ha="center")
                txt(ax, xl0 + wl + g1 + wa + g1, yy, b_, fs=FS_G, c=T1, ha="left")
                txt(ax, xl0 + wl + g1 + wa + g1 + wr + g2, yy, role, fs=FS_G - 0.3, c=T2, ha="left")
        else:
            rng = np.random.default_rng(7)
            yv = 0.86 + 0.035 * rng.standard_normal(16)
            yv[11] = 0.70
            b = Box(gxs + 5.0, gy + 2.0, gw - 7.0, gh - 3.5, xl=(-0.5, 15.5), yl=(0.62, 0.98))
            b.axes(ax, xlabel="lot", ylabel="$Y$")
            hline(ax, b.X(-0.5), b.X(15.5), b.Y(0.86), c=HAIR, lw=0.5)
            ax.plot([b.X(-0.5), b.X(15.5)], [b.Y(0.755)] * 2, zorder=2, **REF)
            yticklab(ax, b, 0.755, "$-3\\sigma$")
            xs = np.arange(16)
            ax.plot(b.X(xs), b.Y(yv), color=tint(SRC, 0.35), lw=0.5, zorder=3)
            for i, v in enumerate(yv):
                dot(ax, b.X(i), b.Y(v), 0.42 if i != 11 else 0.62, SRC, z=4)
            b = Box(gxt + 5.0, gy + 2.0, gw - 7.0, gh - 3.5, xl=(0, 23), yl=(0.01, 1), logy=True)
            pis = [0.048, 0.167, 0.5]
            for lo, hi, f in ((pis[0], pis[1], 0.88), (pis[1], pis[2], 0.75), (pis[2], 1.0, 0.60)):
                ax.add_patch(Rectangle((b.X(0), b.Y(lo)), b.w, b.Y(hi) - b.Y(lo), fc=tint(TGT, f), ec="none",
                                       zorder=1))
            for j, pi_ in enumerate(pis):
                yticklab(ax, b, pi_, f"$\\pi_{j + 1}$")
            t = np.arange(24)
            pv = 0.02 * np.exp(0.17 * np.maximum(t - 6, 0)) * (1 + 0.18 * np.sin(1.3 * t))
            pv = np.clip(pv, 0.012, 0.9)
            b.axes(ax, xlabel="month", ylabel="$p$")
            ax.plot(b.X(t), b.Y(pv), color=shade(TGT, 0.15), lw=1.0, zorder=3, solid_joinstyle="round")
        y1 = y0 - row_gap
    save(fig, "FigS4_three_layer_correspondence")


# ============================================================================ Fig. S1 of the extended results: lineages
def letter(ax, x, y, L):
    txt(ax, x, y, f"({L})", fs=9, c=T1, w="bold", ha="left", va="top")


def lineages():
    """(a) the die-yield lineage as a sequence of mixing distributions of the defect density D0 (point mass,
    triangle, exponential, gamma), each with its yield formula; (b) the gamma-Poisson family: domains around one
    hub that shows the construction (gamma intensity -> Poisson counts -> negative binomial, P(N = 0) singled out)
    and the identity they share; the supply-shock case of this paper below the hub."""
    from scipy.special import gammaln
    W = 180.0
    Ha, Hb = 35.0, 52.0
    H = Ha + Hb
    fig, ax = rf.canvas(W, H)
    # ------------------------------------------------------------------ (a)
    letter(ax, MARGIN, H - MARGIN, "a")
    gw, gh = 24.0, 11.5
    xs = [25.0, 69.0, 113.0, 157.0]
    stations = [("Poisson", "$Y=\\exp(-AD_0)$", "spike"),
                ("Murphy 1964", "$Y=[(1-\\exp(-AD_0))/AD_0]^{2}$", "tri"),
                ("Seeds 1967, Price 1970", "$Y=1/(1+AD_0)$", "exp"),
                ("Stapper 1973", "$Y=(1+AD_0/\\alpha)^{-\\alpha}$", "gamma")]
    d = np.linspace(0, 3, 300)
    dens = {"tri": np.where(d <= 1, d, 2 - d), "exp": np.exp(-d), "gamma": 4 * d * np.exp(-2 * d)}
    gy = H - 5.0 - gh
    for k, (x, (name, form, kind)) in enumerate(zip(xs, stations)):
        last = kind == "gamma"
        col = SRC if last else T2
        b = Box(x - gw / 2, gy, gw, gh, xl=(0, 3), yl=(0, 1.08))
        b.axes(ax, xlabel="$D_0$", ylabel="$f(D_0)$" if k == 0 else None, xlab_dy=2.2)
        if kind == "spike":
            ax.add_patch(FancyArrowPatch((b.X(1), b.Y(0)), (b.X(1), b.Y(1.0)), arrowstyle="-|>", mutation_scale=4.5,
                                         lw=1.1, color=col, shrinkA=0, shrinkB=0, zorder=3))
        else:
            yv = dens[kind]
            ok = d <= 2.0 if kind == "tri" else d >= 0
            ax.fill_between(b.X(d[ok]), b.Y(0), b.Y(yv[ok]), color=tint(col, 0.80 if last else 0.86), lw=0, zorder=2)
            ax.plot(b.X(d[ok]), b.Y(yv[ok]), color=col, lw=1.1, zorder=3, solid_capstyle="round")
        ty = gy - 6.6
        txt(ax, x, ty, name, fs=FS_T, c=T1, w="bold", ha="center")
        txt(ax, x, ty - 4.2, form, fs=FS_S, c=T2, ha="center")
        if k < 3:
            arrow(ax, x + gw / 2 + 3.5, gy + gh * 0.45, xs[k + 1] - gw / 2 - 3.0, gy + gh * 0.45, c=AXIS, lw=0.6)
    # ------------------------------------------------------------------ (b)
    letter(ax, MARGIN, Hb - 0.5, "b")
    hw, hh = 62.0, 31.0
    hx, hy = W / 2 - hw / 2, 15.0
    ax.add_patch(Rectangle((hx, hy), hw, hh, fc="white", ec=DARK, lw=0.8, zorder=2, joinstyle="miter"))
    txt(ax, W / 2, hy + hh - 3.6, "Gamma-mixed Poisson", fs=FS_H, c=T1, w="bold", ha="center")
    a, m = 2.0, 1.0
    lam = np.linspace(0, 3, 300)
    g = np.exp(a * np.log(a) - gammaln(a) + (a - 1) * np.log(np.maximum(lam, 1e-12)) - a * lam)
    b1 = Box(hx + 6.0, hy + 10.0, 18.0, 9.5, xl=(0, 3), yl=(0, 0.9))
    b1.axes(ax, xlabel="$\\lambda$", xlab_dy=2.1)
    ax.fill_between(b1.X(lam), b1.Y(0), b1.Y(g), color=tint(T2, 0.86), lw=0, zorder=2)
    ax.plot(b1.X(lam), b1.Y(g), color=T2, lw=1.0, zorder=3)
    txt(ax, b1.X(1.35), b1.Y(0.62), "gamma", fs=FS_G, c=T2, ha="left")
    arrow(ax, b1.x0 + b1.w + 3.4, b1.y0 + 4.8, b1.x0 + b1.w + 12.6, b1.y0 + 4.8, c=AXIS, lw=0.6, z=4)
    txt(ax, b1.x0 + b1.w + 8.0, b1.y0 + 6.9, "Poisson", fs=FS_G, c=T2, ha="center")
    n = np.arange(6)
    nb = np.exp(gammaln(n + a) - gammaln(n + 1) - gammaln(a) + a * np.log(a / (a + m)) + n * np.log(m / (a + m)))
    b2 = Box(b1.x0 + b1.w + 16.0, hy + 10.0, 18.0, 9.5, xl=(-0.7, 5.7), yl=(0, 0.5))
    b2.axes(ax, xlabel="$N$", xlab_dy=2.1)
    for k_ in n:
        ax.add_patch(Rectangle((b2.X(k_ - 0.32), b2.Y(0)), b2.X(0.32) - b2.X(-0.32), b2.Y(nb[k_]) - b2.Y(0),
                               fc=DARK if k_ == 0 else tint(T2, 0.55), ec="none", zorder=2))
    txt(ax, b2.X(0.6), b2.Y(0.47), "$P(N=0)$", fs=FS_G, c=T1, ha="left")
    txt(ax, W / 2, hy + 4.3, "$P(N=0)=\\mathrm{E}\\,[e^{-\\lambda s}]=(1+\\mu/\\alpha)^{-\\alpha}$", fs=FS_T, c=T1,
        ha="center")
    dom_l = [("Accident proneness", "Greenwood & Yule 1920", None), ("Semiconductor yield", "Stapper 1973", SRC),
             ("Frailty", "Vaupel et al. 1979", None)]
    dom_r = [("Software reliability", "Goel & Okumoto 1979", None), ("Credit risk", "CreditRisk+ 1997", None),
             ("Superspreading", "Lloyd-Smith et al. 2005", None)]
    ys = [hy + hh - 5.0, hy + hh / 2, hy + 5.0]
    xl_txt, xr_txt = hx - 16.0, hx + hw + 16.0
    for side, doms in (("L", dom_l), ("R", dom_r)):
        for (name, ref, c), y in zip(doms, ys):
            xt = xl_txt if side == "L" else xr_txt
            ha = "right" if side == "L" else "left"
            txt(ax, xt, y + 1.9, name, fs=FS_T, c=c or T1, w="bold", ha=ha)
            txt(ax, xt, y - 2.0, ref, fs=FS_S, c=T2, ha=ha)
            x_end = hx if side == "L" else hx + hw
            x_beg = xt + (2.5 if side == "L" else -2.5)
            ax.plot([x_beg, x_end], [y, y], color=c or AXIS, lw=0.6, solid_capstyle="butt", zorder=1)
            dot(ax, x_beg, y, 0.55, c or AXIS, z=3)
    yd = 9.0
    ax.plot([W / 2, W / 2], [hy, yd], color=TGT, lw=0.6, zorder=1)
    dot(ax, W / 2, yd, 0.55, TGT, z=3)
    txt(ax, W / 2, yd - 3.0, "Supply shocks", fs=FS_T, c=TGT, w="bold", ha="center")
    txt(ax, W / 2, yd - 6.7, "this paper", fs=FS_S, c=T2, ha="center")
    save(fig, "FigS1_lineages")


# ============================================================================ Fig. 1: inference stack
BASE = 4.2        # height of a glyph's x-axis above the bottom of its slot (x labels sit below it)


def node(ax, x, y, label, r=2.2, fc="white", fs=FS_G):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=DARK, lw=0.6, zorder=3))
    txt(ax, x, y - 0.05, label, fs=fs, c=T1, ha="center", va="center", zorder=4)


def edge(ax, p, q, r=2.2, c=T2):
    p, q = np.array(p, float), np.array(q, float)
    d = (q - p) / np.hypot(*(q - p))
    a, b = p + d * (r + 0.25), q - d * (r + 0.35)
    ax.add_patch(FancyArrowPatch(tuple(a), tuple(b), arrowstyle="-|>", mutation_scale=4.2, lw=0.55, color=c,
                                 shrinkA=0, shrinkB=0, zorder=2))


def g_data(ax, x0, y0, w, h):
    """counts N (items x periods) with the indicators x above, aligned to the periods, and the exposure
    measurements z as one bar per item on the right."""
    ncol, nrow = 9, 5
    rx, ry = x0 + 5.0, y0 + BASE
    ev = {(0, 5): 1, (1, 5): 2, (1, 6): 1, (3, 2): 1, (4, 7): 1}
    raster(ax, rx, ry, ncol, nrow, PITCH, ev, TGT)
    t = np.arange(ncol) + 0.5
    for k, (c, ph) in enumerate(((T2, 0.0), (T3, 1.7))):
        yv = 0.5 + 0.32 * np.sin(0.75 * t + ph) + 0.12 * k
        ax.plot(rx + t * PITCH, ry + nrow * PITCH + 1.6 + yv * 4.2, color=c, lw=0.8, zorder=3, solid_capstyle="round")
    zx = rx + ncol * PITCH + 1.2
    for i, z in enumerate([0.9, 0.45, 0.7, 0.3, 0.6]):
        y = ry + (nrow - 1 - i) * PITCH + 0.35
        ax.add_patch(Rectangle((zx, y), z * 5.0, PITCH - 0.7, fc=tint(T2, 0.45), ec="none", zorder=2))
    txt(ax, rx - 1.0, ry + nrow * PITCH / 2, "$N$", fs=FS_G, c=T2, ha="right")
    txt(ax, rx - 1.0, ry + nrow * PITCH + 1.6 + 2.4, "$x$", fs=FS_G, c=T2, ha="right")
    txt(ax, zx + 2.4, ry + nrow * PITCH + 1.3, "$z$", fs=FS_G, c=T2, ha="center")


def g_dag(ax, x0, y0, w, h):
    """graphical model: group trend m and lagged indicators x drive the intensity R, measurements z the exposure C
    (weights with an AHP-informed prior); with the group dispersion alpha' they give the observed count N."""
    X = [x0 + 5.6, x0 + 15.4, x0 + 25.2]
    Y = [y0 + h - 2.3, 0.0, y0 + BASE + 2.0]
    Y[1] = (Y[0] + Y[2]) / 2
    pos = {"m": (X[0], Y[0]), "x": (X[0], Y[1]), "z": (X[0], Y[2]), "R": (X[1], (Y[0] + Y[1]) / 2 + 0.4),
           "C": (X[1], Y[2]), "a": (X[2], Y[0]), "N": (X[2], Y[1] - 1.6)}
    for a, b in (("m", "R"), ("x", "R"), ("z", "C"), ("R", "N"), ("C", "N"), ("a", "N")):
        edge(ax, pos[a], pos[b])
    lab = {"m": "$m^g$", "x": "$x$", "z": "$z$", "R": "$R$", "C": "$C$", "a": f"${AP}_g$", "N": "$N$"}
    fill = {"x": tint(T2, 0.80), "z": tint(T2, 0.80), "N": tint(TGT, 0.72)}
    for k, (x, y) in pos.items():
        node(ax, x, y, lab[k], fc=fill.get(k, "white"), fs=FS_G - (0.6 if k in ("m", "a") else 0))
    txt(ax, (X[0] + X[1]) / 2, Y[2] + 1.35, "AHP", fs=FS_G - 0.5, c=T2, ha="center")


def g_particles(ax, x0, y0, w, h):
    """full NUTS refits at a fixed interval (ticks) reset the posterior; between refits it is propagated forward
    (widening band) by particle reweighting."""
    b = Box(x0 + 2.0, y0 + BASE, w - 4.5, h - BASE - 4.6, xl=(0, 18), yl=(-1.0, 1.0))
    b.axes(ax, xlabel="month", yaxis=False)
    lvl = [-0.25, 0.0, 0.2]
    for k, r0 in enumerate((0, 6, 12)):
        tt = np.linspace(r0, r0 + 6, 40)
        c0 = lvl[k] + (tt - r0) * 0.012
        sd = 0.10 + 0.075 * (tt - r0)
        ax.fill_between(b.X(tt), b.Y(c0 - sd), b.Y(c0 + sd), color=tint(T2, 0.84), lw=0, zorder=2)
        ax.plot(b.X(tt), b.Y(c0), color=T2, lw=0.8, zorder=3)
    for r0 in (0, 6, 12, 18):
        ax.plot([b.X(r0)] * 2, [b.Y(-1.0), b.Y(1.0)], color=T1, lw=0.7, zorder=4, solid_capstyle="butt")
    yk = b.Y(1.0) + 2.0
    arrow(ax, b.X(6), yk, b.X(12), yk, both=True, c=T2, lw=0.45, ms=3.0, z=5)
    txt(ax, b.X(9), yk + 1.6, "refit interval", fs=FS_G - 0.5, c=T2, ha="center")
    txt(ax, b.X(0) - 0.4, yk, "NUTS", fs=FS_G - 0.5, c=T2, ha="left", va="center")


def g_uncertainty(ax, x0, y0, w, h):
    """posterior predictive fan of the supply reliability index over the forecast horizon."""
    hb = h - BASE - 2.0
    b = Box(x0 + 7.5, y0 + BASE, w - 13.0, hb, xl=(0, 6), yl=(0, 1))
    b.axes(ax, xlabel="$h$", ylabel="SRI")
    hh = np.linspace(0, 6, 60)
    med = 0.62 - 0.035 * hh
    for q, f in ((0.30, 0.86), (0.16, 0.72)):
        sd = q * np.sqrt(hh / 6)
        ax.fill_between(b.X(hh), b.Y(np.clip(med - sd, 0, 1)), b.Y(np.clip(med + sd, 0, 1)), color=tint(TGT, f), lw=0,
                        zorder=2)
    ax.plot(b.X(hh), b.Y(med), color=TGT, lw=0.9, zorder=3)


def g_alarm(ax, x0, y0, w, h):
    """two channels: the probability of a shock against the cost thresholds (top) and the level-shift
    probability against its bootstrap-calibrated threshold (bottom)."""
    t = np.arange(20)
    ph = (h - BASE - 1.2 - 2.2 - 1.2) / 2
    b2 = Box(x0 + 4.0, y0 + BASE, w - 10.0, ph, xl=(0, 19), yl=(0, 1))
    b = Box(x0 + 4.0, y0 + BASE + ph + 1.2 + 2.2, w - 10.0, ph, xl=(0, 19), yl=(0.01, 1), logy=True)
    pis = [0.048, 0.167, 0.5]
    for lo, hi, f in ((pis[0], pis[1], 0.88), (pis[1], pis[2], 0.75), (pis[2], 1.0, 0.60)):
        ax.add_patch(Rectangle((b.X(0), b.Y(lo)), b.w, b.Y(hi) - b.Y(lo), fc=tint(TGT, f), ec="none", zorder=1))
    pv = np.clip(0.02 * np.exp(0.2 * np.maximum(t - 5, 0)) * (1 + 0.15 * np.sin(1.3 * t)), 0.012, 0.85)
    b.axes(ax, ylabel="$p$")
    for pi_ in pis:
        ax.plot([b.x0 - 0.7, b.x0], [b.Y(pi_)] * 2, color=AXIS, lw=0.5, solid_capstyle="butt", zorder=2)
    txt(ax, b.x0 - 1.0, b.Y(pis[1]), "$\\pi_k$", fs=FS_G - 0.5, c=T2, ha="right")
    ax.plot(b.X(t), b.Y(pv), color=shade(TGT, 0.15), lw=0.9, zorder=3)
    qv = np.clip(0.08 + 0.05 * np.sin(1.7 * t) + np.where(t > 11, 0.11 * (t - 11), 0), 0, 0.95)
    b2.axes(ax, xlabel="month", ylabel="$q$")
    ax.plot([b2.X(0), b2.X(19)], [b2.Y(0.62)] * 2, zorder=2, **REF)
    ax.plot(b2.X(t), b2.Y(qv), color=T2, lw=0.9, zorder=3)


def inference_stack():
    """the inference stack as a left-to-right pipeline: five stages, each a title, a glyph and one short line."""
    W = 180.0
    stages = [("Detection layer", "$N_{it}$, $x_{it}$, $z_i$", g_data),
              ("Hierarchical NB model", f"$N\\sim\\mathrm{{NB}}(RC\\Delta,\\,{AP}_g)$", g_dag),
              ("Posterior computation", "refit and propagate", g_particles),
              ("Uncertainty", "posterior predictive", g_uncertainty),
              ("Two-channel alarm", "triage $p$ and change $q$", g_alarm)]
    n = len(stages)
    gap = 4.6
    cw = (W - 2 * MARGIN - (n - 1) * gap) / n
    gh = 20.0
    H = 32.0
    fig, ax = rf.canvas(W, H)
    ty = H - 2.6
    gy = ty - 4.8 - gh
    for k, (title, sub, g) in enumerate(stages):
        x0 = MARGIN + k * (cw + gap)
        txt(ax, x0 + cw / 2, ty, title, fs=FS_T, c=T1, w="bold", ha="center")
        g(ax, x0, gy, cw, gh)
        txt(ax, x0 + cw / 2, gy - 1.9, sub, fs=FS_S, c=T2, ha="center")
        if k < n - 1:
            arrow(ax, x0 + cw + 0.7, gy + BASE + 6.0, x0 + cw + gap - 0.7, gy + BASE + 6.0, c=AXIS, lw=0.6, ms=4.5)
    save(fig, "Fig1_inference_stack")


# ============================================================================ Fig. S2 of the extended results: industry map
def _units(ax, rects, hits, c, dots):
    for k, (x, y, w, h) in enumerate(rects):
        ax.add_patch(Rectangle((x, y), w, h, fc=tint(c, 0.72) if k in hits else "white", ec=AXIS, lw=0.45, zorder=2))
    for (x, y) in dots:
        dot(ax, x, y, 0.24, c, z=3)


def g_battery(ax, x0, y0, w, h, c):
    """two electrode coating rolls cut into cells; defects cluster along one roll (roll and lot correlation)."""
    n, cwid, ch = 8, (w - 2) / 8, 3.6
    rects, hits, dots = [], set(), []
    for r, yy in enumerate((y0 + h - ch - 1.2, y0 + 1.6)):
        for k in range(n):
            rects.append((x0 + 1 + k * cwid, yy, cwid, ch))
    for k in (2, 3, 4):
        hits.add(k)
    rng = np.random.default_rng(5)
    for k, m in ((2, 2), (3, 3), (4, 1)):
        x, y, ww, hh = rects[k]
        dots += [(x + ww * u, y + hh * v) for u, v in rng.uniform(0.25, 0.75, (m, 2))]
    x, y, ww, hh = rects[n + 6]
    dots.append((x + ww / 2, y + hh / 2))
    hits.add(n + 6)
    _units(ax, rects, hits, c, dots)


def g_display(ax, x0, y0, w, h, c):
    """mother glass cut into six panels; a mura cluster spans two adjacent panels."""
    gw_, gh_ = w - 4, h - 1.5
    gx, gy_ = x0 + 2, y0 + 0.6
    ax.add_patch(Rectangle((gx, gy_), gw_, gh_, fc="white", ec=AXIS, lw=0.5, zorder=1))
    pw, ph = (gw_ - 4 * 0.8) / 3, (gh_ - 3 * 0.8) / 2
    rects = [(gx + 0.8 + i * (pw + 0.8), gy_ + 0.8 + j * (ph + 0.8), pw, ph) for j in (1, 0) for i in range(3)]
    rng = np.random.default_rng(9)
    cx, cy = rects[1][0] + pw * 0.92, rects[1][1] + ph * 0.45
    dots = [(cx + 1.3 * u, cy + 1.0 * v) for u, v in rng.standard_normal((7, 2))]
    hits, keep = set(), []
    for (px, py) in dots:
        for k, (x, y, ww, hh) in enumerate(rects):
            if x + 0.2 < px < x + ww - 0.2 and y + 0.2 < py < y + hh - 0.2:
                hits.add(k)
                keep.append((px, py))
    keep.append((rects[3][0] + pw * 0.3, rects[3][1] + ph * 0.6))
    hits.add(3)
    _units(ax, rects, hits, c, keep)


def g_biopharma(ax, x0, y0, w, h, c):
    """two lines of successive batches; a contamination run hits consecutive batches that share a raw-material
    lot."""
    n, r = 7, 1.45
    step = (w - 2 * r - 2) / (n - 1)
    for row, yy in enumerate((y0 + h - r - 1.0, y0 + r + 0.9)):
        for k in range(n):
            x = x0 + 1 + r + k * step
            hit = (row == 0 and k in (3, 4, 5)) or (row == 1 and k == 1)
            ax.add_patch(Circle((x, yy), r, fc=tint(c, 0.72) if hit else "white", ec=AXIS, lw=0.45, zorder=2))
            if hit:
                dot(ax, x, yy, 0.24, c, z=3)


def g_datacentre(ax, x0, y0, w, h, c):
    """four racks of nodes; failures cluster in one rack (shared power, a common cause)."""
    racks, nodes = 4, 5
    rw = 3.4
    gap_ = (w - 2 - racks * rw) / (racks - 1)
    nh = (h - 1.6 - (nodes - 1) * 0.5) / nodes
    rects, hits, dots = [], set(), []
    for rk in range(racks):
        for j in range(nodes):
            x, y = x0 + 1 + rk * (rw + gap_), y0 + 0.8 + j * (nh + 0.5)
            rects.append((x, y, rw, nh))
            if rk == 2 and j in (1, 2, 3):
                hits.add(len(rects) - 1)
                dots.append((x + rw / 2, y + nh / 2))
    hits.add(4)
    dots.append((rects[4][0] + rw / 2, rects[4][1] + nh / 2))
    _units(ax, rects, hits, c, dots)


def g_space(ax, x0, y0, w, h, c):
    """missions as timelines of different duration (exposure), from launch to end (dot); a common-cause cluster of
    failures ends one mission early (cross)."""
    lens = [0.95, 0.62, 0.80]
    ys = [y0 + h - 2.2, y0 + h / 2, y0 + 2.2]
    xa = x0 + 1.5
    span = w - 3.0
    for k, (L, yy) in enumerate(zip(lens, ys)):
        lost = k == 1
        xe = xa + L * span
        ax.plot([xa, xe], [yy, yy], color=c if lost else AXIS, lw=1.0 if lost else 0.8, solid_capstyle="round",
                zorder=2)
        if lost:
            for u in (0.80, 0.88, 0.95):
                dot(ax, xa + L * span * u, yy, 0.30, c, z=3)
            s_ = 0.9
            ax.plot([xe + 0.9 - s_, xe + 0.9 + s_], [yy - s_, yy + s_], color=c, lw=0.9, zorder=3)
            ax.plot([xe + 0.9 - s_, xe + 0.9 + s_], [yy + s_, yy - s_], color=c, lw=0.9, zorder=3)
        else:
            dot(ax, xe, yy, 0.42, AXIS, z=3)


def industry_map():
    """five industries as columns. Process level (where the source model already operates): a glyph of the
    industry's own counting unit with clustered defects, and its output. Supply level (where the transferred
    framework applies): an item-period raster with one row per named critical input, and the inputs. Between the
    two levels, framed: the shared structure."""
    W = 180.0
    inds = [("Li-ion batteries", "cell yield, lot scrap", "precursors, separators", g_battery, 2,
             {(0, 4): 1, (1, 4): 1, (1, 5): 1, (0, 11): 1}),
            ("Displays", "panel yield, breakage", "emitter materials", g_display, 1,
             {(0, 2): 1, (0, 8): 1, (0, 9): 1}),
            ("Biopharma", "batch purity, rejection", "media, resins", g_biopharma, 2,
             {(1, 3): 1, (0, 7): 1, (1, 7): 1, (0, 8): 1}),
            ("AI data centres", "job completion, uptime", "HBM, accelerators", g_datacentre, 2,
             {(0, 9): 1, (1, 9): 1, (1, 10): 1, (1, 1): 1}),
            ("Space systems", "mission success, loss", "space-grade parts", g_space, 1,
             {(0, 5): 1, (0, 6): 1, (0, 12): 1})]
    x0, gap = 17.0, 3.0
    xr = W - 1.5
    cw = (xr - x0 - 4 * gap) / 5
    gh, bh = 12.0, 7.0
    ncol = 14
    H = 45.0
    fig, ax = rf.canvas(W, H)
    head_y = H - 3.0
    rule_y = H - 6.0
    gy = rule_y - 2.2 - gh
    out_y = gy - 3.0
    band_y = out_y - 3.6 - bh
    r_top = band_y - 2.6
    sup_y = r_top - 2 * PITCH - 3.2
    hline(ax, x0, xr, rule_y, c=DARK, lw=0.6)
    for k, (name, out, sup, g, nrow, ev) in enumerate(inds):
        x = x0 + k * (cw + gap)
        xc = x + cw / 2
        txt(ax, xc, head_y, name, fs=FS_T, c=T1, w="bold", ha="center")
        g(ax, x + 2.0, gy, cw - 4.0, gh, SRC)
        txt(ax, xc, out_y, out, fs=FS_S, c=T2, ha="center")
        vline(ax, xc, out_y - 1.8, band_y + bh, c=AXIS, lw=0.5)
        vline(ax, xc, band_y, r_top, c=AXIS, lw=0.5)
        raster(ax, xc - ncol * PITCH / 2, r_top - nrow * PITCH, ncol, nrow, PITCH, ev, TGT)
        txt(ax, xc, sup_y, sup, fs=FS_S, c=T2, ha="center")
    ax.add_patch(Rectangle((x0, band_y), xr - x0, bh, fc="white", ec=DARK, lw=0.8, zorder=3, joinstyle="miter"))
    txt(ax, x0 + (xr - x0) / 2, band_y + bh / 2,
        "One structure:   intensity $\\lambda$   ·   exposure $s$   ·   dispersion $\\alpha$",
        fs=FS_T, c=T1, w="bold", ha="center", zorder=4)
    for y, lab, c in ((gy + gh / 2, ("Process", "level"), SRC), (r_top - PITCH, ("Supply", "level"), TGT)):
        ax.add_patch(Rectangle((MARGIN, y + 0.9), 2.0, 2.0, fc=c, ec="none"))
        txt(ax, MARGIN + 3.2, y + 1.9, lab[0], fs=FS_T, c=T1, w="bold", ha="left")
        txt(ax, MARGIN + 3.2, y - 1.6, lab[1], fs=FS_T, c=T1, w="bold", ha="left")
    save(fig, "FigS2_industry_map")


FIGS = dict(three_layer_correspondence=three_layer_correspondence, lineages=lineages, inference_stack=inference_stack,
            industry_map=industry_map)
if __name__ == "__main__":
    for n in (sys.argv[2:] or FIGS):
        FIGS[n]()
        print("done", n)
