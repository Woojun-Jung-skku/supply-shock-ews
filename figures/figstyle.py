"""Colour figure style shared by every figure of the manuscript.

Follows the usual style of journal figures in the field (coloured series told apart by colour and marker, bold (a)/(b)
panel letters, shared legend in a row above the panels, tinted boxes in schematics) within one fixed
palette and one type scale:
  * Liberation Sans (Arial metrics), math in the same face; 8 pt labels, 7.5 pt ticks and legends,
    9 pt bold panel letters; drawn at 180 mm (double column) or 140 mm, PDF with embedded TrueType fonts.
  * nine-colour palette in a fixed order; an ordered parameter (alpha', mu, n) takes the cool-to-warm ramp,
    a method keeps one colour and marker in every figure.
  * text in black or grey only; no titles or notes inside a figure (they are in the captions);
    no grid, no top/right spines, square box corners, no shadows or gradients.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

MM = 1 / 25.4
W2, W15, W1 = 180 * MM, 140 * MM, 90 * MM

PAL = ["#2E5FA3", "#D4553F", "#2A9D8F", "#B8860B", "#7B5EA7", "#5B8C3E", "#E69F00", "#56B4E9", "#CC79A7"]
BLUE, RED, TEAL, GOLD, PURPLE, GREEN, ORANGE, SKY, PINK = PAL
INK, SUB, MUTED, RULE = "#1a1a1a", "#555555", "#8c8c8c", "#bdbdbd"


def tint(c, f):
    """mix colour c with white; f = share of white (0 = c, 1 = white)."""
    r, g, b = mcolors.to_rgb(c)
    return mcolors.to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def shade(c, f):
    """mix colour c with black; f = share of black."""
    r, g, b = mcolors.to_rgb(c)
    return mcolors.to_hex((r * (1 - f), g * (1 - f), b * (1 - f)))


# ordered parameter levels: cool -> warm (low -> high), each level also has its own marker
RAMP = {3: [BLUE, TEAL, RED], 4: [BLUE, TEAL, GOLD, RED], 5: [PURPLE, BLUE, TEAL, GOLD, RED]}
MARKS = ["o", "s", "^", "D", "v", "p"]

# methods: one colour and marker everywhere (Figs. 4-7)
#   name -> (colour, marker, filled)
METHOD = {
    "M2": (RED, "o", True),               # proposed; the hierarchical model of Fig. 4
    "NBcomplete": (BLUE, "^", True),      # complete pooling
    "NBnone": (TEAL, "s", True),          # no pooling
    "M0": (GOLD, "o", False),             # Poisson M0 (open circle)
    "xgb": (GREEN, "D", True),
    "lstm": (tint(GREEN, 0.0), "v", False),
    "tft": (shade(GREEN, 0.15), "X", True),
    "sigma": (PURPLE, "P", True),
    "ewma": (PURPLE, "D", False),
    "cusum": (PURPLE, "s", False),
    "farr": (PURPLE, "^", False),         # Farrington-type threshold (A3)
    "nbcusum": (PURPLE, "s", True),       # CUSUM designed under the negative binomial (B1; v1.24)
}

# schematic roles (Fig. 3, Figs. S1, S2, S4): source side, target side, shared structure, other domains
ROLE = {"source": BLUE, "target": RED, "shared": PURPLE, "other": TEAL, "neutral": "#6b7785"}
FILL_T = 0.88          # white share of a box fill
FILL_T_STRONG = 0.80   # emphasised box


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold", "mathtext.sf": "Liberation Sans", "mathtext.cal": "Liberation Sans:italic",
        "mathtext.fallback": "stixsans",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.4, "legend.borderaxespad": 0.3,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45, "xtick.major.size": 2.8, "ytick.major.size": 2.8, "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6, "xtick.major.pad": 2.2, "ytick.major.pad": 2.2,
        "lines.linewidth": 1.1, "lines.markersize": 4.0, "lines.markeredgewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK, "axes.grid": False,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "figure.dpi": 150, "savefig.pad_inches": 0.02,
    })


def panel(ax, letter, dx=-30, dy=5):
    """bold (a), (b), ... at the top left of an axes; dx, dy offsets in points from the axes corner."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="left", va="bottom", annotation_clip=False)


def letters(fig, axes, labels="abcdefgh", align=None, groups=()):
    """bold (a), (b), ... as left titles of the axes (constrained layout keeps room for them), moved left to the
    outer edge of each axes' y-axis labels; `align` per axes: "ylabel" (default) or an x in axes fraction;
    `groups`: tuples of axes indices whose letters share the left-most x."""
    align = align or ["ylabel"] * len(axes)
    ts = [ax.set_title(f"({L})", loc="left", fontsize=9, fontweight="bold", pad=4) for ax, L in zip(axes, labels)]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xpix = []
    for ax, al in zip(axes, align):
        ab = ax.get_window_extent(r)
        xpix.append(ax.yaxis.get_tightbbox(r).x0 if al == "ylabel" else ab.x0 + al * ab.width)
    for g in groups:
        m = min(xpix[i] for i in g)
        for i in g:
            xpix[i] = m
    for ax, t, x in zip(axes, ts, xpix):
        ab = ax.get_window_extent(r)
        t.set_x((x - ab.x0) / ab.width)
    return ts


def series(ax, x, y, k, n, label=None, marks=8, ls="-", lw=None, logx=True, mrange=None, ymin=None, **kw):
    """ordered-parameter curve: ramp colour k of n, open marker at `marks` evenly spaced (log) positions, within
    `mrange` (x interval) if given, e.g. to keep markers off a range where all curves coincide."""
    c = RAMP[n][k]
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = np.log(x) if logx else x
    lo, hi = (u[0], u[-1]) if mrange is None else ((np.log(mrange[0]), np.log(mrange[1])) if logx else mrange)
    step = (hi - lo) / (marks + 1)
    phase = (k - (n - 1) / 2) / n * 0.9          # stagger the markers of different curves where curves coincide
    targets = lo + step * (np.arange(1, marks + 1) + phase)
    idx = sorted({int(np.argmin(np.abs(u - t))) for t in targets})
    if ymin is not None:                          # no marker sitting on the axis
        idx = [i for i in idx if y[i] >= ymin]
    ax.plot(x, y, ls=ls, color=c, lw=lw or plt.rcParams["lines.linewidth"], marker=MARKS[k], markevery=idx,
            mfc="white", mec=c, mew=0.9, ms=3.8, label=label, **kw)
    return c


def method_handle(name, label, ms=4.6, line=False, ls="-"):
    from matplotlib.lines import Line2D
    c, m, filled = METHOD[name]
    return Line2D([], [], ls=ls if line else "none", color=c, marker=m, mfc=c if filled else "white", mec=c if not filled
                  else "white", mew=0.5 if filled else 0.9, ms=ms, label=label)


def save(fig, path, png=True, tight=False):
    """fixed size (the figure's own width, so every figure keeps the same type scale) unless tight=True."""
    kw = dict(bbox_inches="tight") if tight else dict(bbox_inches=None)
    fig.savefig(path + ".pdf", **kw)
    if png:
        fig.savefig(path + ".png", dpi=600, **kw)
    plt.close(fig)


def rowmajor(handles, ncol):
    """legend handles reordered so that matplotlib's column-major fill reads row by row."""
    n = len(handles)
    nrow = -(-n // ncol)
    rows = [handles[i * ncol:(i + 1) * ncol] for i in range(nrow)]
    return [rows[r][c] for c in range(ncol) for r in range(nrow) if c < len(rows[r])]


def constrained(fig, w_pad=3.0, h_pad=3.0, wspace=0.04, hspace=0.04):
    fig.get_layout_engine().set(w_pad=w_pad / 72, h_pad=h_pad / 72, wspace=wspace, hspace=hspace)


# ------------------------------------------------------------------------------------------- schematics
def box(ax, x, y, w, h, title, sub=None, role="neutral", strong=False, tfs=8, sfs=7, title_y=0.63, sub_y=0.28,
        lw=None):
    """square box: tinted fill, coloured border (1.6 pt when emphasised), bold title + grey subtitle."""
    c = ROLE.get(role, role)
    fc = tint(c, FILL_T_STRONG if strong else FILL_T)
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=c, lw=lw or (1.6 if strong else 0.8), joinstyle="miter",
                               zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * title_y, title, ha="center", va="center", fontsize=tfs, weight="bold", zorder=3)
        ax.text(x + w / 2, y + h * sub_y, sub, ha="center", va="center", fontsize=sfs, color=SUB, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tfs, weight="bold", zorder=3)


def arrow(ax, x1, y1, x2, y2, color=SUB, lw=0.9, ms=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=ms, shrinkA=0, shrinkB=0),
                zorder=1)


def canvas(w, h):
    """schematic canvas in millimetres: data units = mm, origin bottom left."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax
