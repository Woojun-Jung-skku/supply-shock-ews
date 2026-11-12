"""Black-and-white figure style of the draft figures written by analysis_all.sh (fig6_final.py, fig7_final.py): Liberation Sans
(Arial metrics) 8 pt, 190 mm double column / 90 mm single column, line styles and markers instead of
colour, no top/right spines, no in-figure titles or notes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
DOUBLE, SINGLE = 190 * MM, 90 * MM
INK, GREY, LIGHT = "#1a1a1a", "#6b6b6b", "#bdbdbd"
# series styles: (colour, linestyle, marker, marker face)
STYLES = [("#1a1a1a", "-", "o", "#1a1a1a"), ("#1a1a1a", "--", "s", "white"), ("#6b6b6b", ":", "^", "white"),
          ("#6b6b6b", "-.", "D", "#6b6b6b"), ("#1a1a1a", (0, (1, 1)), "v", "white"), ("#8c8c8c", "-", "x", "#8c8c8c"),
          ("#1a1a1a", (0, (5, 2, 1, 2, 1, 2)), "P", "white")]


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 0.9,
        "lines.markersize": 3.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150})


def panel(ax, letter):
    """Panel letter in the manuscript's convention: (a), (b), ... top left, bold."""
    ax.text(-0.14, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, path):
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight", dpi=600)
    plt.close(fig)
