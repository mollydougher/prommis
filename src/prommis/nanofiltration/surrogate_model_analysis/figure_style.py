#################################################################################
# Shared publication-quality matplotlib style
#
# Follows the guidelines in the Notre Dame "Publication-Quality Figures" notebook
# (https://ndcbe.github.io/data-and-computing/notebooks/01/Publication-Quality-Figures.html):
#   figure DPI 300; line width 3; marker size 8; tick labels 15 pt with ticks
#   pointing inward on all four sides; axis labels 16 pt bold; legend 10 pt;
#   colour-blind-safe colours.
# Call apply_style() before building a figure; save with savefig_kwargs().
#################################################################################
import matplotlib.pyplot as plt

TICK_SIZE = 15
LABEL_SIZE = 16
LEGEND_SIZE = 10
LINE_WIDTH = 3
MARKER_SIZE = 8
DPI = 300


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.labelweight": "bold",
            "axes.titlesize": LABEL_SIZE,
            "axes.titleweight": "bold",
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": False,
            "legend.fontsize": LEGEND_SIZE,
            "legend.frameon": False,
            "axes.linewidth": 1.2,
        }
    )


def savefig_kwargs():
    return {"dpi": DPI, "bbox_inches": "tight"}
