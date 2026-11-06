#################################################################################
# Poster figure and summary table for the surrogate comparison
#
# Reads the CSV files written by compare_surrogate_models.py (figures/comparison)
# and writes poster-ready outputs there; no solves are needed, so it can be
# re-run quickly:
#
#   python poster_figures.py
#
# Outputs (figures/comparison):
#   comparison_percent_error.png  - flux error vs high-fidelity stage
#   comparison_poster_table.csv / .md / .png - summary table
#################################################################################
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

from figure_style import apply_style, savefig_kwargs, MARKER_SIZE, LINE_WIDTH

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "figures/comparison")

CATIONS = ["Na", "Ca", "La"]
STAGE_COLOR = "#0072B2"  # colorblind-safe pair
CHAIN_COLOR = "#D55E00"


def plot_percent_error(results, path=None):
    """Signed percent error for each condition; test points filled, training hollow."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.3}
    )
    rng = np.random.default_rng(0)
    models = (
        ("stage", "Stage", STAGE_COLOR, -0.18),
        ("chain", "Tube chain", CHAIN_COLOR, 0.18),
    )

    for model, name, color, offset in models:
        for k, cation in enumerate(CATIONS):
            for label, filled in (("training", False), ("test", True)):
                g = results[(results["cation"] == cation) & (results["set"] == label)]
                x = k + offset + rng.uniform(-0.07, 0.07, len(g))
                ax1.scatter(
                    x,
                    g[f"{model}_percent_error"],
                    s=MARKER_SIZE**2,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    linewidths=1.8,
                    alpha=0.7,
                )
    ax1.axhline(0, color="k", linewidth=1.2)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(CATIONS)
    ax1.set_ylabel("Error in Flux Prediction (%)")
    ax1.set_title("Surrogate Error for All Data", pad=10)
    grey = "0.35"
    ax1.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=STAGE_COLOR, markersize=MARKER_SIZE + 1, label="Stage"),
            Line2D([], [], marker="o", linestyle="", color=CHAIN_COLOR, markersize=MARKER_SIZE + 1, label="Tube chain"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=grey,
                   markeredgewidth=1.8, markersize=MARKER_SIZE + 1, label="Training"),
            Line2D([], [], marker="o", linestyle="", color=grey, markersize=MARKER_SIZE + 1, label="Test"),
        ],
        loc="upper left", fontsize=13, handletextpad=0.3,
    )

    # (b) held-out test points only: mean and max absolute error
    test = results[results["set"] == "test"]
    x = np.arange(len(CATIONS) + 1)
    groups = [test[test["cation"] == c] for c in CATIONS] + [test]
    for model, name, color, offset in models:
        err = [g[f"{model}_percent_error"].abs() for g in groups]
        ax2.bar(
            x + offset,
            [e.mean() for e in err],
            width=0.34,
            color=color,
            label=name,
        )
        ax2.scatter(
            x + offset,
            [e.max() for e in err],
            marker="_",
            s=450,
            color="k",
            linewidths=LINE_WIDTH,
            label="Maximum" if model == "stage" else None,
            zorder=3,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(CATIONS + ["All"])
    ax2.set_ylabel("Absolute Error (%)")
    ax2.set_title("Mean Surrogate Error for\nTest Data (n=6 per cation)", pad=10)
    ax2.set_ylim(0, 1.8)
    ax2.legend(loc="upper left", fontsize=13, handlelength=1.2, handletextpad=0.7)

    plt.savefig(path or f"{FOLDER}/comparison_percent_error.png", **savefig_kwargs())
    return fig


def poster_table(results, size, cost):
    """Side-by-side summary of the three representations of the 5-tube stage."""
    test = results[results["set"] == "test"]

    def err(model, subset=test):
        return subset[f"{model}_percent_error"].abs()

    def fmt_mape(model):
        parts = [f"{err(model, test[test['cation'] == c]).mean():.2f}" for c in CATIONS]
        return " / ".join(parts)

    hf = "High-fidelity stage"
    stage = "Stage surrogate"
    chain = "Tube surrogate chain"
    hf_time = results["hf_warm_solve_time_s"].median()

    n_tube = int(cost.iloc[0]["high-fidelity runs"])
    n_stage = int(cost.iloc[1]["high-fidelity runs"])
    t_tube = cost.iloc[0]["Kriging training time (s)"]
    t_stage = cost.iloc[1]["Kriging training time (s)"]

    rows = {
        "Variables": [f"{int(size.iloc[2, 0]):,}", f"{int(size.iloc[0, 0])}", f"{int(size.iloc[1, 0])}"],
        "Constraints": [f"{int(size.iloc[2, 1]):,}", f"{int(size.iloc[0, 1])}", f"{int(size.iloc[1, 1])}"],
        "Solve time per point (s)": [
            f"~{hf_time:.1f} (continuation move)",
            f"{results['stage_solve_time_s'].median():.3f}",
            f"{results['chain_solve_time_s'].median():.3f}",
        ],
        "Training HF runs (all data)": ["-", f"{n_stage} stage solves", f"{n_tube} tube solves"],
        "Training cost (tube-solve equiv.)": ["-", f"{n_stage * 5}", f"{n_tube}"],
        "Kriging training time, all data (s)": ["-", f"{t_stage:.1f}", f"{t_tube:.1f}"],
        "Reusable for other stage designs": ["-", "No", "Yes"],
    }
    return pd.DataFrame(rows, index=[hf, stage, chain]).T


def save_table(table):
    table.to_csv(f"{FOLDER}/comparison_poster_table.csv")
    with open(f"{FOLDER}/comparison_poster_table.md", "w") as f:
        header = "| | " + " | ".join(table.columns) + " |"
        f.write(header + "\n|" + "---|" * (len(table.columns) + 1) + "\n")
        for name, row in table.iterrows():
            f.write(f"| {name} | " + " | ".join(row) + " |\n")
        f.write(
            "\nTest set: 18 held-out high-fidelity stage conditions (6 per cation). "
            "Solve times are per evaluation point (one operating condition), medians over the comparison conditions; the HF stage is a continuation move (several solves). Training rows are totals for generating and fitting the all-data models. "
            "HF size counts unit models only.\n"
        )

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    t = ax.table(
        cellText=table.values,
        rowLabels=table.index,
        colLabels=table.columns,
        cellLoc="center",
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)
    t.auto_set_column_width(list(range(len(table.columns))))
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")
        if c == -1:
            cell.set_text_props(ha="right")
    plt.savefig(f"{FOLDER}/comparison_poster_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    results = pd.read_csv(f"{FOLDER}/comparison_results.csv")
    size = pd.read_csv(f"{FOLDER}/comparison_model_size.csv", index_col=0)
    cost = pd.read_csv(f"{FOLDER}/comparison_training_cost.csv", index_col=0)

    plot_percent_error(results)
    table = poster_table(results, size, cost)
    save_table(table)
    print(table.to_string())


if __name__ == "__main__":
    main()
