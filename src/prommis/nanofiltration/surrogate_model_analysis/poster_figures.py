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

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "figures/comparison")

CATIONS = ["Na", "Ca", "La"]
STAGE_COLOR = "#0072B2"  # colorblind-safe pair
CHAIN_COLOR = "#D55E00"


def plot_percent_error(results, path=None):
    """Signed percent error for each condition; test points filled, training hollow."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    rng = np.random.default_rng(0)
    models = (
        ("stage", "Stage surrogate", STAGE_COLOR, -0.18),
        ("chain", "Tube surrogate chain", CHAIN_COLOR, 0.18),
    )

    for model, name, color, offset in models:
        for k, cation in enumerate(CATIONS):
            for label, filled in (("training", False), ("test", True)):
                g = results[(results["cation"] == cation) & (results["set"] == label)]
                x = k + offset + rng.uniform(-0.07, 0.07, len(g))
                ax1.scatter(
                    x,
                    g[f"{model}_percent_error"],
                    s=40,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    linewidths=1.3,
                    label=f"{name}, {label}" if k == 0 else None,
                )
    ax1.axhline(0, color="k", linewidth=0.8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(CATIONS)
    ax1.set_ylabel("Stage-average flux error vs. HF (%)")
    ax1.set_title("(a) Signed error at each condition")
    ax1.legend(fontsize=8, loc="best")

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
            label=f"{name} (mean)",
        )
        ax2.scatter(
            x + offset,
            [e.max() for e in err],
            marker="_",
            s=300,
            color="k",
            linewidths=2,
            label="Maximum" if model == "stage" else None,
            zorder=3,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(CATIONS + ["All"])
    ax2.set_ylabel("Absolute error, held-out test points (%)")
    ax2.set_title("(b) Held-out test error (n = 6 per cation)")
    ax2.set_ylim(0, 2.5)
    ax2.legend(fontsize=8, loc="upper left", ncol=1)

    plt.tight_layout()
    plt.savefig(path or f"{FOLDER}/comparison_percent_error.png", dpi=600)
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
        "Test MAPE, all cations (%)": ["-", f"{err('stage').mean():.2f}", f"{err('chain').mean():.2f}"],
        "Test MAPE, Na / Ca / La (%)": ["-", fmt_mape("stage"), fmt_mape("chain")],
        "Test maximum error (%)": ["-", f"{err('stage').max():.1f}", f"{err('chain').max():.1f}"],
        "Variables": [f"{int(size.iloc[2, 0]):,}", f"{int(size.iloc[0, 0])}", f"{int(size.iloc[1, 0])}"],
        "Constraints": [f"{int(size.iloc[2, 1]):,}", f"{int(size.iloc[0, 1])}", f"{int(size.iloc[1, 1])}"],
        "Solve time (s)": [
            f"~{hf_time:.1f} (continuation move)",
            f"{results['stage_solve_time_s'].median():.3f}",
            f"{results['chain_solve_time_s'].median():.3f}",
        ],
        "Training HF runs": ["-", f"{n_stage} stage solves", f"{n_tube} tube solves"],
        "Training cost (tube-solve equiv.)": ["-", f"{n_stage * 5}", f"{n_tube}"],
        "Kriging training time (s)": ["-", f"{t_stage:.1f}", f"{t_tube:.1f}"],
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
            "Solve time of the HF stage is a continuation move (several solves). "
            "HF size counts unit models only.\n"
        )

    fig, ax = plt.subplots(figsize=(11, 4.2))
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
