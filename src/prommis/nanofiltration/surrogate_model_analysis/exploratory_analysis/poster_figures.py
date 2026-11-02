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
OFF_GRID = ["test", "test_large"]  # points not on the training grid
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
                in_set = results["set"].isin(OFF_GRID) if filled else results["set"] == label
                g = results[(results["cation"] == cation) & in_set]
                x = k + offset + rng.uniform(-0.07, 0.07, len(g))
                ax1.scatter(
                    x,
                    g[f"{model}_percent_error"],
                    s=40,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    linewidths=1.3,
                    alpha=0.7,
                    label=f"{name}, {label}" if k == 0 else None,
                )
    ax1.axhline(0, color="k", linewidth=0.8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(CATIONS)
    ax1.set_ylabel("Stage-average flux error vs. HF (%)")
    ax1.set_title("(a) Signed error at each condition")
    ax1.legend(fontsize=8, loc="best")

    # (b) held-out test points only: mean and max absolute error
    test = results[results["set"].isin(OFF_GRID)]
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
    n = int(test.groupby("cation").size().min())
    ax2.set_title(f"(b) Off-grid test error (n \u2265 {n} per cation)")
    ax2.set_ylim(0, 2.5)
    ax2.legend(fontsize=8, loc="upper left", ncol=1)

    plt.tight_layout()
    plt.savefig(path or f"{FOLDER}/comparison_percent_error.png", dpi=600)
    return fig


STAGE_PRESSURES = {"Na": (5, 6, 7), "Ca": (8, 10, 12), "La": (12, 14, 16)}  # stage training grid


def plot_error_mechanism(results, details, path=None, tube_row=True):
    """Where each model's error comes from.

    Top row: signed stage-average flux error against tube 1 pressure, with the
    stage surrogate's training pressures marked. Bottom row (tube_row=True): tube-level
    flux error by tube position, for the tube surrogates inside the chain (which
    inherit upstream errors) and at the high-fidelity inputs (their own error only).
    """
    nrows = 2 if tube_row else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(11, 3.6 * nrows), squeeze=False)
    rng = np.random.default_rng(1)
    for k, cation in enumerate(CATIONS):
        ax = axes[0, k]
        for pressure in STAGE_PRESSURES[cation]:
            ax.axvline(pressure, color="0.8", linestyle=":", linewidth=1, zorder=0)
        g = results[results["cation"] == cation]
        for model, name, color, _ in (
            ("stage", "Stage surrogate", STAGE_COLOR, 0),
            ("chain", "Tube surrogate chain", CHAIN_COLOR, 0),
        ):
            for filled in (False, True):
                sub = g[g["set"].isin(OFF_GRID)] if filled else g[g["set"] == "training"]
                ax.scatter(
                    sub["pressure"], sub[f"{model}_percent_error"], s=26, alpha=0.7,
                    facecolors=color if filled else "none", edgecolors=color, linewidths=1.2,
                    label=f"{name}, {'off-grid' if filled else 'training'}" if k == 0 else None,
                )
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(cation)
        ax.set_xlabel("Tube 1 pressure (bar)")
        if k == 0:
            ax.set_ylabel("Flux error vs. HF (%)")
    axes[0, 0].legend(fontsize=6.5, loc="best")

    if tube_row:
        for k, cation in enumerate(CATIONS):
            ax = axes[1, k]
            d = details[details["cation"] == cation]
            tubes = sorted(d["tube"].unique())
            ax.plot(tubes, [d[d["tube"] == t]["flux_percent_error"].abs().mean() for t in tubes],
                    "-o", color=CHAIN_COLOR, label="In chain")
            ax.plot(tubes, [d[d["tube"] == t]["isolated_flux_percent_error"].abs().mean() for t in tubes],
                    "--s", color=CHAIN_COLOR, mfc="none", label="At HF inputs")
            ax.set_xticks(tubes)
            ax.set_xlabel("Tube position in stage")
            if k == 0:
                ax.set_ylabel("Tube flux error, mean absolute (%)")
        axes[1, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path or f"{FOLDER}/comparison_error_mechanism.png", dpi=600)
    plt.close(fig)


def plot_training_vs_offgrid(results, path=None):
    """Mean (bar) and maximum (tick) absolute error, training grid vs. off-grid."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    subsets = (
        ("Training grid (in-sample)", results[results["set"] == "training"]),
        ("Off-grid test", results[results["set"].isin(OFF_GRID)]),
    )
    x = np.arange(len(CATIONS) + 1)
    for ax, (title, sub) in zip(axes, subsets):
        groups = [sub[sub["cation"] == c] for c in CATIONS] + [sub]
        for model, name, color, offset in (
            ("stage", "Stage surrogate", STAGE_COLOR, -0.18),
            ("chain", "Tube surrogate chain", CHAIN_COLOR, 0.18),
        ):
            err = [g[f"{model}_percent_error"].abs() for g in groups]
            means = [e.mean() for e in err]
            ax.bar(x + offset, means, width=0.34, color=color, label=f"{name} (mean)")
            ax.scatter(x + offset, [e.max() for e in err], marker="_", s=300, color="k",
                       linewidths=2, zorder=3, label="Maximum" if model == "stage" else None)
            for xi, m, e in zip(x + offset, means, err):
                # put the label above the maximum tick when the two would overlap
                y = max(m, e.max()) + 0.03 if e.max() - m < 0.08 else m + 0.03
                ax.text(xi, y, f"{m:.2f}", ha="center", va="bottom", fontsize=7)
        n = [len(g) for g in groups]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={k})" for c, k in zip(CATIONS + ["All"], n)])
        ax.set_title(title)
    axes[0].set_ylabel("Stage-average flux error, absolute (%)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_ylim(0, 2.5)
    plt.tight_layout()
    plt.savefig(path or f"{FOLDER}/comparison_error_training_vs_offgrid.png", dpi=600)
    plt.close(fig)


def poster_table(results, size, cost):
    """Side-by-side summary of the three representations of the 5-tube stage."""
    test = results[results["set"].isin(OFF_GRID)]

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

    hf_name = [i for i in size.index if i.startswith("High-fidelity")]
    if hf_name:
        hf_vars, hf_cons = f"{int(size.loc[hf_name[0], 'variables']):,}", f"{int(size.loc[hf_name[0], 'constraints']):,}"
    else:  # the HF model could not be built here (prommis not importable)
        hf_vars = hf_cons = "n/a"
    rows = {
        "Test MAPE, all cations (%)": ["-", f"{err('stage').mean():.2f}", f"{err('chain').mean():.2f}"],
        "Test MAPE, Na / Ca / La (%)": ["-", fmt_mape("stage"), fmt_mape("chain")],
        "Test maximum error (%)": ["-", f"{err('stage').max():.1f}", f"{err('chain').max():.1f}"],
        "Variables": [hf_vars, f"{int(size.loc['Stage surrogate', 'variables'])}", f"{int(size.loc['Tube surrogate chain', 'variables'])}"],
        "Constraints": [hf_cons, f"{int(size.loc['Stage surrogate', 'constraints'])}", f"{int(size.loc['Tube surrogate chain', 'constraints'])}"],
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
    table = pd.DataFrame(rows, index=[hf, stage, chain]).T
    table.attrs["n_test"] = len(test)
    return table


def save_table(table):
    table.to_csv(f"{FOLDER}/comparison_poster_table.csv")
    with open(f"{FOLDER}/comparison_poster_table.md", "w") as f:
        header = "| | " + " | ".join(table.columns) + " |"
        f.write(header + "\n|" + "---|" * (len(table.columns) + 1) + "\n")
        for name, row in table.iterrows():
            f.write(f"| {name} | " + " | ".join(row) + " |\n")
        f.write(
            f"\nTest set: {table.attrs.get('n_test', '?')} off-grid high-fidelity stage conditions "
            f"(not on the training grid). "
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
    plot_training_vs_offgrid(results)
    per_tube = f"{FOLDER}/tube_chain_error_per_tube.csv"  # from analyze_tube_chain_error.py
    if os.path.exists(per_tube):
        plot_error_mechanism(results, pd.read_csv(per_tube))
    table = poster_table(results, size, cost)
    save_table(table)
    print(table.to_string())


if __name__ == "__main__":
    main()
