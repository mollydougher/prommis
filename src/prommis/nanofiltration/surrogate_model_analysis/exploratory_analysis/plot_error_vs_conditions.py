#################################################################################
# Where in (tube 1 feed concentration, tube 1 pressure) space are the stage-average
# flux errors of the stage surrogate and the tube chain?
#
# Reads figures/comparison/comparison_results.csv (compare_surrogate_models.py).
#   python plot_error_vs_conditions.py
#
# Outputs (figures/comparison):
#   error_map_by_conditions.png   - signed error on the concentration-pressure plane
#   error_vs_each_condition.png   - signed error vs concentration and vs pressure
#################################################################################
import os

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "figures/comparison")
CATIONS = ["Na", "Ca", "La"]
MODELS = (("stage", "Stage surrogate", "#0072B2"), ("chain", "Tube surrogate chain", "#D55E00"))


def error_map(results, path):
    """Signed % error on the concentration-pressure plane; one colour scale per cation."""
    fig, axes = plt.subplots(3, 2, figsize=(9, 11), constrained_layout=True)
    for i, cation in enumerate(CATIONS):
        g = results[results["cation"] == cation]
        limit = max(g["stage_percent_error"].abs().max(), g["chain_percent_error"].abs().max())
        for j, (model, name, _) in enumerate(MODELS):
            ax = axes[i, j]
            for training, marker, size in ((False, "o", 70), (True, "s", 50)):
                sub = g[(g["set"] == "training") == training]
                sc = ax.scatter(
                    sub["feed_conc"], sub["pressure"], c=sub[f"{model}_percent_error"],
                    cmap="RdBu_r", vmin=-limit, vmax=limit, s=size, marker=marker,
                    edgecolors="k" if training else "0.4", linewidths=1.2 if training else 0.4,
                )
            ax.set_title(f"{cation}: {name}")
            ax.set_xlabel("Tube 1 feed concentration (mol/m$^3$)")
            ax.set_ylabel("Tube 1 pressure (bar)")
            if j == 1:
                fig.colorbar(sc, ax=axes[i, :], label="Signed flux error vs. HF (%)", shrink=0.9)
    axes[0, 0].scatter([], [], marker="o", c="0.8", edgecolors="0.4", label="off-grid test")
    axes[0, 0].scatter([], [], marker="s", c="0.8", edgecolors="k", linewidths=1.2, label="training grid")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def error_vs_each(results, path):
    """Signed error against concentration (left) and pressure (right), per cation."""
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)
    for i, cation in enumerate(CATIONS):
        g = results[results["cation"] == cation]
        for j, (var, xlabel) in enumerate(
            (("feed_conc", "Tube 1 feed concentration (mol/m$^3$)"), ("pressure", "Tube 1 pressure (bar)"))
        ):
            ax = axes[i, j]
            for model, name, color in MODELS:
                for training in (True, False):
                    sub = g[(g["set"] == "training") == training]
                    ax.scatter(
                        sub[var], sub[f"{model}_percent_error"], s=28,
                        facecolors="none" if training else color, edgecolors=color,
                        label=f"{name}, {'training' if training else 'off-grid'}" if (i, j) == (0, 0) else None,
                    )
            ax.axhline(0, color="k", linewidth=0.8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Flux error vs. HF (%)")
            ax.set_title(cation)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    results = pd.read_csv(f"{FOLDER}/comparison_results.csv")
    error_map(results, f"{FOLDER}/error_map_by_conditions.png")
    error_vs_each(results, f"{FOLDER}/error_vs_each_condition.png")


if __name__ == "__main__":
    main()
