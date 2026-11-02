#################################################################################
# Does the tube chain's error come from the tube surrogates themselves, or does it
# compound along the chain?
#
# For every comparison condition (training and test) and every tube we compare
#   isolated: the tube surrogate evaluated at the HIGH-FIDELITY inlet conditions
#             of that tube (flow, concentration, pressure). Only the surrogate's
#             own error; nothing is inherited from upstream tubes.
#   chain:    the tube surrogate inside the flowsheet, which sees the inlet flow
#             and concentration predicted by the upstream tubes (mass balances).
# chain error - isolated error is what the chain adds (compounding, or
# cancellation if negative). The same split is done for the stage-average flux
# and compared with the stage surrogate.
#
# Requires the outputs of compare_surrogate_models.py (figures/comparison).
#   python analyze_tube_chain_error.py
#################################################################################
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "figures/comparison")
TUBE_SURROGATE_FILE = os.path.join(
    HERE, "surrogate_models/tube/single_salt/single_salt_kriging_surrogate"
)
VALENCY = {"Na": 1, "Ca": 2, "La": 3}
CATIONS = ["Na", "Ca", "La"]
COLORS = {"Na": "#0072B2", "Ca": "#E69F00", "La": "#009E73"}
STAGE_COLOR = "#0072B2"
CHAIN_COLOR = "#D55E00"
ISOLATED_COLOR = "#999999"
OFF_GRID = ["test", "test_large"]  # points not on the training grid


def pct(pred, ref):
    return 100 * (pred - ref) / ref


def isolated_tube_predictions(details, surrogate):
    """Tube surrogate at the high-fidelity inlet conditions of each tube."""
    # hf_inlet_conc is the retentate state after mixing, same as the tube feed
    inputs = pd.DataFrame(
        {
            "valency": details["cation"].map(VALENCY),
            "feed_conc": details["hf_inlet_conc"],
            "pressure": details["pressure"] - 0.5 * (details["tube"] - 1),
            "feed_flow": details["hf_inlet_flow"],
        }
    )
    out = surrogate.evaluate_surrogate(inputs)
    return out["ion_flux"].to_numpy(), np.exp(out["log_water_flux"].to_numpy())


def main():
    details = pd.read_csv(f"{FOLDER}/comparison_tube_details.csv")
    results = pd.read_csv(f"{FOLDER}/comparison_results.csv")
    surrogate = PysmoSurrogate.load_from_file(TUBE_SURROGATE_FILE)

    iso_flux, iso_water = isolated_tube_predictions(details, surrogate)
    details["isolated_flux"] = iso_flux
    details["isolated_water_flux"] = iso_water
    details["isolated_flux_percent_error"] = pct(iso_flux, details["hf_flux"])
    details["isolated_water_flux_percent_error"] = pct(iso_water, details["hf_water_flux"])
    # change in flux error caused by the upstream tubes (signed, percentage points)
    details["compounding_flux_pp"] = (
        details["flux_percent_error"] - details["isolated_flux_percent_error"]
    )

    # ---- stage level: stage-average flux = mean of the tube averages
    key = ["set", "cation", "feed_conc", "pressure"]
    stage = details.groupby(key).agg(
        hf_flux_from_tubes=("hf_flux", "mean"),
        isolated_flux=("isolated_flux", "mean"),
        chain_flux_from_tubes=("chain_flux", "mean"),
    ).reset_index()
    stage = stage.merge(results, on=key, how="left")
    print(
        "check: HF stage flux vs mean of HF tube fluxes, max relative diff = "
        f"{(abs(stage['hf_flux'] - stage['hf_flux_from_tubes']) / stage['hf_flux']).max():.2e}; "
        "chain stage flux vs mean of chain tube fluxes = "
        f"{(abs(stage['chain_flux'] - stage['chain_flux_from_tubes']) / stage['chain_flux']).max():.2e}"
    )
    stage["isolated_percent_error"] = pct(stage["isolated_flux"], stage["hf_flux"])
    for col in ("stage", "chain", "isolated"):
        stage[f"{col}_abs_error"] = stage[f"{col}_percent_error"].abs()
    # where the stage surrogate beats the chain, what explains the chain's error?
    stage["stage_better"] = stage["stage_abs_error"] < stage["chain_abs_error"]
    stage["isolated_beats_stage"] = stage["isolated_abs_error"] < stage["stage_abs_error"]

    details.to_csv(f"{FOLDER}/tube_chain_error_per_tube.csv", index=False)
    cols = key + [
        "hf_flux", "stage_percent_error", "chain_percent_error",
        "isolated_percent_error", "stage_better", "isolated_beats_stage",
    ]
    stage[cols].to_csv(f"{FOLDER}/tube_chain_error_stage_level.csv", index=False)

    # ---- summaries
    by_tube = details.groupby(["cation", "tube"]).agg(
        inlet_flow_abs_pct=("inlet_flow_percent_error", lambda x: x.abs().mean()),
        inlet_conc_abs_pct=("inlet_conc_percent_error", lambda x: x.abs().mean()),
        isolated_flux_MAPE=("isolated_flux_percent_error", lambda x: x.abs().mean()),
        chain_flux_MAPE=("flux_percent_error", lambda x: x.abs().mean()),
        isolated_water_MAPE=("isolated_water_flux_percent_error", lambda x: x.abs().mean()),
        chain_water_MAPE=("water_flux_percent_error", lambda x: x.abs().mean()),
    )
    by_tube.to_csv(f"{FOLDER}/tube_chain_error_by_tube.csv")
    print("\nPer-tube mean absolute percent error (all 45 conditions)")
    print(by_tube.to_string(float_format=lambda x: f"{x:.3f}"))

    summary = stage.groupby(["set", "cation"]).agg(
        n=("stage_abs_error", "size"),
        stage_surrogate_MAPE=("stage_abs_error", "mean"),
        chain_MAPE=("chain_abs_error", "mean"),
        isolated_tubes_MAPE=("isolated_abs_error", "mean"),
        n_stage_better_than_chain=("stage_better", "sum"),
        n_isolated_beats_stage=("isolated_beats_stage", "sum"),
    )
    summary.to_csv(f"{FOLDER}/tube_chain_error_summary.csv")
    print("\nStage-average flux error (%)")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    sb = stage[stage["stage_better"]]
    print(
        f"\nStage surrogate beats the chain in {len(sb)} of {len(stage)} conditions. "
        f"In those: mean |error| stage {sb['stage_abs_error'].mean():.3f}%, "
        f"chain {sb['chain_abs_error'].mean():.3f}%, "
        f"tube surrogates at HF inputs {sb['isolated_abs_error'].mean():.3f}%."
    )
    plot(details, stage)


def plot(details, stage):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) per-tube error: chain vs isolated
    ax = axes[0]
    for cation in CATIONS:
        g = details[details["cation"] == cation]
        t = sorted(g["tube"].unique())
        chain = [g[g["tube"] == i]["flux_percent_error"].abs().mean() for i in t]
        iso = [g[g["tube"] == i]["isolated_flux_percent_error"].abs().mean() for i in t]
        ax.plot(t, chain, "-o", color=COLORS[cation], label=f"{cation}, in chain")
        ax.plot(t, iso, "--s", color=COLORS[cation], mfc="none", label=f"{cation}, HF inputs")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Tube position in stage")
    ax.set_ylabel("Tube ion-flux error, mean absolute (%)")
    ax.set_title("(a) Tube error: in chain vs. at HF inputs")
    ax.legend(fontsize=7, ncol=2)

    # (b) what the upstream tubes pass down: inlet flow / concentration error
    ax = axes[1]
    for cation in CATIONS:
        g = details[(details["cation"] == cation) & (details["tube"] > 1)]
        t = sorted(g["tube"].unique())
        ax.plot(t, [g[g["tube"] == i]["inlet_flow_percent_error"].abs().mean() for i in t],
                "-o", color=COLORS[cation], label=f"{cation} flow")
        ax.plot(t, [g[g["tube"] == i]["inlet_conc_percent_error"].abs().mean() for i in t],
                "--^", color=COLORS[cation], mfc="none", label=f"{cation} conc.")
    ax.set_xticks(range(2, 6))
    ax.set_xlabel("Tube position in stage")
    ax.set_ylabel("Error in tube inlet vs. HF, mean absolute (%)")
    ax.set_title("(b) Error inherited from upstream tubes")
    ax.legend(fontsize=7, ncol=2)

    # (c) stage level, test points only
    ax = axes[2]
    test = stage[stage["set"].isin(OFF_GRID)].copy()
    if len(test) > 36:  # a large pool: distributions instead of one marker per point
        return plot_distributions(ax, test)
    test["order"] = test["cation"].map({c: i for i, c in enumerate(CATIONS)})
    test = test.sort_values(["order", "pressure", "feed_conc"]).reset_index(drop=True)
    x = np.arange(len(test))
    ax.scatter(x, test["stage_abs_error"], marker="o", color=STAGE_COLOR, label="Stage surrogate", zorder=3)
    ax.scatter(x, test["chain_abs_error"], marker="s", color=CHAIN_COLOR, label="Tube chain", zorder=3)
    ax.scatter(x, test["isolated_abs_error"], marker="D", facecolors="none",
               edgecolors="k", label="Tube surrogates at HF inputs", zorder=3)
    for i in range(len(test)):
        ax.plot([x[i]] * 2, [test["isolated_abs_error"][i], test["chain_abs_error"][i]],
                color="0.7", zorder=1)
    for k, c in enumerate(CATIONS):
        idx = test.index[test["cation"] == c]
        ax.text(float(np.mean(idx)), -0.04, c, ha="center", va="top", transform=ax.get_xaxis_transform())
        if k:
            ax.axvline(idx.min() - 0.5, color="0.85", linewidth=0.8)
    ax.set_xticks([])
    ax.set_ylabel("Stage-average flux error, absolute (%)")
    ax.set_title("(c) Held-out test conditions")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FOLDER}/tube_chain_error_analysis.png", dpi=600, bbox_inches="tight")


def plot_distributions(ax, test):
    """Panel (c) for a large off-grid pool: box plots by cation."""
    series = (
        ("stage_abs_error", "Stage surrogate", STAGE_COLOR),
        ("chain_abs_error", "Tube chain", CHAIN_COLOR),
        ("isolated_abs_error", "Tube surrogates at HF inputs", ISOLATED_COLOR),
    )
    for j, (col, name, color) in enumerate(series):
        data = [test[test["cation"] == c][col] for c in CATIONS]
        bp = ax.boxplot(
            data, positions=np.arange(3) + (j - 1) * 0.26, widths=0.22,
            patch_artist=True, showfliers=True, flierprops={"markersize": 3},
            medianprops={"color": "k"},
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.8)
        ax.plot([], [], "s", color=color, label=name)
    ax.set_xticks(range(3))
    n = test.groupby("cation").size().min()
    ax.set_xticklabels(CATIONS)
    ax.set_ylabel("Stage-average flux error, absolute (%)")
    ax.set_title(f"(c) Off-grid test conditions (n \u2265 {n} per cation)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FOLDER}/tube_chain_error_analysis.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
