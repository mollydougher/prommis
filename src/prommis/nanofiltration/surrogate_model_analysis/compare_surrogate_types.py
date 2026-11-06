"""
Compares surrogate model types (polynomial, Kriging, six RBFs) for the tube and the
stage over repeated random 80/20 training/validation splits.

For every split each model is trained on 80% of the data and evaluated on the other
20% (same split for every model). Reported metrics are percent based so that the tube
and the stage can be compared:
    MAPE   mean absolute percent error
    maxAPE maximum absolute percent error
    nRMSE  root-mean-square error as a percent of the mean absolute value
For the tube the water flux (back-transformed from the log) is also evaluated.
Bars show the mean over splits; error bars are the standard deviation over splits
(the splits overlap, so this is a spread, not a standard error).

Run from this directory:
    python compare_surrogate_types.py                    # 10 splits, tube and stage
    python compare_surrogate_types.py --splits 20
    python compare_surrogate_types.py --scales stage --models "Kriging" "RBF (IMQ)"
    python compare_surrogate_types.py --plot-only        # redraw from the saved CSVs

The run is resumable: finished (scale, model, split) results are kept in
figures/surrogate_types/surrogate_type_metrics_all_splits.csv.

Author: Claude
"""

import argparse
import contextlib
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import apply_style, savefig_kwargs

import train_surrogate_model as tsm

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(HERE, "figures/surrogate_types")
ALL_SPLITS_FILE = os.path.join(OUTPUT_FOLDER, "surrogate_type_metrics_all_splits.csv")
SUMMARY_FILE = os.path.join(OUTPUT_FOLDER, "surrogate_type_metrics_summary.csv")

N_SPLITS = 10
MODELS = [
    "Polynomial", "Kriging", "RBF (Linear)", "RBF (Cubic)", "RBF (Spline)",
    "RBF (Gaussian)", "RBF (MQ)", "RBF (IMQ)",
]
POSTER_MODELS = ["Kriging", "RBF (IMQ)", "RBF (Gaussian)"]
SCALES = ("tube", "stage")
METRICS = ("MAPE", "maxAPE", "nRMSE")
METRIC_LABELS = {"MAPE": "MAPE (%)", "maxAPE": "maxAPE (%)", "nRMSE": "RMSE (% of mean flux)"}
COLORS = {  # colour-blind-safe
    "Kriging": "#0072B2", "RBF (IMQ)": "#E69F00", "RBF (Gaussian)": "#009E73",
    "Polynomial": "#7F7F7F", "RBF (Linear)": "#CC79A7", "RBF (Cubic)": "#D55E00",
    "RBF (Spline)": "#56B4E9", "RBF (MQ)": "#7B3294",
}


def percent_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    error = np.asarray(predicted, dtype=float) - actual
    return {
        "MAPE": 100 * np.mean(np.abs(error) / np.abs(actual)),
        "maxAPE": 100 * np.max(np.abs(error) / np.abs(actual)),
        "nRMSE": 100 * np.sqrt(np.mean(error**2)) / np.mean(np.abs(actual)),
    }


def train_model(name, datafile, scale, seed):
    """Trains one model type on the 80% split with this seed; returns the surrogate and time."""
    tsm.TRAINING_TIMES.clear()
    quiet = io.StringIO()  # the PySMO trainers and assess_surrogate print a lot
    with contextlib.redirect_stdout(quiet):
        if name == "Polynomial":
            surrogate, _ = tsm.train_polynomial_surrogate_model(datafile, scale, seed=seed)
        elif name == "Kriging":
            surrogate, _ = tsm.train_kriging_surrogate_model(datafile, scale, seed=seed)
        else:
            basis = name[name.index("(") + 1 : -1].lower()
            surrogate, _ = tsm.train_rbf_surrogate_model(
                datafile, scale, basis_func=basis, seed=seed
            )
    return surrogate, float(sum(tsm.TRAINING_TIMES.values()))


def run_splits(scales, models, n_splits):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    rows = pd.read_csv(ALL_SPLITS_FILE).to_dict("records") if os.path.exists(ALL_SPLITS_FILE) else []
    done = {(r["scale"], r["model"], int(r["split"])) for r in rows}

    for scale in scales:
        datafile = os.path.join(HERE, f"training_data/{scale}/single_salt/mixed_chlorides")
        data, input_labels, _, _ = tsm.prepare_training_data(datafile, scale)
        for split in range(n_splits):
            _, validation = tsm.split_data(data, input_labels, seed=split)
            for name in models:
                if (scale, name, split) in done:
                    continue
                surrogate, seconds = train_model(name, datafile, scale, split)
                predicted = surrogate.evaluate_surrogate(validation[input_labels])
                outputs = {"ion_flux": (validation["ion_flux"], predicted["ion_flux"])}
                if "log_water_flux" in predicted:
                    outputs["water_flux"] = (
                        validation["water_flux"], np.exp(predicted["log_water_flux"])
                    )
                for output, (actual, pred) in outputs.items():
                    for metric, value in percent_metrics(actual, pred).items():
                        rows.append({
                            "scale": scale, "model": name, "split": split, "output": output,
                            "metric": metric, "value": value, "n_validation": len(validation),
                            "training_time_s": seconds,
                        })
                pd.DataFrame(rows).to_csv(ALL_SPLITS_FILE, index=False)  # resumable
                print(f"{scale} split {split} {name}: {seconds:.1f} s", flush=True)
    return pd.DataFrame(rows)


def summarize(all_splits):
    summary = (
        all_splits.groupby(["scale", "model", "output", "metric"])["value"]
        .agg(mean="mean", std="std", n_splits="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary.to_csv(SUMMARY_FILE, index=False)
    return summary


def plot_grouped_bars(summary, models, output, scales, path, title, figsize, poster=False):
    """Grouped bars: metric on the x-axis, colour = model, filled = tube, hatched = stage."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    n_bars = len(models) * len(scales)
    width = 0.8 / n_bars
    for i_metric, metric in enumerate(METRICS):
        for i_model, model in enumerate(models):
            for i_scale, scale in enumerate(scales):
                row = summary[
                    (summary["scale"] == scale) & (summary["model"] == model)
                    & (summary["output"] == output) & (summary["metric"] == metric)
                ]
                if row.empty:
                    continue
                mean, std = float(row["mean"].iloc[0]), float(row["std"].iloc[0])
                x = i_metric - 0.4 + width * (i_model * len(scales) + i_scale + 0.5)
                filled = scale == "tube"
                ax.bar(
                    x, mean, width * 0.92,
                    color=COLORS[model] if filled else "white",
                    edgecolor=COLORS[model], hatch=None if filled else "////", linewidth=2,
                )
                # keep the lower error bar on the log axis (spread can exceed the mean)
                ax.errorbar(x, mean, yerr=[[min(std, 0.9 * mean)], [std]], color="k",
                            capsize=3, linewidth=1.5, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_ylabel("Validation Error")
    if title:
        ax.set_title(title, pad=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in models]
    labels = list(models)
    if len(scales) == 2:
        handles += [
            plt.Rectangle((0, 0), 1, 1, facecolor="0.4", edgecolor="0.4"),
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.4", hatch="////"),
        ]
        labels += ["Tube (filled)", "Stage (hatched)"]
    if len(models) <= 4:  # poster version: legend inside the axes, in headroom above the bars
        ax.legend(handles, labels, fontsize=13, ncol=1, loc="upper left")
    else:
        ax.legend(handles, labels, fontsize=10, ncol=2, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    n_splits = int(summary["n_splits"].max())
    note = f"Mean and standard deviation over {n_splits} random 80/20 splits"
    if poster:
        ax.set_title(f"Mean Surrogate Error for {n_splits} Random 80/20 Splits", fontsize=14,
                     fontweight="bold", pad=10)
    else:
        ax.annotate(note + ".", xy=(0, -0.12), xycoords="axes fraction", fontsize=10, ha="left", va="top")
    plt.savefig(path, **savefig_kwargs())
    plt.close(fig)


def make_figures(summary):
    available = set(summary["model"])
    poster = [m for m in POSTER_MODELS if m in available]
    master = [m for m in MODELS if m in available]
    plot_grouped_bars(
        summary, poster, "ion_flux", SCALES,
        f"{OUTPUT_FOLDER}/surrogate_type_error_metrics_poster.png",
        None, (9, 5.5), poster=True,
    )
    plot_grouped_bars(
        summary, master, "ion_flux", SCALES,
        f"{OUTPUT_FOLDER}/surrogate_type_error_metrics_all_models.png",
        "Surrogate type comparison: cation flux (all models)", (13, 5.5),
    )
    if (summary["output"] == "water_flux").any():
        plot_grouped_bars(
            summary, master, "water_flux", ("tube",),
            f"{OUTPUT_FOLDER}/surrogate_type_error_metrics_water_flux_tube.png",
            "Tube surrogates: water flux (supplementary)", (10, 4.8),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--scales", nargs="+", default=list(SCALES), choices=SCALES)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        all_splits = pd.read_csv(ALL_SPLITS_FILE)
    else:
        all_splits = run_splits(args.scales, args.models, args.splits)
    summary = summarize(all_splits)
    print(summary[summary["output"] == "ion_flux"].round(3).to_string())
    make_figures(summary)


if __name__ == "__main__":
    main()
