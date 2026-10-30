"""
Surrogate model training for multi-component diafiltration model.

The tube surrogates have two outputs: the tube-average cation flux (ion_flux)
and the log of the tube-average water flux (log_water_flux). The water flux is
modeled in log space because it spans more than an order of magnitude over the
training data; error metrics for water flux are reported after transforming the
predictions back to linear units (m3/m2/h). The stage surrogates have a single
output, the stage-average cation flux.

Author: Molly Dougher, Claude
"""

import os
import time

import numpy as np

from pandas import DataFrame, read_csv

from idaes.core.surrogate.metrics import compute_fit_metrics
from idaes.core.surrogate.pysmo_surrogate import (
    PysmoPolyTrainer,
    PysmoSurrogate,
    PysmoKrigingTrainer,
    PysmoRBFTrainer,
)
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.plotting.sm_plotter import (
    surrogate_scatter2D,
    surrogate_scatter3D,
    surrogate_parity,
    surrogate_residual,
)

from pyomo.environ import value

import matplotlib.pyplot as plt

RBF_BASIS_FUNCTIONS = ["linear", "cubic", "spline", "gaussian", "mq", "imq"]
TRAINING_TIMES = {}  # wall-clock seconds spent in the PySMO trainer, by model name
STAGE_TUBE_CONDITIONS_FILE = (
    "training_data/stage/single_salt/mixed_chlorides_tube_conditions"
)


def main():
    # generate_per_salt_analysis(solute="Na", salt="Sodium Chloride", length_scale="tube")
    # generate_per_salt_analysis(
    #     solute="Ca", salt="Calcium Chloride", length_scale="tube"
    # )
    # generate_per_salt_analysis(
    #     solute="La", salt="Lanthanum Chloride", length_scale="tube"
    # )

    generate_mixed_salt_analysis(length_scale="tube")

    # generate_per_salt_analysis(
    #     solute="Na", salt="Sodium Chloride", length_scale="stage"
    # )
    # generate_per_salt_analysis(
    #     solute="Ca", salt="Calcium Chloride", length_scale="stage"
    # )
    # generate_per_salt_analysis(
    #     solute="La", salt="Lanthanum Chloride", length_scale="stage"
    # )

    generate_mixed_salt_analysis(length_scale="stage")


def generate_per_salt_analysis(solute, salt, length_scale):
    datafile = f"training_data/{length_scale}/single_salt/{solute}_chloride"

    testing_metrics = train_and_save_all_surrogates(
        datafile, length_scale, save_prefix=f"{solute}Cl"
    )

    plot_solute_testing_error_metrics(
        length_scale,
        solute,
        salt,
        testing_metrics,
        drop_r2=True,
        save_figure=True,
    )

    plt.show()


def generate_mixed_salt_analysis(length_scale):
    datafile = f"training_data/{length_scale}/single_salt/mixed_chlorides"

    testing_metrics, surrogates = train_and_save_all_surrogates(
        datafile, length_scale, save_prefix="single_salt", return_surrogates=True
    )

    plot_single_salt_testing_error_metrics(
        length_scale,
        testing_metrics,
        drop_r2=True,
        drop_sse=True,
        save_figure=True,
    )

    if length_scale == "tube":
        # independent test: the tube conditions encountered in the 5-tube stage
        validate_tube_surrogate_on_stage_conditions(surrogates["Kriging"])

    plt.show()


def train_and_save_all_surrogates(
    datafile, length_scale, save_prefix, return_surrogates=False
):
    """
    Trains a polynomial, a Kriging, and six RBF surrogates, saves them, and
    returns their testing error metrics (and surrogates) keyed by model name.
    """
    surrogates = {}
    testing_metrics = {}

    surrogates["Polynomial"], testing_metrics["Polynomial"] = (
        train_polynomial_surrogate_model(
            datafile=datafile, length_scale=length_scale, visualize=False
        )
    )
    surrogates["Polynomial"].save_to_file(
        f"surrogate_models/{length_scale}/single_salt/{save_prefix}_polynomial_surrogate",
        overwrite=True,
    )

    surrogates["Kriging"], testing_metrics["Kriging"] = train_kriging_surrogate_model(
        datafile=datafile, length_scale=length_scale, visualize=False
    )
    surrogates["Kriging"].save_to_file(
        f"surrogate_models/{length_scale}/single_salt/{save_prefix}_kriging_surrogate",
        overwrite=True,
    )

    for basis_func in RBF_BASIS_FUNCTIONS:
        name = f"RBF ({basis_func.capitalize() if len(basis_func) > 3 else basis_func.upper()})"
        surrogates[name], testing_metrics[name] = train_rbf_surrogate_model(
            datafile=datafile,
            length_scale=length_scale,
            basis_func=basis_func,
            visualize=False,
        )
        surrogates[name].save_to_file(
            f"surrogate_models/{length_scale}/single_salt/{save_prefix}_rbf_{basis_func}_surrogate",
            overwrite=True,
        )

    # training cost: time in the PySMO trainer and number of high-fidelity points
    n_points = len(read_csv(datafile))
    DataFrame(
        {
            "model": list(TRAINING_TIMES.keys()),
            "training_time_s": list(TRAINING_TIMES.values()),
            "n_training_points_total": n_points,
        }
    ).to_csv(f"figures/{length_scale}/training_times.csv", index=False)

    if return_surrogates:
        return testing_metrics, surrogates
    return testing_metrics


def prepare_training_data(datafile, length_scale):
    """
    Loads the training data and defines the surrogate input labels, output
    labels, and input bounds for the data file.

    Returns: data, input_labels, output_labels, input_bounds
    """
    data = read_csv(datafile)

    # datafile = training_data/<length_scale>/single_salt/<Na_chloride | mixed_chlorides>
    name = os.path.basename(datafile)
    cation = name[:2] if name.endswith("_chloride") else None

    if length_scale == "tube":
        if cation is None:  # mixed chlorides
            input_labels = ["valency", "feed_conc", "pressure", "feed_flow"]
            xmin = [1, 50, 3, 4]
            xmax = [3, 275, 16, 17]
            output_labels = ["ion_flux", "log_water_flux"]
            data["log_water_flux"] = np.log(data["water_flux"])
        else:  # single-cation files (original, ion flux only)
            input_labels = ["feed_conc", "pressure", "feed_flow"]
            output_labels = ["ion_flux"]
            if cation == "Na":
                xmin = [50, 3, 8]
                xmax = [200, 13, 17]
            elif cation == "Ca":
                xmin = [50, 5, 8]
                xmax = [200, 15, 17]
            elif cation == "La":
                xmin = [50, 7, 8]
                xmax = [200, 17, 17]
    elif length_scale == "stage":
        output_labels = ["ion_flux"]
        if cation is None:  # mixed chlorides
            input_labels = ["valency", "feed_conc", "pressure"]
            xmin = [1, 50, 5]
            xmax = [3, 200, 16]
        else:  # single-cation files
            input_labels = ["feed_conc", "pressure"]
            if cation == "Na":
                xmin = [150, 5]
                xmax = [200, 9]
            elif cation == "Ca":
                xmin = [150, 8]
                xmax = [200, 12]
            elif cation == "La":
                xmin = [50, 12]
                xmax = [100, 16]

    input_bounds = {
        input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))
    }

    return data, input_labels, output_labels, input_bounds


def linear_error_metrics(actual, predicted):
    """
    Error metrics between actual and predicted values (same names as the IDAES
    fit metrics, plus mean and max absolute percentage error).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    sse = float(np.sum(error**2))
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MSE": float(np.mean(error**2)),
        "MAE": float(np.mean(np.abs(error))),
        "maxAE": float(np.max(np.abs(error))),
        "SSE": sse,
        "R2": float(1 - sse / np.sum((actual - np.mean(actual)) ** 2)),
        "MAPE": float(100 * np.mean(np.abs(error) / np.abs(actual))),
        "maxAPE": float(100 * np.max(np.abs(error) / np.abs(actual))),
    }


def compute_error_metrics(surrogate, data, input_labels, output_labels):
    """
    Error metrics per output. The log water flux is also reported in linear
    units under the key "water_flux".
    """
    fit_metrics = compute_fit_metrics(surrogate, data)
    metrics = {
        var: {name: float(value(val)) for name, val in info.items()}
        for var, info in fit_metrics.items()
    }
    if "log_water_flux" in output_labels:
        predicted = surrogate.evaluate_surrogate(data[input_labels])
        metrics["water_flux"] = linear_error_metrics(
            data["water_flux"].to_numpy(),
            np.exp(predicted["log_water_flux"].to_numpy()),
        )
    return metrics


def assess_surrogate(
    trained, data, data_training, data_validation, labels, visualize=False
):
    """Builds the surrogate, prints training and testing metrics, and visualizes."""
    input_labels, output_labels, input_bounds = labels
    surrogate = PysmoSurrogate(trained, input_labels, output_labels, input_bounds)

    for title, dataset in (
        ("Training Data Set", data_training),
        ("Testing Data Set", data_validation),
    ):
        print(title)
        metrics = compute_error_metrics(surrogate, dataset, input_labels, output_labels)
        for var, info in metrics.items():
            print(var)
            for error_metric, val in info.items():
                print(f"  {error_metric}={val}")
        if title == "Testing Data Set":
            testing_metrics = metrics

    if visualize:
        for dataset in (data_training, data_validation, data):
            surrogate_scatter2D(surrogate, dataset, show=True)
            surrogate_scatter3D(surrogate, dataset, show=True)
            surrogate_parity(surrogate, dataset, show=True)
            surrogate_residual(surrogate, dataset, show=True)

    return (surrogate, testing_metrics)


def split_data(data, input_labels):
    n_data = data[input_labels[0]].size
    return split_training_validation(data, 0.8, seed=n_data)


def train_polynomial_surrogate_model(
    datafile, length_scale, max_order=4, visualize=False
):
    data, input_labels, output_labels, input_bounds = prepare_training_data(
        datafile, length_scale
    )
    data_training, data_validation = split_data(data, input_labels)

    # create PySMO regression trainer object
    pr_trainer = PysmoPolyTrainer(
        input_labels=input_labels,
        output_labels=output_labels,
        training_dataframe=data_training,
    )

    # set PySMO options
    pr_trainer.config.maximum_polynomial_order = max_order

    # train surrogate model
    start = time.perf_counter()
    poly_train = pr_trainer.train_surrogate()
    TRAINING_TIMES["Polynomial"] = time.perf_counter() - start

    return assess_surrogate(
        poly_train,
        data,
        data_training,
        data_validation,
        (input_labels, output_labels, input_bounds),
        visualize,
    )


def train_kriging_surrogate_model(datafile, length_scale, visualize=False):
    data, input_labels, output_labels, input_bounds = prepare_training_data(
        datafile, length_scale
    )
    data_training, data_validation = split_data(data, input_labels)

    krg_trainer = PysmoKrigingTrainer(
        input_labels=input_labels,
        output_labels=output_labels,
        training_dataframe=data_training,
    )

    krg_trainer.config.numerical_gradients = False

    start = time.perf_counter()
    krg_train = krg_trainer.train_surrogate()
    TRAINING_TIMES["Kriging"] = time.perf_counter() - start

    return assess_surrogate(
        krg_train,
        data,
        data_training,
        data_validation,
        (input_labels, output_labels, input_bounds),
        visualize,
    )


def train_rbf_surrogate_model(
    datafile, length_scale, basis_func="cubic", visualize=False
):
    data, input_labels, output_labels, input_bounds = prepare_training_data(
        datafile, length_scale
    )
    data_training, data_validation = split_data(data, input_labels)

    rbf_trainer = PysmoRBFTrainer(
        input_labels=input_labels,
        output_labels=output_labels,
        training_dataframe=data_training,
    )

    rbf_trainer.config.basis_function = basis_func

    start = time.perf_counter()
    rbf_train = rbf_trainer.train_surrogate()
    TRAINING_TIMES[f"RBF ({basis_func})"] = time.perf_counter() - start

    return assess_surrogate(
        rbf_train,
        data,
        data_training,
        data_validation,
        (input_labels, output_labels, input_bounds),
        visualize,
    )


def validate_tube_surrogate_on_stage_conditions(
    surrogate, stage_tube_file=STAGE_TUBE_CONDITIONS_FILE, save=True
):
    """
    Tests a tube surrogate on the conditions each tube sees inside the
    high-fidelity 5-tube stage (feed flow and concentration after mixing with
    diafiltrate, and applied pressure). None of these points are in the tube
    training data, so this is an independent test of the tube surrogate in the
    conditions where it will be used in the stage flowsheet.

    (suggested and written by Claude)
    """
    conditions = read_csv(stage_tube_file)
    inputs = DataFrame(
        {
            "valency": conditions["valency"],
            "feed_conc": conditions["tube_inlet_conc"],
            "pressure": conditions["tube_pressure"],
            "feed_flow": conditions["tube_inlet_flow"],
        }
    )
    predicted = surrogate.evaluate_surrogate(inputs)
    predicted_ion_flux = predicted["ion_flux"].to_numpy()
    predicted_water_flux = np.exp(predicted["log_water_flux"].to_numpy())

    print("Tube surrogate on the tube conditions inside the high-fidelity stage")
    metrics = {
        "ion_flux": linear_error_metrics(
            conditions["tube_ion_flux"], predicted_ion_flux
        ),
        "water_flux": linear_error_metrics(
            conditions["tube_water_flux"], predicted_water_flux
        ),
    }
    for var, info in metrics.items():
        print(var)
        for error_metric, val in info.items():
            print(f"  {error_metric}={val}")

    results = conditions[
        ["valency", "tube_1_feed_conc", "tube_1_pressure", "tube"]
    ].copy()
    results["tube_ion_flux"] = conditions["tube_ion_flux"]
    results["predicted_ion_flux"] = predicted_ion_flux
    results["tube_water_flux"] = conditions["tube_water_flux"]
    results["predicted_water_flux"] = predicted_water_flux
    results["ion_flux_percent_error"] = (
        100 * (predicted_ion_flux - conditions["tube_ion_flux"])
        / conditions["tube_ion_flux"]
    )
    results["water_flux_percent_error"] = (
        100 * (predicted_water_flux - conditions["tube_water_flux"])
        / conditions["tube_water_flux"]
    )

    print("Mean absolute percent error by tube position in the stage")
    print(
        results.assign(
            ion=results["ion_flux_percent_error"].abs(),
            water=results["water_flux_percent_error"].abs(),
        )
        .groupby("tube")[["ion", "water"]]
        .mean()
    )

    if save:
        results.to_csv(
            "figures/tube/stage_tube_conditions_validation.csv", index=False
        )

    return metrics, results


def plot_testing_error_metrics(
    metrics_by_model,
    output_label,
    title,
    figure_path,
    drop_metrics=("R2",),
    save_figure=True,
):
    """Bar chart of testing error metrics of each model for one output."""
    df = DataFrame(
        {model: metrics[output_label] for model, metrics in metrics_by_model.items()}
    )
    df.index.name = "Error Metric"
    if save_figure:
        df.to_csv(f"{figure_path}.csv")  # all metrics, including percent errors
    df = df.drop([m for m in drop_metrics if m in df.index], axis=0)

    print(output_label)
    print(df)

    ax = df.plot.bar(rot=0)
    ax.legend()
    ax.set_title(title)

    if save_figure:
        plt.savefig(f"{figure_path}.png", dpi=600)


def plot_solute_testing_error_metrics(
    length_scale,
    solute,
    salt,
    testing_metrics,
    drop_r2=False,
    save_figure=True,
):
    # percent errors are not plotted (different scale); they are in the printout
    drop = ["MAPE", "maxAPE"] + (["R2"] if drop_r2 else [])
    plot_testing_error_metrics(
        testing_metrics,
        "ion_flux",
        salt,
        f"figures/{length_scale}/error_metrics_{solute}",
        drop_metrics=drop,
        save_figure=save_figure,
    )


def plot_single_salt_testing_error_metrics(
    length_scale,
    testing_metrics,
    drop_r2=False,
    drop_sse=False,
    save_figure=True,
):
    drop = ["MAPE", "maxAPE"] + (["R2"] if drop_r2 else []) + (["SSE"] if drop_sse else [])

    plot_testing_error_metrics(
        testing_metrics,
        "ion_flux",
        "Single Chloride Salts: Cation Flux",
        f"figures/{length_scale}/error_metrics_single_salt",
        drop_metrics=drop,
        save_figure=save_figure,
    )

    first_model = next(iter(testing_metrics.values()))
    if "water_flux" in first_model:
        # water flux error metrics in linear units (m3/m2/h)
        plot_testing_error_metrics(
            testing_metrics,
            "water_flux",
            "Single Chloride Salts: Water Flux",
            f"figures/{length_scale}/error_metrics_single_salt_water_flux",
            drop_metrics=drop,
            save_figure=save_figure,
        )


if __name__ == "__main__":
    main()
