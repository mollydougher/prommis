"""
Surrogate model training for multi-component diafiltration model.

Author: Molly Dougher
"""

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


def generate_per_salt_analysis(solute, salt, length_scale):
    datafile = f"training_data/{length_scale}/single_salt/{solute}_chloride"

    polymonial_surrogate, polynomial_testing_error_metrics = (
        train_polynomial_surrogate_model(
            datafile=datafile,
            visualize=False,
        )
    )
    polymonial_surrogate.save_to_file(
        f"surrogate_models/{length_scale}/single_salt/{solute}Cl_polynomial_surrogate",
        overwrite=True,
    )

    kriging_surrogate, kriging_testing_error_metrics = train_kriging_surrogate_model(
        datafile=datafile,
        visualize=False,
    )
    kriging_surrogate.save_to_file(
        f"surrogate_models/{length_scale}/single_salt/{solute}Cl_kriging_surrogate",
        overwrite=True,
    )

    basis_functions = ["linear", "cubic", "spline", "gaussian", "mq", "imq"]
    for basis_func in basis_functions:
        rbf_surrogate, rbf_testing_error_metrics = train_rbf_surrogate_model(
            datafile=datafile,
            basis_func=basis_func,
            visualize=False,
        )
        rbf_surrogate.save_to_file(
            f"surrogate_models/{length_scale}/single_salt/{solute}Cl_rbf_{basis_func}_surrogate",
            overwrite=True,
        )
        if basis_func == "linear":
            rbf_linear_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "cubic":
            rbf_cubic_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "spline":
            rbf_spline_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "gaussian":
            rbf_gaussian_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "mq":
            rbf_mq_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "imq":
            rbf_imq_testing_error_metrics = rbf_testing_error_metrics

    plot_solute_testing_error_metrics(
        length_scale,
        solute,
        salt,
        polynomial_testing_error_metrics,
        kriging_testing_error_metrics,
        rbf_linear_testing_error_metrics,
        rbf_cubic_testing_error_metrics,
        rbf_spline_testing_error_metrics,
        rbf_gaussian_testing_error_metrics,
        rbf_mq_testing_error_metrics,
        rbf_imq_testing_error_metrics,
        drop_r2=True,
        save_figure=True,
    )

    plt.show()


def generate_mixed_salt_analysis(length_scale):
    datafile = f"training_data/{length_scale}/single_salt/mixed_chlorides"

    polymonial_surrogate, polynomial_testing_error_metrics = (
        train_polynomial_surrogate_model(
            datafile=datafile,
            visualize=False,
        )
    )
    polymonial_surrogate.save_to_file(
        f"surrogate_models/{length_scale}/single_salt/single_salt_polynomial_surrogate",
        overwrite=True,
    )

    kriging_surrogate, kriging_testing_error_metrics = train_kriging_surrogate_model(
        datafile=datafile,
        visualize=False,
    )
    kriging_surrogate.save_to_file(
        f"surrogate_models/{length_scale}/single_salt/single_salt_kriging_surrogate",
        overwrite=True,
    )

    basis_functions = ["linear", "cubic", "spline", "gaussian", "mq", "imq"]
    for basis_func in basis_functions:
        rbf_surrogate, rbf_testing_error_metrics = train_rbf_surrogate_model(
            datafile=datafile,
            basis_func=basis_func,
            visualize=False,
        )
        rbf_surrogate.save_to_file(
            f"surrogate_models/{length_scale}/single_salt/single_salt_rbf_{basis_func}_surrogate",
            overwrite=True,
        )
        if basis_func == "linear":
            rbf_linear_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "cubic":
            rbf_cubic_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "spline":
            rbf_spline_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "gaussian":
            rbf_gaussian_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "mq":
            rbf_mq_testing_error_metrics = rbf_testing_error_metrics
        elif basis_func == "imq":
            rbf_imq_testing_error_metrics = rbf_testing_error_metrics

    plot_single_salt_testing_error_metrics(
        length_scale,
        polynomial_testing_error_metrics,
        kriging_testing_error_metrics,
        rbf_linear_testing_error_metrics,
        rbf_cubic_testing_error_metrics,
        rbf_spline_testing_error_metrics,
        rbf_gaussian_testing_error_metrics,
        rbf_mq_testing_error_metrics,
        rbf_imq_testing_error_metrics,
        drop_r2=True,
        drop_sse=True,
        save_figure=True,
    )

    plt.show()


def train_polynomial_surrogate_model(datafile, max_order=4, visualize=False):
    # load data from full expressions
    data = read_csv(datafile)

    # define input and output labels
    input_labels = ["feed_conc", "pressure", "feed_flow"]
    output_labels = ["ion_flux"]
    xmin = [25, 5, 10]
    xmax = [200, 15, 17]
    input_bounds = {
        input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))
    }

    n_data = data[input_labels[0]].size
    data_training, data_validation = split_training_validation(data, 0.8, seed=n_data)

    # create PySMO regression trainer object
    pr_trainer = PysmoPolyTrainer(
        input_labels=input_labels,
        output_labels=output_labels,
        training_dataframe=data_training,
    )

    # set PySMO options
    pr_trainer.config.maximum_polynomial_order = max_order

    # train surrogate model
    poly_train = pr_trainer.train_surrogate()

    surrogate = PysmoSurrogate(poly_train, input_labels, output_labels, input_bounds)

    training_metrics = compute_fit_metrics(surrogate, data_training)
    print("Training Data Set")
    for var, info in training_metrics.items():
        for error_metric, val in info.items():
            print(f"{error_metric}={value(val)}")

    testing_metrics = compute_fit_metrics(surrogate, data_validation)
    print("Testing Data Set")
    for var, info in testing_metrics.items():
        for error_metric, val in info.items():
            print(f"{error_metric}={value(val)}")

    if visualize:
        # visualize
        surrogate_scatter2D(surrogate, data_training, show=True)
        surrogate_scatter3D(surrogate, data_training, show=True)
        surrogate_parity(surrogate, data_training, show=True)
        surrogate_residual(surrogate, data_training, show=True)

        surrogate_scatter2D(surrogate, data_validation, show=True)
        surrogate_scatter3D(surrogate, data_validation, show=True)
        surrogate_parity(surrogate, data_validation, show=True)
        surrogate_residual(surrogate, data_validation, show=True)

        surrogate_scatter2D(surrogate, data, show=True)
        surrogate_scatter3D(surrogate, data, show=True)
        surrogate_parity(surrogate, data, show=True)
        surrogate_residual(surrogate, data, show=True)

    return (surrogate, testing_metrics)


def train_kriging_surrogate_model(datafile, visualize=False):
    # load data from full expressions
    data = read_csv(datafile)

    # define input and output labels
    input_labels = ["feed_conc", "pressure", "feed_flow"]
    output_labels = ["ion_flux"]
    xmin = [25, 5, 10]
    xmax = [200, 15, 17]
    input_bounds = {
        input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))
    }

    n_data = data[input_labels[0]].size
    data_training, data_validation = split_training_validation(data, 0.8, seed=n_data)

    krg_trainer = PysmoKrigingTrainer(
        input_labels=input_labels,
        output_labels=output_labels,
        training_dataframe=data_training,
    )

    krg_trainer.config.numerical_gradients = False

    krg_train = krg_trainer.train_surrogate()

    surrogate = PysmoSurrogate(krg_train, input_labels, output_labels, input_bounds)

    training_metrics = compute_fit_metrics(surrogate, data_training)
    print("Training Data Set")
    for var, info in training_metrics.items():
        for error_metric, val in info.items():
            print(f"{error_metric}={value(val)}")

    testing_metrics = compute_fit_metrics(surrogate, data_validation)
    print("Testing Data Set")
    for var, info in testing_metrics.items():
        for error_metric, val in info.items():
            print(f"{error_metric}={value(val)}")

    if visualize:
        # visualize
        surrogate_scatter2D(surrogate, data_training, show=True)
        surrogate_scatter3D(surrogate, data_training, show=True)
        surrogate_parity(surrogate, data_training, show=True)
        surrogate_residual(surrogate, data_training, show=True)

        surrogate_scatter2D(surrogate, data_validation, show=True)
        surrogate_scatter3D(surrogate, data_validation, show=True)
        surrogate_parity(surrogate, data_validation, show=True)
        surrogate_residual(surrogate, data_validation, show=True)

        surrogate_scatter2D(surrogate, data, show=True)
        surrogate_scatter3D(surrogate, data, show=True)
        surrogate_parity(surrogate, data, show=True)
        surrogate_residual(surrogate, data, show=True)

    return (surrogate, testing_metrics)


def train_rbf_surrogate_model(datafile, basis_func="cubic", visualize=False):
    # load data from full expressions
    data = read_csv(datafile)

    # define input and output labels
    input_labels = ["feed_conc", "pressure", "feed_flow"]
    output_labels = ["ion_flux"]
    xmin = [25, 5, 10]
    xmax = [200, 15, 17]
    input_bounds = {
        input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))
    }

    n_data = data[input_labels[0]].size
    data_training, data_validation = split_training_validation(data, 0.8, seed=n_data)

    rbf_trainer = PysmoRBFTrainer(
        input_labels=input_labels,
        output_labels=output_labels,
        training_dataframe=data_training,
    )

    rbf_trainer.config.basis_function = basis_func

    rbf_train = rbf_trainer.train_surrogate()

    surrogate = PysmoSurrogate(rbf_train, input_labels, output_labels, input_bounds)

    training_metrics = compute_fit_metrics(surrogate, data_training)
    print("Training Data Set")
    for var, info in training_metrics.items():
        for error_metric, val in info.items():
            print(f"{error_metric}={value(val)}")

    testing_metrics = compute_fit_metrics(surrogate, data_validation)
    print("Testing Data Set")
    for var, info in testing_metrics.items():
        for error_metric, val in info.items():
            print(f"{error_metric}={value(val)}")

    if visualize:
        # visualize
        surrogate_scatter2D(surrogate, data_training, show=True)
        surrogate_scatter3D(surrogate, data_training, show=True)
        surrogate_parity(surrogate, data_training, show=True)
        surrogate_residual(surrogate, data_training, show=True)

        surrogate_scatter2D(surrogate, data_validation, show=True)
        surrogate_scatter3D(surrogate, data_validation, show=True)
        surrogate_parity(surrogate, data_validation, show=True)
        surrogate_residual(surrogate, data_validation, show=True)

        surrogate_scatter2D(surrogate, data, show=True)
        surrogate_scatter3D(surrogate, data, show=True)
        surrogate_parity(surrogate, data, show=True)
        surrogate_residual(surrogate, data, show=True)

    return (surrogate, testing_metrics)


def plot_solute_testing_error_metrics(
    length_scale,
    solute,
    salt,
    polynomial_testing_error_metrics,
    kriging_testing_error_metrics,
    rbf_linear_testing_error_metrics,
    rbf_cubic_testing_error_metrics,
    rbf_spline_testing_error_metrics,
    rbf_gaussian_testing_error_metrics,
    rbf_mq_testing_error_metrics,
    rbf_imq_testing_error_metrics,
    drop_r2=False,
    save_figure=True,
):

    metric_list = []
    poly_error_list = []
    krig_error_list = []
    rbf_linear_error_list = []
    rbf_cubic_error_list = []
    rbf_spline_error_list = []
    rbf_gauss_error_list = []
    rbf_mq_error_list = []
    rbf_imq_error_list = []

    for var, info in polynomial_testing_error_metrics.items():
        for error_metric, val in info.items():
            metric_list.append(error_metric)
            poly_error_list.append(value(val))
    for var, info in kriging_testing_error_metrics.items():
        for error_metric, val in info.items():
            krig_error_list.append(value(val))
    for var, info in rbf_linear_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_linear_error_list.append(value(val))
    for var, info in rbf_cubic_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_cubic_error_list.append(value(val))
    for var, info in rbf_spline_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_spline_error_list.append(value(val))
    for var, info in rbf_gaussian_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_gauss_error_list.append(value(val))
    for var, info in rbf_mq_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_mq_error_list.append(value(val))
    for var, info in rbf_imq_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_imq_error_list.append(value(val))

    dict = {
        "Error Metric": metric_list,
        "Polynomial": poly_error_list,
        "Kriging": krig_error_list,
        "RBF (Linear)": rbf_cubic_error_list,
        "RBF (Cubic)": rbf_cubic_error_list,
        "RBF (Spline)": rbf_cubic_error_list,
        "RBF (Gaussian)": rbf_gauss_error_list,
        "RBF (MQ)": rbf_mq_error_list,
        "RBF (IMQ)": rbf_imq_error_list,
    }

    df = DataFrame(dict).set_index("Error Metric")

    if drop_r2:
        df = df.drop("R2", axis=0)

    print(df)

    ax1 = df.plot.bar(rot=0)

    ax1.legend()
    ax1.set_title(f"{salt}")

    if save_figure:
        plt.savefig(f"figures/{length_scale}/error_metrics_{solute}.png", dpi=600)


def plot_single_salt_testing_error_metrics(
    length_scale,
    polynomial_testing_error_metrics,
    kriging_testing_error_metrics,
    rbf_linear_testing_error_metrics,
    rbf_cubic_testing_error_metrics,
    rbf_spline_testing_error_metrics,
    rbf_gaussian_testing_error_metrics,
    rbf_mq_testing_error_metrics,
    rbf_imq_testing_error_metrics,
    drop_r2=False,
    drop_sse=False,
    save_figure=True,
):

    metric_list = []
    poly_error_list = []
    krig_error_list = []
    rbf_linear_error_list = []
    rbf_cubic_error_list = []
    rbf_spline_error_list = []
    rbf_gauss_error_list = []
    rbf_mq_error_list = []
    rbf_imq_error_list = []

    for var, info in polynomial_testing_error_metrics.items():
        for error_metric, val in info.items():
            metric_list.append(error_metric)
            poly_error_list.append(value(val))
    for var, info in kriging_testing_error_metrics.items():
        for error_metric, val in info.items():
            krig_error_list.append(value(val))
    for var, info in rbf_linear_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_linear_error_list.append(value(val))
    for var, info in rbf_cubic_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_cubic_error_list.append(value(val))
    for var, info in rbf_spline_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_spline_error_list.append(value(val))
    for var, info in rbf_gaussian_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_gauss_error_list.append(value(val))
    for var, info in rbf_mq_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_mq_error_list.append(value(val))
    for var, info in rbf_imq_testing_error_metrics.items():
        for error_metric, val in info.items():
            rbf_imq_error_list.append(value(val))

    dict = {
        "Error Metric": metric_list,
        "Polynomial": poly_error_list,
        "Kriging": krig_error_list,
        "RBF (Linear)": rbf_cubic_error_list,
        "RBF (Cubic)": rbf_cubic_error_list,
        "RBF (Spline)": rbf_cubic_error_list,
        "RBF (Gaussian)": rbf_gauss_error_list,
        "RBF (MQ)": rbf_mq_error_list,
        "RBF (IMQ)": rbf_imq_error_list,
    }

    df = DataFrame(dict).set_index("Error Metric")

    if drop_r2:
        df = df.drop("R2", axis=0)
    if drop_sse:
        df = df.drop("SSE", axis=0)

    print(df)

    ax1 = df.plot.bar(rot=0)

    ax1.legend()
    ax1.set_title("Single Chloride Salts")

    if save_figure:
        plt.savefig(f"figures/{length_scale}/error_metrics_single_salt.png", dpi=600)


if __name__ == "__main__":
    main()
