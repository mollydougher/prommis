"""
Surrogate model training for multi-component diafiltration model.

Author: Molly Dougher
"""

from pandas import read_csv

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


def main():
    polymonial_surrogate = train_polynomial_surrogate_model(
        datafile="training_data/single_salt/Na_chloride"
    )
    kriging_surrogate = train_kriging_surrogate_model(
        datafile="training_data/single_salt/Na_chloride"
    )
    rbf_surrogate = train_rbf_surrogate_model(
        datafile="training_data/single_salt/Na_chloride"
    )


def train_polynomial_surrogate_model(datafile, max_order=4, visualize=True):
    # load data from full expressions
    data = read_csv(datafile)

    # define input and output labels
    input_labels = ["feed_conc", "water_flux"]
    output_labels = ["ion_flux"]
    xmin = [25, 0.01]
    xmax = [200, 0.03]
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
        surrogate_scatter2D(surrogate, data_validation, show=True)
        surrogate_scatter3D(surrogate, data_validation, show=True)
        surrogate_parity(surrogate, data_validation, show=True)
        surrogate_residual(surrogate, data_validation, show=True)

    return surrogate


def train_kriging_surrogate_model(datafile, visualize=True):
    # load data from full expressions
    data = read_csv(datafile)

    # define input and output labels
    input_labels = ["feed_conc", "water_flux"]
    output_labels = ["ion_flux"]
    xmin = [25, 0.01]
    xmax = [200, 0.03]
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
        surrogate_scatter2D(surrogate, data_validation, show=True)
        surrogate_scatter3D(surrogate, data_validation, show=True)
        surrogate_parity(surrogate, data_validation, show=True)
        surrogate_residual(surrogate, data_validation, show=True)

    return surrogate


def train_rbf_surrogate_model(datafile, basis_func="cubic", visualize=True):
    # load data from full expressions
    data = read_csv(datafile)

    # define input and output labels
    input_labels = ["feed_conc", "water_flux"]
    output_labels = ["ion_flux"]
    xmin = [25, 0.01]
    xmax = [200, 0.03]
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
        surrogate_scatter2D(surrogate, data_validation, show=True)
        surrogate_scatter3D(surrogate, data_validation, show=True)
        surrogate_parity(surrogate, data_validation, show=True)
        surrogate_residual(surrogate, data_validation, show=True)

    return surrogate


if __name__ == "__main__":
    main()
