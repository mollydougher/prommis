from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


def main():
    plot_residuals_two_salt()
    # plot_residuals_three_salt()


def plot_residuals_two_salt():
    surrogate_model_file_dict_two_salt = {
        "D_11": "surrogate_models/lithium_cobalt_chloride/with_chi_input/fractional_factorial_1/with_extra_center/rbf_surrogate_d11_scaled",
        "D_12": "surrogate_models/lithium_cobalt_chloride/with_chi_input/fractional_factorial_1/with_extra_center/rbf_surrogate_d12_scaled",
        "D_21": "surrogate_models/lithium_cobalt_chloride/with_chi_input/fractional_factorial_1/with_extra_center/rbf_surrogate_d21_scaled",
        "D_22": "surrogate_models/lithium_cobalt_chloride/with_chi_input/fractional_factorial_1/with_extra_center/rbf_surrogate_d22_scaled",
        "alpha_1": "surrogate_models/lithium_cobalt_chloride/with_chi_input/fractional_factorial_1/with_extra_center/rbf_surrogate_alpha_1",
        "alpha_2": "surrogate_models/lithium_cobalt_chloride/with_chi_input/fractional_factorial_1/with_extra_center/rbf_surrogate_alpha_2",
    }

    _surrogates_obj_D_11 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict_two_salt["D_11"]
    )
    _surrogates_obj_D_12 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict_two_salt["D_12"]
    )
    _surrogates_obj_D_21 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict_two_salt["D_21"]
    )
    _surrogates_obj_D_22 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict_two_salt["D_22"]
    )
    _surrogates_obj_alpha_1 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict_two_salt["alpha_1"]
    )
    _surrogates_obj_alpha_2 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict_two_salt["alpha_2"]
    )

    conc_array = np.arange(50, 205, 5)

    c1_vals = []
    c2_vals = []
    for c1 in conc_array:
        c1_vals.append(c1)
    for c2 in conc_array:
        c2_vals.append(c2)

    chi = -140

    # RBF_surrogate_vals = []
    # actual_vals = []
    abs_residual_error_vals_rbf_d11 = []
    percent_error_vals_rbf_d11 = []

    abs_residual_error_vals_rbf_d12 = []
    percent_error_vals_rbf_d12 = []

    abs_residual_error_vals_rbf_d21 = []
    percent_error_vals_rbf_d21 = []

    abs_residual_error_vals_rbf_d22 = []
    percent_error_vals_rbf_d22 = []

    abs_residual_error_vals_rbf_alpha1 = []
    percent_error_vals_rbf_alpha1 = []

    abs_residual_error_vals_rbf_alpha2 = []
    percent_error_vals_rbf_alpha2 = []

    # RBF_surrogate_vals_dict = {}
    # actual_vals_dict = {}
    abs_residual_error_dict_rbf_d11 = {}
    percent_error_dict_rbf_d11 = {}

    abs_residual_error_dict_rbf_d12 = {}
    percent_error_dict_rbf_d12 = {}

    abs_residual_error_dict_rbf_d21 = {}
    percent_error_dict_rbf_d21 = {}

    abs_residual_error_dict_rbf_d22 = {}
    percent_error_dict_rbf_d22 = {}

    abs_residual_error_dict_rbf_alpha1 = {}
    percent_error_dict_rbf_alpha1 = {}

    abs_residual_error_dict_rbf_alpha2 = {}
    percent_error_dict_rbf_alpha2 = {}

    for c1 in conc_array:
        for c2 in conc_array:
            rbf_val_d11 = evaluate_surrogate_two_salt(
                c1, c2, chi, _surrogates_obj_D_11, "D_11_scaled"
            )
            # RBF_surrogate_vals.append(rbf_val)
            rbf_val_d12 = evaluate_surrogate_two_salt(
                c1, c2, chi, _surrogates_obj_D_12, "D_12_scaled"
            )
            rbf_val_d21 = evaluate_surrogate_two_salt(
                c1, c2, chi, _surrogates_obj_D_21, "D_21_scaled"
            )
            rbf_val_d22 = evaluate_surrogate_two_salt(
                c1, c2, chi, _surrogates_obj_D_22, "D_22_scaled"
            )
            rbf_val_alpha1 = evaluate_surrogate_two_salt(
                c1, c2, chi, _surrogates_obj_alpha_1, "alpha_1"
            )
            rbf_val_alpha2 = evaluate_surrogate_two_salt(
                c1, c2, chi, _surrogates_obj_alpha_2, "alpha_2"
            )

            actual_val_d11 = calculate_true_value_two_salt_d11(c1, c2, chi)
            # actual_vals.append(actual_val)
            actual_val_d12 = calculate_true_value_two_salt_d12(c1, c2, chi)
            actual_val_d21 = calculate_true_value_two_salt_d21(c1, c2, chi)
            actual_val_d22 = calculate_true_value_two_salt_d22(c1, c2, chi)
            actual_val_alpha1 = calculate_true_value_two_salt_alpha1(c1, c2, chi)
            actual_val_alpha2 = calculate_true_value_two_salt_alpha2(c1, c2, chi)

            abs_residual_error_vals_rbf_d11.append(abs(rbf_val_d11 - actual_val_d11))
            abs_residual_error_vals_rbf_d12.append(abs(rbf_val_d12 - actual_val_d12))
            abs_residual_error_vals_rbf_d21.append(abs(rbf_val_d21 - actual_val_d21))
            abs_residual_error_vals_rbf_d22.append(abs(rbf_val_d22 - actual_val_d22))
            abs_residual_error_vals_rbf_alpha1.append(
                abs(rbf_val_alpha1 - actual_val_alpha1)
            )
            abs_residual_error_vals_rbf_alpha2.append(
                abs(rbf_val_alpha2 - actual_val_alpha2)
            )

            percent_error_val_rbf_d11 = (
                abs(rbf_val_d11 - actual_val_d11) / abs(actual_val_d11) * 100
            )
            percent_error_val_rbf_d12 = (
                abs(rbf_val_d12 - actual_val_d12) / abs(actual_val_d12) * 100
            )
            percent_error_val_rbf_d21 = (
                abs(rbf_val_d21 - actual_val_d21) / abs(actual_val_d21) * 100
            )
            percent_error_val_rbf_d22 = (
                abs(rbf_val_d22 - actual_val_d22) / abs(actual_val_d22) * 100
            )
            percent_error_val_rbf_alpha1 = (
                abs(rbf_val_alpha1 - actual_val_alpha1) / abs(actual_val_alpha1) * 100
            )
            percent_error_val_rbf_alpha2 = (
                abs(rbf_val_alpha2 - actual_val_alpha2) / abs(actual_val_alpha2) * 100
            )

            percent_error_vals_rbf_d11.append(percent_error_val_rbf_d11)
            percent_error_vals_rbf_d12.append(percent_error_val_rbf_d12)
            percent_error_vals_rbf_d21.append(percent_error_val_rbf_d21)
            percent_error_vals_rbf_d22.append(percent_error_val_rbf_d22)
            percent_error_vals_rbf_alpha1.append(percent_error_val_rbf_alpha1)
            percent_error_vals_rbf_alpha2.append(percent_error_val_rbf_alpha2)

        # RBF_surrogate_vals_dict[f"{c1}"] = RBF_surrogate_vals
        # actual_vals_dict[f"{c1}"] = actual_vals
        abs_residual_error_dict_rbf_d11[f"{c1}"] = abs_residual_error_vals_rbf_d11
        percent_error_dict_rbf_d11[f"{c1}"] = percent_error_vals_rbf_d11

        abs_residual_error_dict_rbf_d12[f"{c1}"] = abs_residual_error_vals_rbf_d12
        percent_error_dict_rbf_d12[f"{c1}"] = percent_error_vals_rbf_d12

        abs_residual_error_dict_rbf_d21[f"{c1}"] = abs_residual_error_vals_rbf_d21
        percent_error_dict_rbf_d21[f"{c1}"] = percent_error_vals_rbf_d21

        abs_residual_error_dict_rbf_d22[f"{c1}"] = abs_residual_error_vals_rbf_d22
        percent_error_dict_rbf_d22[f"{c1}"] = percent_error_vals_rbf_d22

        abs_residual_error_dict_rbf_alpha1[f"{c1}"] = abs_residual_error_vals_rbf_alpha1
        percent_error_dict_rbf_alpha1[f"{c1}"] = percent_error_vals_rbf_alpha1

        abs_residual_error_dict_rbf_alpha2[f"{c1}"] = abs_residual_error_vals_rbf_alpha2
        percent_error_dict_rbf_alpha2[f"{c1}"] = percent_error_vals_rbf_alpha2

        RBF_surrogate_vals = []
        actual_vals = []
        abs_residual_error_vals_rbf_d11 = []
        percent_error_vals_rbf_d11 = []

        abs_residual_error_vals_rbf_d12 = []
        percent_error_vals_rbf_d12 = []

        abs_residual_error_vals_rbf_d21 = []
        percent_error_vals_rbf_d21 = []

        abs_residual_error_vals_rbf_d22 = []
        percent_error_vals_rbf_d22 = []

        abs_residual_error_vals_rbf_alpha1 = []
        percent_error_vals_rbf_alpha1 = []

        abs_residual_error_vals_rbf_alpha2 = []
        percent_error_vals_rbf_alpha2 = []

    # RBF_surrogate_vals_df = DataFrame(index=c2_vals, data=RBF_surrogate_vals_dict)
    # actual_vals_df = DataFrame(index=c2_vals, data=actual_vals_dict)
    abs_residual_error_df_rbf_d11 = DataFrame(
        index=c2_vals, data=abs_residual_error_dict_rbf_d11
    )
    percent_error_df_rbf_d11 = DataFrame(index=c2_vals, data=percent_error_dict_rbf_d11)

    abs_residual_error_df_rbf_d12 = DataFrame(
        index=c2_vals, data=abs_residual_error_dict_rbf_d12
    )
    percent_error_df_rbf_d12 = DataFrame(index=c2_vals, data=percent_error_dict_rbf_d12)

    abs_residual_error_df_rbf_d21 = DataFrame(
        index=c2_vals, data=abs_residual_error_dict_rbf_d21
    )
    percent_error_df_rbf_d21 = DataFrame(index=c2_vals, data=percent_error_dict_rbf_d21)

    abs_residual_error_df_rbf_d22 = DataFrame(
        index=c2_vals, data=abs_residual_error_dict_rbf_d22
    )
    percent_error_df_rbf_d22 = DataFrame(index=c2_vals, data=percent_error_dict_rbf_d22)

    abs_residual_error_df_rbf_alpha1 = DataFrame(
        index=c2_vals, data=abs_residual_error_dict_rbf_alpha1
    )
    percent_error_df_rbf_alpha1 = DataFrame(
        index=c2_vals, data=percent_error_dict_rbf_alpha1
    )

    abs_residual_error_df_rbf_alpha2 = DataFrame(
        index=c2_vals, data=abs_residual_error_dict_rbf_alpha2
    )
    percent_error_df_rbf_alpha2 = DataFrame(
        index=c2_vals, data=percent_error_dict_rbf_alpha2
    )

    # print(RBF_surrogate_vals_df)
    # print(actual_vals_df)
    # print(abs_residual_error_df_rbf)
    # print(percent_error_df_rbf)

    fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 10))
    percent_error_plot_rbf_d11 = ax1.pcolor(c1_vals, c2_vals, percent_error_df_rbf_d11)
    percent_error_plot_rbf_d12 = ax2.pcolor(c1_vals, c2_vals, percent_error_df_rbf_d12)
    percent_error_plot_rbf_d21 = ax4.pcolor(c1_vals, c2_vals, percent_error_df_rbf_d21)
    percent_error_plot_rbf_d22 = ax5.pcolor(c1_vals, c2_vals, percent_error_df_rbf_d22)
    percent_error_plot_rbf_alpha1 = ax3.pcolor(
        c1_vals, c2_vals, percent_error_df_rbf_alpha1
    )
    percent_error_plot_rbf_alpha2 = ax6.pcolor(
        c1_vals, c2_vals, percent_error_df_rbf_alpha2
    )

    plt.suptitle("Percent Error (%)\nRBF Surrogate", fontsize=12, fontweight="bold")

    ax1.set_title("D_11", fontsize=12, fontweight="bold")
    ax2.set_title("D_12", fontsize=12, fontweight="bold")
    ax4.set_title("D_21", fontsize=12, fontweight="bold")
    ax5.set_title("D_22", fontsize=12, fontweight="bold")
    ax3.set_title("alpha_1", fontsize=12, fontweight="bold")
    ax6.set_title("alpha_2", fontsize=12, fontweight="bold")

    for ax in (ax1, ax4):
        ax.set_ylabel(
            "Cobalt Concentration in Membrane (mM)",
            fontsize=12,
            fontweight="bold",
        )
    for ax in (ax4, ax5, ax6):
        ax.tick_params(direction="in", labelsize=10)
        ax.set_xlabel(
            "Lithium Concentration\n in Membrane (mM)",
            fontsize=12,
            fontweight="bold",
        )

    fig1.colorbar(percent_error_plot_rbf_d11, ax=ax1)
    fig1.colorbar(percent_error_plot_rbf_d12, ax=ax2)
    fig1.colorbar(percent_error_plot_rbf_alpha1, ax=ax3)
    fig1.colorbar(percent_error_plot_rbf_d21, ax=ax4)
    fig1.colorbar(percent_error_plot_rbf_d22, ax=ax5)
    fig1.colorbar(percent_error_plot_rbf_alpha2, ax=ax6)
    plt.show()

    fig2, ((ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(2, 3, figsize=(10, 10))
    abs_residual_error_plot_rbf_d11 = ax7.pcolor(
        c1_vals, c2_vals, abs_residual_error_df_rbf_d11
    )
    abs_residual_error_plot_rbf_d12 = ax8.pcolor(
        c1_vals, c2_vals, abs_residual_error_df_rbf_d12
    )
    abs_residual_error_plot_rbf_d21 = ax10.pcolor(
        c1_vals, c2_vals, abs_residual_error_df_rbf_d21
    )
    abs_residual_error_plot_rbf_d22 = ax11.pcolor(
        c1_vals, c2_vals, abs_residual_error_df_rbf_d22
    )
    abs_residual_error_plot_rbf_alpha1 = ax9.pcolor(
        c1_vals, c2_vals, abs_residual_error_df_rbf_alpha1
    )
    abs_residual_error_plot_rbf_alpha2 = ax12.pcolor(
        c1_vals, c2_vals, abs_residual_error_df_rbf_alpha2
    )

    plt.suptitle(
        "Residual Error (m2/h or dimensionless)\nRBF Surrogate",
        fontsize=12,
        fontweight="bold",
    )

    ax7.set_title("D_11", fontsize=12, fontweight="bold")
    ax8.set_title("D_12", fontsize=12, fontweight="bold")
    ax10.set_title("D_21", fontsize=12, fontweight="bold")
    ax11.set_title("D_22", fontsize=12, fontweight="bold")
    ax9.set_title("alpha_1", fontsize=12, fontweight="bold")
    ax12.set_title("alpha_2", fontsize=12, fontweight="bold")

    for ax in (ax7, ax10):
        ax.set_ylabel(
            "Cobalt Concentration in Membrane (mM)",
            fontsize=12,
            fontweight="bold",
        )
    for ax in (ax10, ax11, ax12):
        ax.tick_params(direction="in", labelsize=10)
        ax.set_xlabel(
            "Lithium Concentration\n in Membrane (mM)",
            fontsize=12,
            fontweight="bold",
        )

    fig2.colorbar(abs_residual_error_plot_rbf_d11, ax=ax7)
    fig2.colorbar(abs_residual_error_plot_rbf_d12, ax=ax8)
    fig2.colorbar(abs_residual_error_plot_rbf_alpha1, ax=ax9)
    fig2.colorbar(abs_residual_error_plot_rbf_d21, ax=ax10)
    fig2.colorbar(abs_residual_error_plot_rbf_d22, ax=ax11)
    fig2.colorbar(abs_residual_error_plot_rbf_alpha2, ax=ax12)
    plt.show()


def evaluate_linear_surrogate_two_salt(c1, c2):
    val = -4.33e-06 + -4.21e-09 * c1 + 5.10e-09 * c2

    return val


def evaluate_linear_surrogate_three_salt(c1, c2, c3):
    val = -4.15e-06 + -3.61e-09 * c1 + 3.03e-09 * c2 + 4.75e-09 * c3

    return val


def plot_residuals_three_salt():
    surrogate_model_file_dict_three_salt = {
        "D_11": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d11_scaled",
        # "D_12": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d12_scaled",
        # "D_13": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d13_scaled",
        # "D_21": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d21_scaled",
        # "D_22": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d22_scaled",
        # "D_23": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d23_scaled",
        # "D_31": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d31_scaled",
        # "D_32": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d32_scaled",
        # "D_33": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_d33_scaled",
        # "alpha_1": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_alpha1",
        # "alpha_2": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_alpha2",
        # "alpha_3": "surrogate_models/lithium_cobalt_aluminum_chloride/rbf_pysmo_surrogate_alpha3",
    }

    _surrogates_obj_D_11 = PysmoSurrogate.load_from_file(
        surrogate_model_file_dict_three_salt["D_11"]
    )

    c1_vals = []
    c2_vals = []
    c3_vals = []
    for c1 in np.arange(75, 235, 10):
        c1_vals.append(c1)
    for c2 in np.arange(50, 210, 10):
        c2_vals.append(c2)
    for c3 in [2, 17, 32]:
        c3_vals.append(c3)

    linear_surrogate_vals = []
    RBF_surrogate_vals = []
    actual_vals = []
    percent_error_vals_linear = []
    percent_error_vals_rbf = []

    for c3 in [2, 17, 32]:
        for c1 in np.arange(75, 235, 10):
            for c2 in np.arange(50, 210, 10):
                linear_val = evaluate_linear_surrogate_three_salt(c1, c2, c3)
                linear_surrogate_vals.append(linear_val)

                rbf_val = evaluate_surrogate_three_salt(
                    c1, c2, c3, -140, _surrogates_obj_D_11, "D_11_scaled"
                )
                RBF_surrogate_vals.append(rbf_val)

                actual_val = calculate_true_value_three_salt(c1, c2, c3)
                actual_vals.append(actual_val)

                percent_error_val_linear = (
                    abs(linear_val - actual_val) / abs(actual_val) * 100
                )
                percent_error_vals_linear.append(percent_error_val_linear)

                percent_error_val_rbf = (
                    abs(rbf_val - actual_val) / abs(actual_val) * 100
                )
                percent_error_vals_rbf.append(percent_error_val_rbf)

            if c3 == 2:
                if c1 == 75:
                    linear_surrogate_vals_dict_l = {f"{c1})": linear_surrogate_vals}
                    RBF_surrogate_vals_dict_l = {f"{c1}": RBF_surrogate_vals}
                    actual_vals_dict_l = {f"{c1}": actual_vals}
                    percent_error_dict_linear_l = {f"{c1}": percent_error_vals_linear}
                    percent_error_dict_rbf_l = {f"{c1}": percent_error_vals_rbf}
                else:
                    linear_surrogate_vals_dict_l[f"{c1})"] = linear_surrogate_vals
                    RBF_surrogate_vals_dict_l[f"{c1}"] = RBF_surrogate_vals
                    actual_vals_dict_l[f"{c1}"] = actual_vals
                    percent_error_dict_linear_l[f"{c1}"] = percent_error_vals_linear
                    percent_error_dict_rbf_l[f"{c1}"] = percent_error_vals_rbf
            elif c3 == 17:
                if c1 == 75:
                    linear_surrogate_vals_dict_m = {f"{c1})": linear_surrogate_vals}
                    RBF_surrogate_vals_dict_m = {f"{c1}": RBF_surrogate_vals}
                    actual_vals_dict_m = {f"{c1}": actual_vals}
                    percent_error_dict_linear_m = {f"{c1}": percent_error_vals_linear}
                    percent_error_dict_rbf_m = {f"{c1}": percent_error_vals_rbf}
                else:
                    linear_surrogate_vals_dict_m[f"{c1})"] = linear_surrogate_vals
                    RBF_surrogate_vals_dict_m[f"{c1}"] = RBF_surrogate_vals
                    actual_vals_dict_m[f"{c1}"] = actual_vals
                    percent_error_dict_linear_m[f"{c1}"] = percent_error_vals_linear
                    percent_error_dict_rbf_m[f"{c1}"] = percent_error_vals_rbf
            elif c3 == 32:
                if c1 == 75:
                    linear_surrogate_vals_dict_h = {f"{c1})": linear_surrogate_vals}
                    RBF_surrogate_vals_dict_h = {f"{c1}": RBF_surrogate_vals}
                    actual_vals_dict_h = {f"{c1}": actual_vals}
                    percent_error_dict_linear_h = {f"{c1}": percent_error_vals_linear}
                    percent_error_dict_rbf_h = {f"{c1}": percent_error_vals_rbf}
                else:
                    linear_surrogate_vals_dict_h[f"{c1})"] = linear_surrogate_vals
                    RBF_surrogate_vals_dict_h[f"{c1}"] = RBF_surrogate_vals
                    actual_vals_dict_h[f"{c1}"] = actual_vals
                    percent_error_dict_linear_h[f"{c1}"] = percent_error_vals_linear
                    percent_error_dict_rbf_h[f"{c1}"] = percent_error_vals_rbf

            linear_surrogate_vals = []
            RBF_surrogate_vals = []
            actual_vals = []
            percent_error_vals_linear = []
            percent_error_vals_rbf = []

    linear_surrogate_vals_df_l = DataFrame(
        index=c2_vals, data=linear_surrogate_vals_dict_l
    )
    RBF_surrogate_vals_df_l = DataFrame(index=c2_vals, data=RBF_surrogate_vals_dict_l)
    actual_vals_df_l = DataFrame(index=c2_vals, data=actual_vals_dict_l)
    percent_error_df_linear_l = DataFrame(
        index=c2_vals, data=percent_error_dict_linear_l
    )
    percent_error_df_rbf_l = DataFrame(index=c2_vals, data=percent_error_dict_rbf_l)

    linear_surrogate_vals_df_m = DataFrame(
        index=c2_vals, data=linear_surrogate_vals_dict_m
    )
    RBF_surrogate_vals_df_m = DataFrame(index=c2_vals, data=RBF_surrogate_vals_dict_m)
    actual_vals_df_m = DataFrame(index=c2_vals, data=actual_vals_dict_m)
    percent_error_df_linear_m = DataFrame(
        index=c2_vals, data=percent_error_dict_linear_m
    )
    percent_error_df_rbf_m = DataFrame(index=c2_vals, data=percent_error_dict_rbf_m)

    linear_surrogate_vals_df_h = DataFrame(
        index=c2_vals, data=linear_surrogate_vals_dict_h
    )
    RBF_surrogate_vals_df_h = DataFrame(index=c2_vals, data=RBF_surrogate_vals_dict_h)
    actual_vals_df_h = DataFrame(index=c2_vals, data=actual_vals_dict_h)
    percent_error_df_linear_h = DataFrame(
        index=c2_vals, data=percent_error_dict_linear_h
    )
    percent_error_df_rbf_h = DataFrame(index=c2_vals, data=percent_error_dict_rbf_h)

    # fig1, (ax1,ax2, ax3) = plt.subplots(1,3)

    # linear_plot = ax1.pcolor(c1_vals, c2_vals, linear_surrogate_vals_df)
    # ax2.pcolor(c1_vals,c2_vals,RBF_surrogate_vals_df)
    # ax3.pcolor(c1_vals,c2_vals,actual_vals_df)

    # fig1.colorbar(linear_plot, ax=[ax1,ax2,ax3])

    # plt.show()

    fig2, ((ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(2, 3)
    percent_error_plot_linear_l = ax4.pcolor(
        c1_vals, c2_vals, percent_error_df_linear_l
    )
    percent_error_plot_linear_m = ax5.pcolor(
        c1_vals, c2_vals, percent_error_df_linear_m
    )
    percent_error_plot_linear_h = ax6.pcolor(
        c1_vals, c2_vals, percent_error_df_linear_h
    )

    percent_error_plot_rbf_l = ax7.pcolor(c1_vals, c2_vals, percent_error_df_rbf_l)
    percent_error_plot_rbf_m = ax8.pcolor(c1_vals, c2_vals, percent_error_df_rbf_m)
    percent_error_plot_rbf_h = ax9.pcolor(c1_vals, c2_vals, percent_error_df_rbf_h)

    for ax in (ax4, ax5, ax6):
        ax.plot([50, 80], [80, 80], "m-")
        ax.plot([50, 80], [110, 110], "m-")
        ax.plot([50, 50], [80, 110], "m-")
        ax.plot([80, 80], [80, 110], "m-")

    # ax4.plot([73], [108], "ms", markersize=7)
    # ax5.plot([73], [108], "ms", markersize=7)

    for ax in (ax4, ax5, ax6, ax7, ax8, ax9):
        ax.tick_params(direction="in", labelsize=10)
    for ax in (ax4, ax5, ax6):
        ax.set_title(
            "Percent Error (%)\nLinear Surrogate", fontsize=12, fontweight="bold"
        )
    for ax in (ax7, ax8, ax9):
        ax.set_title("Percent Error (%)\nRBF Surrogate", fontsize=12, fontweight="bold")
        ax.set_xlabel(
            "Lithium Concentration\n in Membrane (mM)",
            fontsize=12,
            fontweight="bold",
        )
    for ax in (ax4, ax7):
        ax.set_ylabel(
            "Cobalt Concentration\nin Membrane (mM)",
            fontsize=12,
            fontweight="bold",
        )

    fig2.colorbar(percent_error_plot_linear_l, ax=ax4)
    fig2.colorbar(percent_error_plot_linear_m, ax=ax5)
    fig2.colorbar(percent_error_plot_linear_h, ax=ax6)
    fig2.colorbar(percent_error_plot_rbf_l, ax=ax7)
    fig2.colorbar(percent_error_plot_rbf_m, ax=ax8)
    fig2.colorbar(percent_error_plot_rbf_h, ax=ax9)

    plt.show()


def evaluate_surrogate_two_salt(conc1, conc2, chi, surrogate_obj, var):
    input_dict = {
        "conc_1": conc1,
        "conc_2": conc2,
        "chi": chi,
    }
    input_df = DataFrame(data=input_dict, index=[0])
    surrogate_value = surrogate_obj.evaluate_surrogate(input_df)[var][0] * 1e-7

    return surrogate_value


def evaluate_surrogate_three_salt(conc1, conc2, conc3, chi, surrogate_obj, var):
    input_dict = {
        "conc_1": conc1,
        "conc_2": conc2,
        "conc_3": conc3,
        "chi": chi,
    }
    input_df = DataFrame(data=input_dict, index=[0])
    surrogate_value = surrogate_obj.evaluate_surrogate(input_df)[var][0] * 1e-7

    return surrogate_value


def calculate_true_value_two_salt_d11(c1, c2, chi):
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.71e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 7.31e-6  # m2/h (chloride)

    chi = chi

    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )

    D_11_calc = (
        (z1 * z3 * D1 * D3 - (z1**2) * D1 * D3) * c1
        + (z2 * z3 * D1 * D3 - (z2**2) * D1 * D2) * c2
        + (z3 * D1 * D3 * chi)
    ) / D_denom

    return D_11_calc


def calculate_true_value_two_salt_d12(c1, c2, chi):
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.71e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 7.31e-6  # m2/h (chloride)

    chi = chi

    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )

    D_12_calc = ((z1 * z2 * D1 * D2 - z1 * z2 * D1 * D3) * c1) / D_denom

    return D_12_calc


def calculate_true_value_two_salt_d21(c1, c2, chi):
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.71e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 7.31e-6  # m2/h (chloride)

    chi = chi

    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )

    D_21_calc = ((z1 * z2 * D1 * D2 - z1 * z2 * D2 * D3) * c2) / D_denom

    return D_21_calc


def calculate_true_value_two_salt_d22(c1, c2, chi):
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.71e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 7.31e-6  # m2/h (chloride)

    chi = chi

    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )

    D_22_calc = (
        (z1 * z3 * D2 * D3 - (z1**2) * D1 * D2) * c1
        + (z2 * z3 * D2 * D3 - (z2**2) * D2 * D3) * c2
        + (z3 * D2 * D3 * chi)
    ) / D_denom

    return D_22_calc


def calculate_true_value_two_salt_alpha1(c1, c2, chi):
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.71e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 7.31e-6  # m2/h (chloride)

    chi = chi

    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )

    alpha_1_calc = 1 + (z1 * D1 * chi) / D_denom

    return alpha_1_calc


def calculate_true_value_two_salt_alpha2(c1, c2, chi):
    z1 = 1
    z2 = 2
    z3 = -1

    D1 = 3.71e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 7.31e-6  # m2/h (chloride)

    chi = chi

    D_denom = (
        (((z1**2) * D1 - z1 * z3 * D3) * c1)
        + (((z2**2) * D2 - z2 * z3 * D3) * c2)
        - (z3 * D3 * chi)
    )

    alpha_2_calc = 1 + (z2 * D2 * chi) / D_denom

    return alpha_2_calc


def calculate_true_value_three_salt(c1, c2, c3):
    z1 = 1  # m2/h (lithium)
    z2 = 2  # m2/h (cobalt)
    z3 = 3  # m2/h (aluminum)
    z4 = -1  # m2/h (chloride)

    D1 = 3.71e-6  # m2/h (lithium)
    D2 = 2.64e-6  # m2/h (cobalt)
    D3 = 2.01e-6  # m2/h (aluminum)
    D4 = 7.31e-6  # m2/h (chloride)

    chi = -140

    D_denom = (
        ((((z1**2) * D1) - (z1 * z4 * D4)) * c1)
        + ((((z2**2) * D2) - (z2 * z4 * D4)) * c2)
        + ((((z3**2) * D3) - (z3 * z4 * D4)) * c3)
        - (z4 * D4 * chi)
    )

    D_11 = (
        (((z1 * z4 * D1 * D4) - ((z1**2) * D1 * D4)) * c1)
        + (((z2 * z4 * D1 * D4) - ((z2**2) * D1 * D2)) * c2)
        + (((z3 * z4 * D1 * D4) - ((z3**2) * D1 * D3)) * c3)
        + (z4 * D1 * D4 * chi)
    ) / D_denom

    return D_11


if __name__ == "__main__":
    main()
